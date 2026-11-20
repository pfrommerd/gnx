//! GNX execution using traced TorchScript modules on CPU or CUDA.
mod lower;
mod regions;

use std::future::Future;
use std::io;
use std::sync::{Arc, LazyLock, Mutex};

use gnx_expr::pass::{CheckSupported, Dce, Pipeline};
use gnx_expr::{Expr, Value};
use gnx_math::{
    ArrayInfo, DataHandle, Device, DeviceId, ExecOpts, ExecutionBoundary, HostBuffer, HostId,
    LocalAllocations, MemoryLocation, ShardStorage, TransferPolicy,
};
use gnx_transforms::ExpandTransforms;
use tch::CModule;

#[doc(hidden)]
pub use gnx_expr as __gnx_expr;
pub use lower::{TorchLowering, trace_if, trace_while, tracing_device};
pub use tch::Tensor;

fn invalid(message: impl Into<Box<dyn std::error::Error + Send + Sync>>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}
fn unsupported(message: impl Into<Box<dyn std::error::Error + Send + Sync>>) -> io::Error {
    io::Error::new(io::ErrorKind::Unsupported, message)
}

#[derive(Clone)]
pub struct TorchDevice {
    target: tch::Device,
    allocations: Arc<LocalAllocations<Mutex<Tensor>>>,
}

impl TorchDevice {
    /// The process-wide CPU runtime and allocation table.
    pub fn cpu() -> io::Result<Self> {
        if cfg!(target_endian = "big") {
            return Err(unsupported(
                "Torch host transfers require a little-endian host",
            ));
        }
        static CPU: LazyLock<TorchDevice> = LazyLock::new(|| TorchDevice {
            target: tch::Device::Cpu,
            allocations: Arc::default(),
        });
        Ok(CPU.clone())
    }

    /// Open the CUDA-visible ordinal. Repeated calls share its allocation table.
    pub fn cuda(ordinal: usize) -> io::Result<Self> {
        if cfg!(target_endian = "big") {
            return Err(unsupported(
                "Torch host transfers require a little-endian host",
            ));
        }
        unsafe extern "C" {
            fn gnx_torch_prepare_cuda_driver() -> *const std::ffi::c_char;
        }
        // SAFETY: the native function returns null or a static NUL-terminated diagnostic.
        let load_error = unsafe { gnx_torch_prepare_cuda_driver() };
        if !load_error.is_null() {
            let reason = unsafe { std::ffi::CStr::from_ptr(load_error) };
            return Err(unsupported(format!(
                "loading LibTorch CUDA: {}",
                reason.to_string_lossy()
            )));
        }
        if !tch::Cuda::is_available() {
            return Err(unsupported(
                "Torch CUDA is unavailable: use CUDA-enabled LibTorch and a visible GPU with a compatible NVIDIA driver",
            ));
        }
        let count = tch::Cuda::device_count();
        if ordinal >= count as usize {
            return Err(invalid(format!(
                "Torch CUDA ordinal {ordinal} is out of range ({count} visible devices)"
            )));
        }
        static DEVICES: Mutex<std::collections::BTreeMap<usize, TorchDevice>> =
            Mutex::new(std::collections::BTreeMap::new());
        let mut devices = DEVICES
            .lock()
            .map_err(|_| io::Error::other("poisoned Torch device table"))?;
        if let Some(device) = devices.get(&ordinal) {
            return Ok(device.clone());
        }
        let target = tch::Device::Cuda(ordinal);
        // Initialize the selected runtime now so initialization errors reach the caller.
        unsafe extern "C" {
            fn gnx_torch_open_cuda(
                ordinal: std::ffi::c_int,
                error: *mut std::ffi::c_char,
                capacity: usize,
            ) -> bool;
        }
        let mut error = [0 as std::ffi::c_char; 1024];
        // SAFETY: native borrows the writable buffer for this call and writes
        // a NUL-terminated diagnostic on failure; it catches C++ exceptions.
        if !unsafe {
            gnx_torch_open_cuda(
                ordinal.try_into().map_err(invalid)?,
                error.as_mut_ptr(),
                error.len(),
            )
        } {
            let msg = unsafe { std::ffi::CStr::from_ptr(error.as_ptr()) };
            return Err(io::Error::other(msg.to_string_lossy().into_owned()));
        }
        let device = Self {
            target,
            allocations: Arc::default(),
        };
        devices.insert(ordinal, device.clone());
        Ok(device)
    }

    pub fn id(&self) -> DeviceId {
        DeviceId::Runtime {
            owner: self.allocations.owner(),
            ordinal: match self.target {
                tch::Device::Cuda(n) => n as u64,
                _ => 0,
            },
        }
    }

    pub fn compile(&self, expr: &Expr) -> io::Result<TorchExecutable> {
        let (projected, boundary) = ExecutionBoundary::project(expr)?;
        let mut modules = Vec::new();
        if let Some(body) = projected {
            let body = Pipeline::new()
                .with(ExpandTransforms)
                .with(regions::StaticRegions)
                .with(Dce)
                .with(CheckSupported::<dyn TorchLowering>::new())
                .run(body)
                .map_err(|e| {
                    if e.pass == "check_supported" {
                        unsupported(e)
                    } else {
                        invalid(e)
                    }
                })?;
            modules = lower::trace(&body, self.target)?;
        }
        Ok(TorchExecutable {
            device: self.clone(),
            signature: expr.clone(),
            boundary,
            modules,
        })
    }

    fn resolve(&self, data: &DataHandle) -> io::Result<Option<Tensor>> {
        let Some(t) = data
            .single_allocation()
            .and_then(|r| self.allocations.resolve(r))
        else {
            return Ok(None);
        };
        let t = t
            .lock()
            .map_err(|_| io::Error::other("poisoned Torch allocation"))?
            .shallow_clone();
        lower::check(&t, data.array_info(), self.target)?;
        Ok(Some(t))
    }

    fn copy_from_host(&self, data: &DataHandle, policy: TransferPolicy) -> io::Result<Tensor> {
        if policy == TransferPolicy::RequireZeroCopy {
            return Err(unsupported("Torch host import requires AllowCopy"));
        }
        let host = data.clone().into_host(policy)?;
        let (dims, kind) = lower::layout(host.info())?;
        Tensor::f_from_data_size(host.bytes(), &dims, kind)
            .and_then(|v| v.f_to_device(self.target))
            .map_err(io::Error::other)
    }

    fn retain(&self, t: Tensor, info: ArrayInfo) -> io::Result<DataHandle> {
        lower::check(&t, &info, self.target)?;
        let location = match self.target {
            tch::Device::Cuda(_) => MemoryLocation::Device {
                host: HostId::local(),
                device: self.id(),
                kind: None,
            },
            _ => MemoryLocation::host(HostId::local()),
        };
        DataHandle::single(
            info,
            location,
            ShardStorage::Reference(self.allocations.insert(Mutex::new(t))),
        )
    }

    pub fn import(&self, data: &DataHandle, policy: TransferPolicy) -> io::Result<DataHandle> {
        if self.resolve(data)?.is_some() {
            return Ok(data.clone());
        }
        self.retain(self.copy_from_host(data, policy)?, data.info())
    }

    pub fn download(&self, data: &DataHandle) -> io::Result<HostBuffer> {
        let Some(t) = self.resolve(data)? else {
            return data.to_host();
        };
        HostBuffer::try_fill(data.info(), |dst| {
            t.f_resolve_conj()
                .and_then(|v| v.f_resolve_neg()?.f_copy_data_u8(dst, t.numel()))
                .map_err(io::Error::other)
        })
    }
}

/// Retains compiled modules; `run` never traces or recompiles the GNX expression.
pub struct TorchExecutable {
    device: TorchDevice,
    signature: Expr,
    boundary: ExecutionBoundary,
    modules: Vec<(CModule, ArrayInfo)>,
}

impl TorchExecutable {
    /// Inspect or save the native TorchScript modules in computed-output order.
    /// Each takes the projected captures followed by projected inputs. Forwarded
    /// arguments and duplicate results are reconstructed by `run`.
    pub fn modules(&self) -> impl ExactSizeIterator<Item = &CModule> {
        self.modules.iter().map(|(module, _)| module)
    }

    pub fn run(&self, args: &[Value], opts: ExecOpts) -> io::Result<Vec<Value>> {
        ExecutionBoundary::validate_arguments(&self.signature, args)?;
        let inputs = self
            .boundary
            .arguments(args)
            .map(|v| {
                let d = v?
                    .downcast::<DataHandle>()
                    .map_err(|_| invalid("expected concrete array argument"))?;
                match self.device.resolve(d)? {
                    Some(t) => Ok(t),
                    None => self.device.copy_from_host(d, opts.transfers),
                }
            })
            .collect::<io::Result<Vec<_>>>()?;
        let computed = self
            .modules
            .iter()
            .map(|(m, info)| {
                let result = m.forward_ts(&inputs).map_err(io::Error::other)?;
                Ok(Value::new(self.device.retain(result, info.clone())?))
            })
            .collect::<io::Result<Vec<_>>>()?;
        self.boundary.finish(args, &computed)
    }
}

impl Device for TorchDevice {
    type Outputs = Vec<Value>;
    fn execute(
        &self,
        expr: Expr,
        args: impl IntoIterator<Item = Value>,
        opts: ExecOpts,
    ) -> impl Future<Output = io::Result<Vec<Value>>> + Send {
        let args: Vec<_> = args.into_iter().collect();
        let device = self.clone();
        async move {
            let (tx, rx) = futures_channel::oneshot::channel();
            std::thread::Builder::new()
                .name("gnx-torch".into())
                .spawn(move || {
                    let _ = tx.send(
                        device
                            .compile(&expr)
                            .and_then(|program| program.run(&args, opts)),
                    );
                })?;
            rx.await
                .map_err(|_| io::Error::other("Torch execution worker stopped"))?
        }
    }
}

//! GNX execution using traced TorchScript modules on CPU or CUDA.
mod lower;
mod regions;

use std::future::Future;
use std::io;
use std::sync::{Arc, Mutex, OnceLock};

use gnx_base::cache::Cache;
use gnx_base::device::run_blocking;
use gnx_base::pass::{
    CheckSupported, Dce, Pass, PassError, Pipeline, PipelineError, RuntimePipeline,
};
use gnx_expr::{RootExpr, Value};
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
    cache: Cache,
    pipeline: Arc<RuntimePipeline<TorchModules, Vec<CModule>>>,
    joint_pipeline: Arc<RuntimePipeline<TorchModules, Vec<CModule>>>,
}

impl TorchDevice {
    /// The process-wide CPU runtime and allocation table.
    ///
    /// The first call decides the cache compiled programs are kept in; later
    /// calls return the same device, whatever cache they are given.
    pub fn cpu(cache: Cache) -> io::Result<Self> {
        if cfg!(target_endian = "big") {
            return Err(unsupported(
                "Torch host transfers require a little-endian host",
            ));
        }
        static CPU: OnceLock<TorchDevice> = OnceLock::new();
        let device = CPU.get_or_init(|| TorchDevice {
            target: tch::Device::Cpu,
            pipeline: Arc::new(pipeline(tch::Device::Cpu, false)),
            joint_pipeline: Arc::new(pipeline(tch::Device::Cpu, true)),
            allocations: Arc::default(),
            cache,
        });
        Ok(device.clone())
    }

    /// Open the CUDA-visible ordinal. Repeated calls share its allocation table.
    /// The cache is that of the first call for the ordinal.
    pub fn cuda(ordinal: usize, cache: Cache) -> io::Result<Self> {
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
            pipeline: Arc::new(pipeline(target, false)),
            joint_pipeline: Arc::new(pipeline(target, true)),
            allocations: Arc::default(),
            cache,
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

    /// Lower `expr` and trace it into TorchScript modules for this device.
    /// Both are skipped if the device's cache holds the modules from an
    /// earlier compilation.
    pub async fn compile(&self, expr: &Arc<RootExpr>) -> io::Result<TorchExecutable> {
        self.compile_outputs(expr, false).await
    }

    /// Compile all results into one method, so shared subexpressions execute
    /// once. A graph with several computed results returns a TorchScript tuple.
    pub async fn compile_joint(&self, expr: &Arc<RootExpr>) -> io::Result<TorchExecutable> {
        self.compile_outputs(expr, true).await
    }

    async fn compile_outputs(
        &self,
        expr: &Arc<RootExpr>,
        joint: bool,
    ) -> io::Result<TorchExecutable> {
        let (projected, boundary) = ExecutionBoundary::project(expr.view())?;
        let mut modules = Arc::new(Vec::new());
        let mut outputs = Vec::new();
        let mut capacities = std::collections::HashMap::new();
        if let Some(body) = projected {
            let view = body.view();
            for (i, p) in view.closure_inputs().chain(view.inputs()).enumerate() {
                if !p.info().symbols().is_empty() {
                    capacities.insert(i, lower::layout(&lower::metadata(p.info_ref())?)?.0);
                }
            }
            outputs = (view.outputs().iter())
                .map(|&var| lower::metadata(view.value_info(var).expect("checked SSA output")))
                .collect::<io::Result<Vec<_>>>()?;
            modules = (if joint {
                &self.joint_pipeline
            } else {
                &self.pipeline
            })
            .run(body, &self.cache)
            .await
            .map_err(pipeline_error)?;
        }
        Ok(TorchExecutable {
            device: self.clone(),
            signature: expr.clone(),
            boundary,
            capacities,
            outputs,
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

    pub fn download_blocking(&self, data: &DataHandle) -> io::Result<HostBuffer> {
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
    signature: Arc<RootExpr>,
    boundary: ExecutionBoundary,
    capacities: std::collections::HashMap<usize, Vec<i64>>,
    outputs: Vec<ArrayInfo>,
    modules: Arc<Vec<CModule>>,
}

impl TorchExecutable {
    /// Inspect or save the native TorchScript modules in computed-output order.
    /// Each takes the projected captures followed by projected inputs. Forwarded
    /// arguments and duplicate results are reconstructed by `run`.
    pub fn modules(&self) -> impl ExactSizeIterator<Item = &CModule> {
        self.modules.iter()
    }

    pub fn run(&self, args: &[Value], opts: ExecOpts) -> io::Result<Vec<Value>> {
        let sizes = ExecutionBoundary::bind_arguments(self.signature.view(), args)?;
        let mut inputs = self
            .boundary
            .arguments(args)
            .enumerate()
            .map(|(i, v)| {
                let v = v?;
                let d = if let Ok(buffer) = v.downcast::<gnx_math::BoundedBuffer>() {
                    buffer.storage()
                } else {
                    v.downcast::<DataHandle>()
                        .map_err(|_| invalid("expected concrete array argument"))?
                };
                let tensor = match self.device.resolve(d)? {
                    Some(t) => Ok(t),
                    None => self.device.copy_from_host(d, opts.transfers),
                }?;
                if let Some(cap) = self.capacities.get(&i)
                    && tensor.size() != *cap
                {
                    if opts.transfers == TransferPolicy::RequireZeroCopy {
                        return Err(unsupported("padding bounded inputs requires AllowCopy"));
                    }
                    let pad = cap
                        .iter()
                        .zip(tensor.size())
                        .rev()
                        .flat_map(|(c, n)| [0, *c - n])
                        .collect::<Vec<_>>();
                    return tensor.f_constant_pad_nd(pad).map_err(io::Error::other);
                }
                Ok(tensor)
            })
            .collect::<io::Result<Vec<_>>>()?;
        for scalar in self.boundary.witness_arguments(&sizes) {
            inputs.push(self.device.copy_from_host(&scalar?, opts.transfers)?);
        }
        let results = if self.modules.len() == 1 && self.outputs.len() > 1 {
            let inputs = inputs
                .into_iter()
                .map(tch::IValue::Tensor)
                .collect::<Vec<_>>();
            let tch::IValue::Tuple(items) = self.modules[0]
                .forward_is(&inputs)
                .map_err(io::Error::other)?
            else {
                return Err(invalid("expected TorchScript tuple"));
            };
            items
                .into_iter()
                .map(|v| match v {
                    tch::IValue::Tensor(t) => Ok(t),
                    _ => Err(invalid("expected tensor output")),
                })
                .collect::<io::Result<Vec<_>>>()?
        } else {
            self.modules
                .iter()
                .map(|m| m.forward_ts(&inputs).map_err(io::Error::other))
                .collect::<io::Result<_>>()?
        };
        if results.len() != self.outputs.len() {
            return Err(invalid("unexpected TorchScript output count"));
        }
        let computed = results
            .into_iter()
            .zip(&self.outputs)
            .map(|(mut result, info)| {
                let logical = sizes.resolve(info)?;
                if info.has_bounded_axes() {
                    for (axis, (n, c)) in lower::layout(&logical)?
                        .0
                        .into_iter()
                        .zip(result.size())
                        .enumerate()
                    {
                        if n != c {
                            result = result
                                .f_narrow(axis as i64, 0, n)
                                .map_err(io::Error::other)?;
                        }
                    }
                }
                Ok(Value::new(self.device.retain(result, logical)?))
            })
            .collect::<io::Result<Vec<_>>>()?;
        self.boundary.finish(args, &computed)
    }
}

impl Device for TorchDevice {
    type Outputs = Vec<Value>;
    async fn upload(&self, data: &DataHandle, policy: TransferPolicy) -> io::Result<DataHandle> {
        let (device, data) = (self.clone(), data.clone());
        run_blocking("gnx-torch", move || device.import(&data, policy)).await
    }

    async fn download(&self, data: &DataHandle) -> io::Result<HostBuffer> {
        let (device, data) = (self.clone(), data.clone());
        run_blocking("gnx-torch", move || device.download_blocking(&data)).await
    }

    fn execute(
        &self,
        expr: Arc<RootExpr>,
        args: impl IntoIterator<Item = Value>,
        opts: ExecOpts,
    ) -> impl Future<Output = io::Result<Vec<Value>>> + Send {
        let args: Vec<_> = args.into_iter().collect();
        let device = self.clone();
        run_blocking("gnx-torch", move || {
            futures_executor::block_on(device.compile_joint(&expr))?.run(&args, opts)
        })
    }
}

fn pipeline(target: tch::Device, joint: bool) -> RuntimePipeline<TorchModules, Vec<CModule>> {
    Pipeline::new()
        .with(ExpandTransforms)
        .with(regions::StaticRegions)
        .with(Dce)
        .with(CheckSupported::<dyn TorchLowering>::new())
        .finalize(TraceTorch { target, joint })
        .on_runtime("torch.load", move |traced| {
            traced
                .modules
                .iter()
                .map(|bytes| Ok(CModule::load_data_on_device(&mut &bytes[..], target)?))
                .collect()
        })
}

/// Traced programs: one serialized TorchScript module per computed output.
#[derive(rkyv::Archive, rkyv::Serialize)]
pub struct TorchModules {
    pub modules: Vec<Vec<u8>>,
}

/// The final stage of compilation: traces the expression on `target`.
pub struct TraceTorch {
    pub target: tch::Device,
    pub joint: bool,
}

impl gnx_base::anyhash::Hash for TraceTorch {
    fn hash<H: gnx_base::anyhash::HasherWrite>(&self, state: &mut H) {
        // Tracing runs on the target, and a module may depend on it.
        state.write_str(&format!("{:?}", self.target));
        state.write_u8(self.joint as u8);
    }
}

impl Pass<TorchModules> for TraceTorch {
    fn name(&self) -> &'static str {
        "torch.trace"
    }

    fn run(&self, expr: Arc<RootExpr>) -> Result<TorchModules, PassError> {
        lower::trace(expr.view(), self.target, self.joint)
            .and_then(|modules| (modules.iter()).map(lower::save).collect())
            .map(|modules| TorchModules { modules })
            .map_err(|error| PassError::new(self.name(), error))
    }
}

fn pipeline_error(error: PipelineError) -> io::Error {
    match error {
        PipelineError::Pass(error) if error.pass == "check_supported" => unsupported(error),
        // Tracing reports the kind of its own failures.
        PipelineError::Pass(error) => match error.source.downcast_ref::<io::Error>() {
            Some(source) => io::Error::new(source.kind(), error),
            None => invalid(error),
        },
        PipelineError::Cache(error) => io::Error::other(error),
    }
}

pub(crate) mod cpu;
mod lower;
mod pipeline;

use std::fmt;
use std::future::Future;
use std::io::{self, ErrorKind};
use std::sync::{Arc, Mutex};

use cpu::{Buffer, Client, Executable};
use futures_channel::oneshot;
use gnx_expr::{Expr, Value};
use gnx_math::{
    ArrayInfo, DataHandle, Device, DeviceId, ExecOpts, ExecutionBoundary, HostBuffer, HostId,
    LocalAllocations, MemoryLocation, ShardStorage, TransferPolicy,
};

#[doc(hidden)]
pub use gnx_expr as __gnx_expr;
pub use lower::{LoweringContext, XlaLowering};
pub use pipeline::pipeline;
pub use xla::XlaOp;

#[derive(Clone)]
pub struct XlaDevice(Arc<CpuRuntime>);

struct CpuRuntime {
    host: HostId,
    ordinal: u64,
    kind: String,
    client: Arc<Client>,
    allocations: LocalAllocations<Buffer>,
}

impl XlaDevice {
    /// The process's default single-device CPU client. This does not load Python.
    pub fn cpu() -> io::Result<Self> {
        static CPU: Mutex<Option<XlaDevice>> = Mutex::new(None);
        let mut cpu = CPU.lock().unwrap();
        if let Some(device) = &*cpu {
            return Ok(device.clone());
        }
        let device = Self::cpu_on(HostId::local())?;
        *cpu = Some(device.clone());
        Ok(device)
    }

    /// Create a CPU runtime with an explicitly supplied identity for this host.
    pub fn cpu_on(host: HostId) -> io::Result<Self> {
        if cfg!(target_endian = "big") {
            return Err(unsupported(
                "XLA host transfers require a little-endian host",
            ));
        }
        let client = Client::new()?;
        let (ordinal, kind) = client.device_info()?;
        Ok(Self(Arc::new(CpuRuntime {
            host,
            ordinal,
            kind,
            client,
            allocations: LocalAllocations::new(),
        })))
    }

    pub fn platform(&self) -> &'static str {
        "cpu"
    }
    pub fn host(&self) -> HostId {
        self.0.host
    }
    pub fn id(&self) -> DeviceId {
        DeviceId::Runtime {
            owner: self.0.allocations.owner(),
            ordinal: self.0.ordinal,
        }
    }
    pub fn hardware_kind(&self) -> &str {
        &self.0.kind
    }
    pub fn ordinal(&self) -> u64 {
        self.0.ordinal
    }

    pub fn compile(&self, expr: &Expr) -> io::Result<XlaExecutable> {
        let (projected, boundary) = ExecutionBoundary::project(expr)?;
        let program = projected
            .map(|expr| {
                let output = expr
                    .outputs()
                    .iter()
                    .map(|var| {
                        lower::metadata(expr.value_info(var).expect("checked SSA output")).cloned()
                    })
                    .collect::<io::Result<Vec<_>>>()?;
                let native_program = pipeline().run(expr).map_err(pass_error)?;
                let bytes = native(native_program.proto().to_bytes())?;
                let executable = self.0.client.compile(&bytes)?;
                Ok::<_, io::Error>((executable, output))
            })
            .transpose()?;
        Ok(XlaExecutable {
            device: self.clone(),
            signature: expr.clone(),
            boundary,
            program,
        })
    }

    #[cfg(feature = "mesh")]
    pub(crate) fn compile_partition(&self, bytes: &[u8]) -> io::Result<Executable> {
        self.0.client.compile(bytes)
    }

    #[cfg(feature = "mesh")]
    pub(crate) fn run_partition(
        &self,
        executable: &Executable,
        inputs: &[DataHandle],
        output: &[ArrayInfo],
    ) -> io::Result<Vec<DataHandle>> {
        let inputs = inputs
            .iter()
            .map(|data| self.import_native(data, TransferPolicy::AllowCopy))
            .collect::<io::Result<Vec<_>>>()?;
        self.run_native(executable, &inputs, output)
    }

    fn run_native(
        &self,
        executable: &Executable,
        inputs: &[Arc<Buffer>],
        output: &[ArrayInfo],
    ) -> io::Result<Vec<DataHandle>> {
        executable
            .run(inputs, output.len())?
            .into_iter()
            .zip(output)
            .map(|(buffer, info)| self.retain(info.clone(), buffer))
            .collect()
    }

    /// Borrow a compatible packed host allocation, or retain an allocation
    /// that this client owns. Incompatible storage requires `AllowCopy`.
    pub fn import(&self, data: &DataHandle, policy: TransferPolicy) -> io::Result<DataHandle> {
        if self.resolve(data)?.is_some() {
            return Ok(data.clone());
        }
        let buffer = self.host_import(data, policy)?;
        self.retain(data.info(), buffer)
    }

    /// Explicit readback. Unknown owners are never contacted by this backend.
    pub fn download(&self, data: &DataHandle) -> io::Result<HostBuffer> {
        if let Ok(host) = data.to_host() {
            return Ok(host);
        }
        let buffer = self
            .resolve(data)?
            .ok_or_else(|| unsupported("this XLA client does not own the allocation"))?;
        buffer.download(data.info())
    }

    fn resolve(&self, data: &DataHandle) -> io::Result<Option<Arc<Buffer>>> {
        let buffer = data
            .single_allocation()
            .and_then(|reference| self.0.allocations.resolve(reference));
        if let Some(ref buffer) = buffer {
            buffer.check(data.array_info())?;
        }
        Ok(buffer)
    }

    fn import_native(&self, data: &DataHandle, policy: TransferPolicy) -> io::Result<Arc<Buffer>> {
        if let Some(buffer) = self.resolve(data)? {
            return Ok(buffer);
        }
        self.host_import(data, policy).map(Arc::new)
    }

    fn host_import(&self, data: &DataHandle, policy: TransferPolicy) -> io::Result<Buffer> {
        self.0
            .client
            .import(data.clone().into_host(policy)?, policy)
    }

    fn retain(&self, info: ArrayInfo, buffer: Buffer) -> io::Result<DataHandle> {
        buffer.check(&info)?;
        DataHandle::single(
            info,
            MemoryLocation::host(self.host()),
            ShardStorage::Reference(self.0.allocations.insert(buffer)),
        )
    }
}

impl fmt::Debug for XlaDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("XlaDevice")
            .field("platform", &self.platform())
            .field("host", &self.host())
            .field("id", &self.id())
            .finish()
    }
}

/// A compiled expression can run repeatedly on its owning CPU runtime.
pub struct XlaExecutable {
    device: XlaDevice,
    signature: Expr,
    boundary: ExecutionBoundary,
    program: Option<(Executable, Vec<ArrayInfo>)>,
}

impl XlaExecutable {
    pub fn run(&self, args: &[Value], opts: ExecOpts) -> io::Result<Vec<Value>> {
        ExecutionBoundary::validate_arguments(&self.signature, args)?;
        let Some((program, output_info)) = &self.program else {
            return self.boundary.finish(args, &[]);
        };
        let inputs = self
            .boundary
            .arguments(args)
            .map(|value| {
                let data = value?
                    .downcast::<DataHandle>()
                    .map_err(|_| invalid("XLA argument is not a concrete array"))?;
                self.device.import_native(data, opts.transfers)
            })
            .collect::<io::Result<Vec<_>>>()?;
        let computed: Vec<_> = self
            .device
            .run_native(program, &inputs, output_info)?
            .into_iter()
            .map(Value::new)
            .collect();
        self.boundary.finish(args, &computed)
    }
}

impl Device for XlaDevice {
    type Outputs = Vec<Value>;

    fn execute(
        &self,
        expr: Expr,
        args: impl IntoIterator<Item = Value>,
        opts: ExecOpts,
    ) -> impl Future<Output = io::Result<Self::Outputs>> + Send {
        let args: Vec<_> = args.into_iter().collect();
        let device = self.clone();
        async move {
            let (sender, receiver) = oneshot::channel();
            std::thread::Builder::new()
                .name("gnx-xla".into())
                .spawn(move || {
                    let result = device
                        .compile(&expr)
                        .and_then(|compiled| compiled.run(&args, opts));
                    let _ = sender.send(result);
                })?;
            receiver
                .await
                .map_err(|_| io::Error::other("XLA execution worker stopped without a result"))?
        }
    }
}

fn invalid(message: impl Into<Box<dyn std::error::Error + Send + Sync>>) -> io::Error {
    io::Error::new(ErrorKind::InvalidInput, message)
}
fn unsupported(message: impl Into<Box<dyn std::error::Error + Send + Sync>>) -> io::Error {
    io::Error::new(ErrorKind::Unsupported, message)
}
/// Preserve the kind of an I/O error raised inside a pass.
pub(crate) fn pass_error(error: gnx_expr::pass::PassError) -> io::Error {
    let kind = error
        .source
        .downcast_ref::<io::Error>()
        .map_or(ErrorKind::Other, io::Error::kind);
    io::Error::new(kind, error)
}
fn native<T>(result: xla::Result<T>) -> io::Result<T> {
    result.map_err(io::Error::other)
}

#[cfg(test)]
mod tests {
    use super::*;
    use gnx_math::{ByteOrder, DType, Lease, MemoryView, Shape, StridedLayout};

    #[test]
    fn cpu_aliases_an_aligned_snapshot_and_retains_its_storage() {
        let device = XlaDevice::cpu().unwrap();
        let info = ArrayInfo::new(Shape::fixed([2]), DType::F32);
        let values = [1_f32, 2.]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let host = HostBuffer::copy_from_slice(info, &values).unwrap();
        let address = host.bytes().as_ptr();
        let source = DataHandle::new(host);
        let buffer = device
            .import(&source, TransferPolicy::RequireZeroCopy)
            .unwrap();
        assert_eq!(
            device.resolve(&buffer).unwrap().unwrap().address().unwrap(),
            address
        );
        drop(source);
        assert_eq!(device.download(&buffer).unwrap().bytes(), values);
    }

    #[test]
    fn strict_import_rejects_a_misaligned_view_before_copying() {
        let device = XlaDevice::cpu().unwrap();
        let source =
            HostBuffer::copy_from_slice(ArrayInfo::new(Shape::fixed([5]), DType::U8), &[0; 5])
                .unwrap();
        let info = ArrayInfo::new(Shape::fixed([1]), DType::F32);
        let address = source.bytes().as_ptr().cast_mut();
        // SAFETY: the view reads four initialized bytes after the first byte,
        // and the lease keeps the immutable source allocation alive.
        let view = unsafe {
            MemoryView::new(
                info.clone(),
                address,
                Some(5),
                StridedLayout {
                    byte_offset: 1,
                    byte_strides: vec![4],
                    byte_order: ByteOrder::Little,
                },
                true,
                None,
                Lease::on_drop(move || drop(source)),
            )
        }
        .unwrap();
        let data = DataHandle::single(
            info,
            MemoryLocation::host(device.host()),
            ShardStorage::Local {
                view,
                allocation: None,
            },
        )
        .unwrap();
        assert_eq!(
            device
                .import(&data, TransferPolicy::RequireZeroCopy)
                .unwrap_err()
                .kind(),
            ErrorKind::Unsupported
        );
        let copied = device.import(&data, TransferPolicy::AllowCopy).unwrap();
        assert_eq!(device.download(&copied).unwrap().bytes(), [0; 4]);
    }
}

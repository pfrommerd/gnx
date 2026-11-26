//! GNX execution on XLA's CPU and CUDA PJRT runtimes.
mod lower;

use std::fmt;
use std::future::Future;
use std::io::{self, ErrorKind};
use std::sync::{Arc, Mutex};

use futures_channel::oneshot;
use gnx_base::cache::Cache;
use gnx_base::pass::{
    CheckSupported, Dce, FinalPipeline, Pass, PassError, Pipeline, PipelineError, RuntimePipeline,
};
use gnx_expr::{RootExpr, Value};
use gnx_math::{
    ArrayInfo, DataHandle, Device, DeviceId, ExecOpts, ExecutionBoundary, HostBuffer, HostId,
    LocalAllocations, MemoryLocation, ShardStorage, TransferPolicy,
};
use gnx_transforms::ExpandTransforms;
use xla::pjrt::{Buffer, Client, Executable};

#[doc(hidden)]
pub use gnx_expr as __gnx_expr;
pub use lower::{LoweringContext, XlaLowering};
pub use xla::XlaOp;

/// A lowered program: a serialized HLO module.
#[derive(rkyv::Archive, rkyv::Serialize)]
pub struct HloModule {
    pub bytes: Vec<u8>,
}

/// The portable lowering stage of the XLA pipeline.
pub struct LowerXla;

impl gnx_base::anyhash::Hash for LowerXla {
    fn hash<H: gnx_base::anyhash::HasherWrite>(&self, _: &mut H) {}
}

impl Pass<HloModule> for LowerXla {
    fn name(&self) -> &'static str {
        "xla.lower"
    }

    fn run(&self, expr: Arc<RootExpr>) -> Result<HloModule, PassError> {
        lower::lower(expr.view())
            .and_then(|program| native(program.proto().to_bytes()))
            .map(|bytes| HloModule { bytes })
            .map_err(|error| PassError::new(self.name(), error))
    }
}

/// The portable prefix, also available for inspecting HLO without a client.
pub fn hlo_pipeline() -> FinalPipeline<HloModule> {
    Pipeline::new()
        .with(ExpandTransforms)
        .with(Dce)
        .with(CheckSupported::<dyn XlaLowering>::new())
        .finalize(LowerXla)
}

/// Inspect the portable lowering stages without opening a PJRT client.
pub fn pipeline() -> &'static FinalPipeline<HloModule> {
    static HLO: std::sync::LazyLock<FinalPipeline<HloModule>> =
        std::sync::LazyLock::new(hlo_pipeline);
    &HLO
}

#[derive(Clone)]
pub struct XlaDevice(Arc<Runtime>);

#[derive(Clone, Copy)]
enum Platform {
    Cpu,
    Cuda,
}

struct Runtime {
    platform: Platform,
    host: HostId,
    ordinal: u64,
    kind: String,
    client: Arc<Client>,
    allocations: LocalAllocations<Buffer>,
    cache: Cache,
    pipeline: RuntimePipeline<HloModule, Executable>,
}

impl XlaDevice {
    /// The process's default single-device CPU client. This does not load Python.
    ///
    /// The first call decides the cache compiled programs are kept in; later
    /// calls return the same device, whatever cache they are given.
    pub fn cpu(cache: Cache) -> io::Result<Self> {
        static CPU: Mutex<Option<XlaDevice>> = Mutex::new(None);
        let mut cpu = CPU.lock().unwrap();
        if let Some(device) = &*cpu {
            return Ok(device.clone());
        }
        let device = Self::cpu_on(HostId::local(), cache)?;
        *cpu = Some(device.clone());
        Ok(device)
    }

    /// Create a CPU runtime with an explicitly supplied identity for this host.
    pub fn cpu_on(host: HostId, cache: Cache) -> io::Result<Self> {
        Self::from_client(host, Platform::Cpu, Client::new()?, cache)
    }

    /// Open the CUDA-visible GPU with this local ordinal. Repeated calls for
    /// the same ordinal share a client and its allocation table.
    /// The cache is that of the first call for the ordinal.
    pub fn cuda(ordinal: usize, cache: Cache) -> io::Result<Self> {
        use std::collections::BTreeMap;
        static DEVICES: Mutex<BTreeMap<usize, XlaDevice>> = Mutex::new(BTreeMap::new());
        let mut devices = DEVICES.lock().unwrap();
        if let Some(device) = devices.get(&ordinal) {
            return Ok(device.clone());
        }
        let client = Client::cuda(ordinal)?;
        let device = Self::from_client(HostId::local(), Platform::Cuda, client, cache)?;
        devices.insert(ordinal, device.clone());
        Ok(device)
    }

    fn from_client(
        host: HostId,
        platform: Platform,
        client: Arc<Client>,
        cache: Cache,
    ) -> io::Result<Self> {
        if cfg!(target_endian = "big") {
            return Err(unsupported(
                "XLA host transfers require a little-endian host",
            ));
        }
        let (ordinal, kind) = client.device_info()?;
        let compiler = client.clone();
        let pipeline =
            hlo_pipeline().on_runtime("xla.compile", move |hlo| Ok(compiler.compile(&hlo.bytes)?));
        Ok(Self(Arc::new(Runtime {
            platform,
            host,
            ordinal,
            kind,
            client,
            allocations: LocalAllocations::new(),
            pipeline,
            cache,
        })))
    }

    /// The native compilation pipeline for a projected computation. Its final
    /// `xla.compile` stage produces a loaded executable owned by this client.
    pub fn pipeline(&self) -> &RuntimePipeline<HloModule, Executable> {
        &self.0.pipeline
    }

    pub fn platform(&self) -> &'static str {
        match self.0.platform {
            Platform::Cpu => "cpu",
            Platform::Cuda => "cuda",
        }
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

    /// Lower `expr` and compile it for this device. Lowering is skipped if
    /// the device's cache holds the program from an earlier compilation.
    pub async fn compile(&self, expr: &Arc<RootExpr>) -> io::Result<XlaExecutable> {
        let (projected, boundary) = ExecutionBoundary::project(expr.view())?;
        let mut bounded_inputs = Vec::new();
        let program = match projected {
            Some(expr) => {
                let view = expr.view();
                for (i, p) in view.closure_inputs().chain(view.inputs()).enumerate() {
                    if !p.info().symbols().is_empty() {
                        bounded_inputs.push((i, lower::metadata(p.info_ref())?.capacity()));
                    }
                }
                let output = (view.outputs().iter())
                    .map(|&var| lower::metadata(view.value_info(var).expect("checked SSA output")))
                    .collect::<io::Result<Vec<_>>>()?;
                Some((
                    self.0
                        .pipeline
                        .run(expr, &self.0.cache)
                        .await
                        .map_err(pipeline_error)?,
                    output,
                ))
            }
            None => None,
        };
        Ok(XlaExecutable {
            device: self.clone(),
            signature: expr.clone(),
            boundary,
            bounded_inputs,
            program,
        })
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
            .map(|(buffer, info)| {
                self.retain(
                    if info.has_bounded_axes() {
                        info.capacity()
                    } else {
                        info.clone()
                    },
                    buffer,
                )
            })
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
        HostBuffer::try_fill(data.info(), |bytes| buffer.download_into(bytes))
    }

    fn resolve(&self, data: &DataHandle) -> io::Result<Option<Arc<Buffer>>> {
        let buffer = data
            .single_allocation()
            .and_then(|reference| self.0.allocations.resolve(reference));
        if let Some(ref buffer) = buffer {
            let info = data.array_info();
            buffer.check(lower::element(info.dtype()), &lower::dimensions(info)?)?;
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
        let host = data.clone().into_host(policy)?;
        let dims = lower::dimensions(host.info())?;
        let dtype = lower::element(host.info().dtype());
        self.0.client.import(
            host,
            HostBuffer::bytes,
            dtype,
            &dims,
            policy == TransferPolicy::RequireZeroCopy,
        )
    }

    fn retain(&self, info: ArrayInfo, buffer: Buffer) -> io::Result<DataHandle> {
        buffer.check(lower::element(info.dtype()), &lower::dimensions(&info)?)?;
        DataHandle::single(
            info,
            match self.0.platform {
                Platform::Cpu => MemoryLocation::host(self.host()),
                Platform::Cuda => MemoryLocation::Device {
                    host: self.host(),
                    device: self.id(),
                    kind: None,
                },
            },
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

/// A compiled expression can run repeatedly on its owning runtime.
pub struct XlaExecutable {
    device: XlaDevice,
    signature: Arc<RootExpr>,
    boundary: ExecutionBoundary,
    bounded_inputs: Vec<(usize, ArrayInfo)>,
    program: Option<(Arc<Executable>, Vec<ArrayInfo>)>,
}

impl XlaExecutable {
    pub fn run(&self, args: &[Value], opts: ExecOpts) -> io::Result<Vec<Value>> {
        let sizes = ExecutionBoundary::bind_arguments(self.signature.view(), args)?;
        let Some((program, output_info)) = &self.program else {
            return self.boundary.finish(args, &[]);
        };
        let mut inputs = self
            .boundary
            .arguments(args)
            .enumerate()
            .map(|(i, value)| {
                let value = value?;
                let data = if let Ok(buffer) = value.downcast::<gnx_math::BoundedBuffer>() {
                    buffer.storage()
                } else {
                    value
                        .downcast::<DataHandle>()
                        .map_err(|_| invalid("XLA argument is not a concrete array"))?
                };
                if let Some((_, capacity)) =
                    self.bounded_inputs.iter().find(|(index, _)| *index == i)
                    && data.array_info() != capacity
                {
                    if opts.transfers == TransferPolicy::RequireZeroCopy {
                        return Err(unsupported(
                            "padding a compact bounded input requires AllowCopy",
                        ));
                    }
                    let packed = self
                        .device
                        .download(data)?
                        .resize_prefix(capacity.clone())?;
                    return self
                        .device
                        .import_native(&DataHandle::new(packed), opts.transfers);
                }
                self.device.import_native(data, opts.transfers)
            })
            .collect::<io::Result<Vec<_>>>()?;
        for scalar in self.boundary.witness_arguments(&sizes) {
            inputs.push(self.device.import_native(&scalar?, opts.transfers)?);
        }
        let computed: Vec<_> = self
            .device
            .run_native(program, &inputs, output_info)?
            .into_iter()
            .zip(output_info)
            .map(|(data, formal)| {
                if formal.has_bounded_axes() {
                    let logical = sizes.resolve(formal)?;
                    if data.array_info() != &logical {
                        if opts.transfers == TransferPolicy::RequireZeroCopy {
                            return Err(unsupported(
                                "unpadding a bounded result requires AllowCopy",
                            ));
                        }
                        let compact = self.device.download(&data)?.resize_prefix(logical)?;
                        return Ok(Value::new(
                            self.device
                                .import(&DataHandle::new(compact), opts.transfers)?,
                        ));
                    }
                }
                Ok(Value::new(data))
            })
            .collect::<io::Result<_>>()?;
        self.boundary.finish(args, &computed)
    }
}

impl Device for XlaDevice {
    type Outputs = Vec<Value>;

    fn execute(
        &self,
        expr: Arc<RootExpr>,
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
                    let result = futures_executor::block_on(device.compile(&expr))
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
pub(crate) fn pipeline_error(error: PipelineError) -> io::Error {
    match error {
        PipelineError::Pass(error) => pass_error(error),
        PipelineError::Cache(error) => io::Error::other(error),
    }
}
/// Preserve the kind of an I/O error raised inside a pass.
pub(crate) fn pass_error(error: PassError) -> io::Error {
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
    fn repeated_compile_reuses_the_loaded_pjrt_executable() {
        use gnx_base::pass::Builder;
        use gnx_math::Dim;
        let device = XlaDevice::cpu(Cache::in_memory()).unwrap();
        let mut b = Builder::new();
        let s = b.symbol_input();
        let info = gnx_expr::VarInfo::new(ArrayInfo::new(
            [Dim::Bounded { symbol: s, max: 8 }]
                .into_iter()
                .collect::<gnx_base::Shape>(),
            DType::F32,
        ));
        let x = b.input(info.clone());
        let y = b.emit1(gnx_math::ops::add_op(), &[x, x], info);
        let expr = b.finish(vec![y]).build().unwrap();
        let a = futures_executor::block_on(device.compile(&expr)).unwrap();
        let b = futures_executor::block_on(device.compile(&expr)).unwrap();
        assert!(Arc::ptr_eq(&a.program.unwrap().0, &b.program.unwrap().0));
    }

    #[test]
    fn cpu_aliases_an_aligned_snapshot_and_retains_its_storage() {
        let device = XlaDevice::cpu(gnx_base::cache::Cache::in_memory()).unwrap();
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
        let device = XlaDevice::cpu(gnx_base::cache::Cache::in_memory()).unwrap();
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

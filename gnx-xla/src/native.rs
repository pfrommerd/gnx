mod lower;

use std::fmt;
use std::future::Future;
use std::io::{self, ErrorKind};
use std::sync::{Arc, Mutex};

use futures_channel::oneshot;
use gnx_expr::{Expr, Value};
use gnx_math::{
    ArrayInfo, DataHandle, Device, DeviceId, ExecOpts, ExecutionBoundary, HostBuffer, HostId,
    LocalAllocations, MemoryLocation, ShardStorage, TransferPolicy,
};
use xla::{PjRtBuffer, PjRtClient, PjRtLoadedExecutable};

#[doc(hidden)]
pub use inventory as __inventory;
pub use lower::{LoweringContext, Registration, XlaLowering};
pub use xla::XlaOp;

#[derive(Clone)]
pub struct XlaDevice(Arc<CpuRuntime>);

struct CpuRuntime {
    host: HostId,
    ordinal: u64,
    kind: String,
    client: PjRtClient,
    allocations: LocalAllocations<PjRtBuffer>,
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
        let client = native(PjRtClient::cpu())?;
        let devices = client.addressable_devices();
        let [device] = devices.as_slice() else {
            return Err(unsupported(
                "gnx-xla currently requires one addressable CPU device",
            ));
        };
        if client.device_count() != 1 {
            return Err(unsupported(
                "gnx-xla does not support distributed CPU execution",
            ));
        }
        let ordinal =
            u64::try_from(device.id()).map_err(|_| invalid("invalid XLA device ordinal"))?;
        let kind = device.kind();
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
                let native_program = lower::lower(&expr)?;
                let executable = native(self.0.client.compile(&native_program))?;
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
    pub(crate) fn compile_partition(&self, bytes: &[u8]) -> io::Result<PjRtLoadedExecutable> {
        let proto = native(xla::HloModuleProto::parse_proto(bytes, true))?;
        let computation = xla::XlaComputation::from_proto(&proto);
        native(self.0.client.compile(&computation))
    }

    #[cfg(feature = "mesh")]
    pub(crate) fn run_partition(
        &self,
        executable: &PjRtLoadedExecutable,
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
        executable: &PjRtLoadedExecutable,
        inputs: &[Arc<PjRtBuffer>],
        output: &[ArrayInfo],
    ) -> io::Result<Vec<DataHandle>> {
        let mut results = native(executable.execute_b(inputs))?;
        if results.len() != 1 || results[0].len() != output.len() {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                "unexpected XLA replica or output count",
            ));
        }
        results
            .pop()
            .unwrap()
            .into_iter()
            .zip(output)
            .map(|(buffer, info)| self.retain(info.clone(), Arc::new(buffer)))
            .collect()
    }

    /// Copy a packed or strided host array into this PJRT client, or retain a
    /// reference to an allocation it already owns.
    pub fn import(&self, data: &DataHandle, policy: TransferPolicy) -> io::Result<DataHandle> {
        if self.resolve(data)?.is_some() {
            return Ok(data.clone());
        }
        let buffer = self.import_native(data, policy)?;
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
        let literal = native(buffer.to_literal_sync())?;
        let mut bytes = vec![0; literal.size_bytes()];
        native(literal.copy_untyped_to(&mut bytes))?;
        HostBuffer::new(data.info(), bytes)
    }

    fn resolve(&self, data: &DataHandle) -> io::Result<Option<Arc<PjRtBuffer>>> {
        let buffer = data
            .single_allocation()
            .and_then(|reference| self.0.allocations.resolve(reference));
        if let Some(ref buffer) = buffer {
            check_buffer(buffer, data.array_info())?;
        }
        Ok(buffer)
    }

    fn import_native(
        &self,
        data: &DataHandle,
        policy: TransferPolicy,
    ) -> io::Result<Arc<PjRtBuffer>> {
        if let Some(buffer) = self.resolve(data)? {
            return Ok(buffer);
        }
        // xla-rs's host upload uses PJRT kImmutableOnlyDuringCall. It does
        // not expose borrowing or external-reference hooks for other clients.
        if policy == TransferPolicy::RequireZeroCopy {
            return Err(unsupported("xla-rs cannot alias this external allocation"));
        }
        let host = data.to_host()?;
        let dimensions = lower::dimensions(host.info())?
            .into_iter()
            .map(|n| n as usize)
            .collect::<Vec<_>>();
        let devices = self.0.client.addressable_devices();
        Ok(Arc::new(native(self.0.client.buffer_from_host_raw_bytes(
            lower::element(host.info().dtype()),
            host.bytes(),
            &dimensions,
            devices.first(),
        ))?))
    }

    fn retain(&self, info: ArrayInfo, buffer: Arc<PjRtBuffer>) -> io::Result<DataHandle> {
        check_buffer(&buffer, &info)?;
        let buffer = Arc::try_unwrap(buffer)
            .map_err(|_| io::Error::other("unexpected shared native allocation"))?;
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

fn check_buffer(buffer: &PjRtBuffer, info: &ArrayInfo) -> io::Result<()> {
    if native(buffer.on_device_shape())? != lower::shape(info)? {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            "XLA returned mismatched array metadata",
        ));
    }
    Ok(())
}

/// A compiled expression can run repeatedly on its owning CPU runtime.
pub struct XlaExecutable {
    device: XlaDevice,
    signature: Expr,
    boundary: ExecutionBoundary,
    program: Option<(PjRtLoadedExecutable, Vec<ArrayInfo>)>,
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

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(ErrorKind::InvalidInput, message)
}
fn unsupported(message: &'static str) -> io::Error {
    io::Error::new(ErrorKind::Unsupported, message)
}
fn native<T>(result: xla::Result<T>) -> io::Result<T> {
    result.map_err(io::Error::other)
}

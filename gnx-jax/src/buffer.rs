use std::io::{self, ErrorKind};
use std::sync::LazyLock;

use gnx_math::{
    ArrayInfo, DataHandle, DeviceId, DlpackRequest, HostBuffer, HostId, HostMemory,
    LocalAllocations, MemoryLocation, ShardStorage, TransferPolicy,
};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyTuple};

use crate::{JaxDevice, dlpack, jaxpr};

static ALLOCATIONS: LazyLock<LocalAllocations<JaxBuffer>> = LazyLock::new(LocalAllocations::new);

/// A backend adapter handle, not the representation stored in a GNX array.
#[derive(Debug)]
pub struct JaxBuffer {
    info: ArrayInfo,
    device: JaxDevice,
    memory_kind: Option<String>,
    native: Py<PyAny>,
}

impl JaxBuffer {
    pub fn from_native(array: Bound<'_, PyAny>) -> jaxpr::Result<Self> {
        let py = array.py();
        if !array.is_instance(&py.import("jax")?.getattr("Array")?)? {
            return Err(jaxpr::JaxLoweringError::InvalidValue(
                "expected a concrete jax.Array".into(),
            ));
        }
        let devices = array.call_method0("devices")?;
        if devices.len()? != 1 {
            return Err(jaxpr::JaxLoweringError::InvalidValue(
                "only single-device JAX arrays are supported".into(),
            ));
        }
        let native_device = devices.try_iter()?.next().transpose()?.ok_or_else(|| {
            jaxpr::JaxLoweringError::InvalidValue("JAX array has no device".into())
        })?;
        let device = JaxDevice::from_native(native_device)?;
        let memory_kind = array
            .getattr("sharding")?
            .getattr("memory_kind")?
            .extract()?;
        let info = jaxpr::array_info(&array)?;
        Ok(Self {
            info,
            device,
            memory_kind,
            native: array.unbind(),
        })
    }

    pub fn info(&self) -> &ArrayInfo {
        &self.info
    }
    pub fn device(&self) -> &JaxDevice {
        &self.device
    }
    pub fn native<'a, 'py>(&'a self, py: Python<'py>) -> &'a Bound<'py, PyAny> {
        self.native.bind(py)
    }

    /// Retain the allocation in the JAX adapter. CPU allocations additionally
    /// expose a shared view when JAX can export one without copying.
    ///
    /// # Safety
    /// No external alias may mutate or resize the JAX allocation. In
    /// particular a JAX array imported from a mutable framework is not by
    /// itself evidence of immutability.
    pub unsafe fn into_data(self, py: Python<'_>) -> io::Result<DataHandle> {
        let info = self.info.clone();
        let location = if self.device.platform() == "cpu" {
            MemoryLocation::host(HostId::local())
        } else if self.memory_kind.as_deref() == Some("pinned_host") {
            MemoryLocation::Host {
                host: HostId::local(),
                numa_node: None,
                kind: HostMemory::Pinned,
            }
        } else if self.memory_kind.as_deref() == Some("device") {
            MemoryLocation::Device {
                host: HostId::local(),
                device: DeviceId::Runtime {
                    owner: ALLOCATIONS.owner(),
                    ordinal: self.device.ordinal(),
                },
                kind: self.memory_kind.clone(),
            }
        } else {
            MemoryLocation::Unknown
        };
        let native = self.native(py);
        if !native.getattr("is_fully_addressable")?.extract::<bool>()? {
            return Err(io::Error::new(
                ErrorKind::Unsupported,
                "the JAX adapter does not retain non-addressable JAX arrays",
            ));
        }
        native.call_method0("block_until_ready")?;
        // CPU DLPack is immediately host-readable. Other platforms stay opaque
        // until their device/context and synchronization can be established.
        let tensor = if self.device.platform() == "cpu" {
            // SAFETY: caller promises no foreign writes; block_until_ready just completed.
            unsafe {
                dlpack::export(
                    native,
                    DlpackRequest {
                        stream: None,
                        max_version: Some((1, 0)),
                    },
                )
            }
            .ok()
        } else {
            None
        };
        let allocation = ALLOCATIONS.insert(self);
        if let Some(tensor) = tensor {
            // SAFETY: this is the ready, immutable CPU lease obtained above.
            let data = unsafe { tensor.into_data(location, Some(allocation)) }?;
            if data.array_info() != &info {
                return Err(io::Error::new(
                    ErrorKind::InvalidData,
                    "JAX exported unexpected metadata",
                ));
            }
            Ok(data)
        } else {
            DataHandle::single(info, location, ShardStorage::Reference(allocation))
        }
    }

    fn download(&self, py: Python<'_>) -> io::Result<HostBuffer> {
        let numpy = py.import("numpy")?;
        let dtype: &'static str = self.info.dtype().into();
        let packed = numpy
            .call_method1("dtype", (dtype,))?
            .call_method1("newbyteorder", ("<",))?;
        let array = numpy.call_method1("asarray", (self.native(py), packed))?;
        let bytes = array.call_method1("tobytes", ("C",))?;
        HostBuffer::new(
            self.info.clone(),
            bytes
                .cast::<PyBytes>()
                .map_err(PyErr::from)?
                .as_bytes()
                .to_vec(),
        )
    }
}

impl JaxDevice {
    pub(crate) fn single_device_sharding<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        py.import("jax.sharding")?
            .call_method1("SingleDeviceSharding", (self.native(py),))
    }

    /// Explicit host readback for a JAX-owned opaque reference.
    pub fn download(&self, py: Python<'_>, data: &DataHandle) -> io::Result<HostBuffer> {
        if let Ok(host) = data.to_host() {
            return Ok(host);
        }
        data.single_allocation()
            .and_then(|id| ALLOCATIONS.resolve(id))
            .ok_or_else(|| {
                io::Error::new(ErrorKind::Unsupported, "JAX does not own this allocation")
            })?
            .download(py)
    }

    pub fn import_buffer(
        &self,
        py: Python<'_>,
        buffer: DataHandle,
        policy: TransferPolicy,
    ) -> io::Result<JaxBuffer> {
        let info = buffer.info();
        let jax = py.import("jax")?;
        let sharding = self.single_device_sharding(py)?;
        let memory_kind: Option<String> = sharding.getattr("memory_kind")?.extract()?;
        let dtype: &'static str = info.dtype().into();
        let canonical = jax
            .getattr("dtypes")?
            .call_method1("canonicalize_dtype", (dtype,))?;
        if canonical.getattr("name")?.extract::<String>()? != dtype {
            return Err(io::Error::new(
                ErrorKind::Unsupported,
                format!("JAX would change dtype {dtype}; check its dtype configuration"),
            ));
        }
        let owned = buffer
            .single_allocation()
            .and_then(|id| ALLOCATIONS.resolve(id));
        let array = if let Some(native) = owned {
            if self.native(py).eq(native.device.native(py))? && native.memory_kind == memory_kind {
                native.native(py).clone()
            } else if policy == TransferPolicy::AllowCopy {
                jax.call_method1("device_put", (native.native(py), &sharding))?
            } else {
                return Err(io::Error::new(
                    ErrorKind::Unsupported,
                    "moving to this JAX device would require a copy",
                ));
            }
        } else if policy == TransferPolicy::RequireZeroCopy {
            if self.platform() != "cpu"
                || !matches!(buffer.single_shard().map(|s| &s.location), Some(MemoryLocation::Host { host, .. }) if *host == HostId::local())
            {
                return Err(io::Error::new(
                    ErrorKind::Unsupported,
                    "JAX cannot alias this memory location",
                ));
            }
            import_dlpack(py, buffer, &sharding)?
        } else {
            import_host(py, buffer.to_host()?, &sharding, &info)?
        };
        let imported = JaxBuffer::from_native(array).map_err(io::Error::other)?;
        if imported.info != info
            || !self.native(py).eq(imported.device.native(py))?
            || imported.memory_kind != memory_kind
        {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                "JAX import did not preserve array metadata or placement",
            ));
        }
        Ok(imported)
    }
}

fn import_dlpack<'py>(
    py: Python<'py>,
    buffer: DataHandle,
    sharding: &Bound<'py, PyAny>,
) -> io::Result<Bound<'py, PyAny>> {
    let source = buffer.into_dlpack().map_err(|_| {
        io::Error::new(
            ErrorKind::Unsupported,
            "source cannot export a zero-copy DLPack lease",
        )
    })?;
    let address = source.address().ok_or_else(|| {
        io::Error::new(ErrorKind::Unsupported, "JAX needs an addressable CPU view")
    })?;
    // The CPU runtime can copy misaligned input even on the DLPack path.
    if source.device().0 != 1 || !address.is_multiple_of(64) {
        return Err(io::Error::new(
            ErrorKind::Unsupported,
            "JAX needs a 64-byte-aligned CPU view for zero-copy import",
        ));
    }
    // SAFETY: JAX imports into its immutable array API.
    let producer = unsafe { dlpack::producer_with_legacy(py, source) }?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("device", sharding)?;
    kwargs.set_item("copy", false)?;
    let array = py
        .import("jax.dlpack")?
        .call_method("from_dlpack", (producer,), Some(&kwargs))
        .map_err(|err| io::Error::new(ErrorKind::Unsupported, err.to_string()))?;
    if array
        .call_method0("unsafe_buffer_pointer")?
        .extract::<usize>()?
        != address
    {
        return Err(io::Error::new(
            ErrorKind::Unsupported,
            "JAX did not alias the source allocation",
        ));
    }
    Ok(array)
}

fn import_host<'py>(
    py: Python<'py>,
    host: HostBuffer,
    sharding: &Bound<'py, PyAny>,
    info: &ArrayInfo,
) -> io::Result<Bound<'py, PyAny>> {
    if host.info() != info {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            "host export changed array metadata",
        ));
    }
    let numpy = py.import("numpy")?;
    let dtype: &'static str = info.dtype().into();
    let packed = numpy
        .call_method1("dtype", (dtype,))?
        .call_method1("newbyteorder", ("<",))?;
    let bytes = PyBytes::new(py, host.bytes());
    let array = numpy.call_method1("frombuffer", (bytes, packed))?;
    let dims = jaxpr::dimensions(info).map_err(io::Error::other)?;
    let array = array.call_method1("reshape", (PyTuple::new(py, dims)?,))?;
    let array = numpy.call_method1("asarray", (array, dtype))?;
    Ok(py
        .import("jax")?
        .call_method1("device_put", (array, sharding))?)
}

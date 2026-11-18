use std::fmt;
use std::io::{self, ErrorKind};

use gnx_math::{
    ArrayInfo, BufferInfo, BufferLayout, DataHandle, DataImpl, DlpackDevice, DlpackRequest,
    DlpackTensor, HostBuffer, MemoryLocation, TransferPolicy,
};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyTuple};

use crate::{JaxDevice, dlpack, jaxpr};

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

    pub fn device(&self) -> &JaxDevice {
        &self.device
    }

    pub fn native<'a, 'py>(&'a self, py: Python<'py>) -> &'a Bound<'py, PyAny> {
        self.native.bind(py)
    }
}

impl fmt::Display for JaxBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "JaxBuffer({} on {})", self.info, self.device)
    }
}

impl DataImpl for JaxBuffer {
    fn info(&self) -> ArrayInfo {
        self.info.clone()
    }

    fn buffer_info(&self) -> BufferInfo {
        BufferInfo {
            location: MemoryLocation::Device {
                runtime: "jax",
                device_id: self.device.uuid().to_string(),
            },
            memory_kind: self.memory_kind.clone(),
            layout: BufferLayout::Opaque,
        }
    }

    fn to_host(&self) -> io::Result<HostBuffer> {
        let bytes = Python::attach(|py| -> PyResult<_> {
            let numpy = py.import("numpy")?;
            let dtype: &'static str = self.info.dtype().into();
            let packed = numpy
                .call_method1("dtype", (dtype,))?
                .call_method1("newbyteorder", ("<",))?;
            let array = numpy.call_method1("asarray", (self.native(py), packed))?;
            let bytes = array.call_method1("tobytes", ("C",))?;
            Ok(bytes.cast::<PyBytes>()?.as_bytes().to_vec())
        })?;
        HostBuffer::new(self.info.clone(), bytes)
    }

    fn dlpack_device(&self) -> Option<DlpackDevice> {
        Python::attach(|py| {
            self.native(py)
                .call_method0("__dlpack_device__")?
                .extract::<(i32, i32)>()
                .map(|(device_type, id)| DlpackDevice(device_type, id))
        })
        .ok()
    }

    fn export_dlpack(&self, request: DlpackRequest) -> io::Result<DlpackTensor> {
        Python::attach(|py| {
            // SAFETY: a concrete native JAX array implements the protocol and
            // is immutable. Foreign producers must uphold the lease contract.
            unsafe { dlpack::export(self.native(py), request) }
        })
        .map_err(io::Error::other)
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

    /// Import concrete storage onto this device, subject to the copy policy.
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
        let array = if let Some(buffer) = buffer.downcast_ref::<JaxBuffer>() {
            if self.native(py).eq(buffer.device.native(py))? && buffer.memory_kind == memory_kind {
                buffer.native(py).clone()
            } else if policy == TransferPolicy::AllowCopy {
                jax.call_method1("device_put", (buffer.native(py), &sharding))?
            } else {
                return Err(io::Error::new(
                    ErrorKind::Unsupported,
                    "moving to this JAX device would require a copy",
                ));
            }
        } else if policy == TransferPolicy::RequireZeroCopy {
            import_dlpack(py, buffer, &sharding, Some(false))?
        } else if buffer.downcast_ref::<HostBuffer>().is_some() {
            import_host(py, buffer.into_host(policy)?, &sharding, &info)?
        } else {
            match buffer.to_host() {
                Ok(host) => import_host(py, host, &sharding, &info)?,
                Err(err) if err.kind() == ErrorKind::Unsupported => {
                    import_dlpack(py, buffer, &sharding, None)?
                }
                Err(err) => return Err(err),
            }
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
    copy: Option<bool>,
) -> io::Result<Bound<'py, PyAny>> {
    let source = buffer.into_dlpack().map_err(|_| {
        io::Error::new(
            ErrorKind::Unsupported,
            "source cannot export a zero-copy DLPack lease",
        )
    })?;
    let producer = dlpack::producer(py, source)?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("device", sharding)?;
    kwargs.set_item("copy", copy)?;
    Ok(py
        .import("jax.dlpack")?
        .call_method("from_dlpack", (producer,), Some(&kwargs))?)
}

fn import_host<'py>(
    py: Python<'py>,
    host: HostBuffer,
    sharding: &Bound<'py, PyAny>,
    info: &ArrayInfo,
) -> io::Result<Bound<'py, PyAny>> {
    if host.info() != *info {
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
    let bytes = PyBytes::new(py, &host.into_bytes());
    let array = numpy.call_method1("frombuffer", (bytes, packed))?;
    let dims = jaxpr::dimensions(info).map_err(io::Error::other)?;
    let array = array.call_method1("reshape", (PyTuple::new(py, dims)?,))?;
    // JAX expects native-endian host scalars, including on big-endian hosts.
    let array = numpy.call_method1("asarray", (array, dtype))?;
    Ok(py
        .import("jax")?
        .call_method1("device_put", (array, sharding))?)
}

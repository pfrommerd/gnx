//! Bridge a portable GNX DLPack lease to Python's single-use capsule protocol.

use std::ffi::{CStr, c_void};
use std::ptr::NonNull;
use std::sync::Mutex;

use gnx_math::{DlpackAbi, DlpackDevice, DlpackRequest, DlpackSource, DlpackTensor};
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn names(abi: DlpackAbi) -> (&'static CStr, &'static CStr) {
    match abi {
        DlpackAbi::Legacy => (c"dltensor", c"used_dltensor"),
        DlpackAbi::Versioned => (c"dltensor_versioned", c"used_dltensor_versioned"),
    }
}

// Only the stable prefix through `deleter` is accessed. Tail fields of the
// DLPack managed tensor are passed untouched to the receiving runtime.
#[repr(C)]
struct DLTensor {
    data: *mut c_void,
    device_type: i32,
    device_id: i32,
    ndim: i32,
    dtype_code: u8,
    dtype_bits: u8,
    dtype_lanes: u16,
    shape: *mut i64,
    strides: *mut i64,
    byte_offset: u64,
}

#[repr(C)]
struct Legacy {
    tensor: DLTensor,
    manager_ctx: *mut c_void,
    deleter: Option<unsafe extern "C" fn(*mut Legacy)>,
}

#[repr(C)]
struct Versioned {
    major: u32,
    minor: u32,
    manager_ctx: *mut c_void,
    deleter: Option<unsafe extern "C" fn(*mut Versioned)>,
}

#[repr(C)]
struct VersionOne {
    prefix: Versioned,
    flags: u64,
}

unsafe fn release_legacy(ptr: NonNull<c_void>) {
    let ptr = ptr.cast::<Legacy>().as_ptr();
    // SAFETY: callers own a live legacy tensor lease.
    unsafe {
        if let Some(release) = (*ptr).deleter {
            release(ptr)
        }
    }
}

unsafe fn release_versioned(ptr: NonNull<c_void>) {
    let ptr = ptr.cast::<Versioned>().as_ptr();
    // SAFETY: callers own a live versioned tensor lease.
    unsafe {
        if let Some(release) = (*ptr).deleter {
            release(ptr)
        }
    }
}

unsafe extern "C" fn capsule_destructor(capsule: *mut ffi::PyObject) {
    // A consumer renames the capsule to `used_*` when it takes the lease.
    for (abi, release) in [
        (
            DlpackAbi::Legacy,
            release_legacy as unsafe fn(NonNull<c_void>),
        ),
        (DlpackAbi::Versioned, release_versioned),
    ] {
        let (name, _) = names(abi);
        // SAFETY: CPython calls this with a live capsule while attached.
        unsafe {
            if ffi::PyCapsule_IsValid(capsule, name.as_ptr()) != 0 {
                if let Some(ptr) = NonNull::new(ffi::PyCapsule_GetPointer(capsule, name.as_ptr())) {
                    release(ptr);
                }
                return;
            }
        }
    }
}

fn into_capsule(py: Python<'_>, tensor: DlpackTensor) -> PyResult<Bound<'_, PyAny>> {
    let (name, _) = names(tensor.abi());
    // SAFETY: valid managed pointer, static name, and a C ABI destructor. The
    // Rust token still owns the lease if capsule allocation fails.
    let capsule = unsafe {
        Bound::from_owned_ptr_or_err(
            py,
            ffi::PyCapsule_New(
                tensor.as_ptr().as_ptr(),
                name.as_ptr(),
                Some(capsule_destructor),
            ),
        )?
    };
    tensor.into_raw();
    Ok(capsule)
}

/// Request a zero-copy lease from a native Python DLPack producer.
///
/// # Safety
/// `array` must implement the DLPack protocol correctly, including valid
/// pointers, the requested synchronization and no writes during the lease.
pub unsafe fn export(array: &Bound<'_, PyAny>, request: DlpackRequest) -> PyResult<DlpackTensor> {
    let py = array.py();
    let kwargs = PyDict::new(py);
    kwargs.set_item("stream", request.stream)?;
    kwargs.set_item("max_version", request.max_version)?;
    kwargs.set_item("copy", false)?;
    let capsule = array.call_method("__dlpack__", (), Some(&kwargs))?;
    for (abi, release) in [
        (
            DlpackAbi::Legacy,
            release_legacy as unsafe fn(NonNull<c_void>),
        ),
        (DlpackAbi::Versioned, release_versioned),
    ] {
        let (name, consumed) = names(abi);
        // SAFETY: the producer supplies a valid capsule and lease. Transfer
        // responsibility only after renaming defeats the producer's destructor.
        unsafe {
            if ffi::PyCapsule_IsValid(capsule.as_ptr(), name.as_ptr()) == 0 {
                continue;
            }
            let ptr = NonNull::new(ffi::PyCapsule_GetPointer(capsule.as_ptr(), name.as_ptr()))
                .ok_or_else(|| PyErr::fetch(py))?;
            if ffi::PyCapsule_SetName(capsule.as_ptr(), consumed.as_ptr()) != 0 {
                return Err(PyErr::fetch(py));
            }
            let tensor = DlpackTensor::from_raw(ptr, abi, release);
            if abi == DlpackAbi::Versioned {
                let version = ptr.cast::<Versioned>().as_ref();
                // Minor revisions share the ABI. In particular JAX may
                // report its newer minor revision to a v1.0 consumer.
                if version.major != 1 || request.max_version.is_none_or(|(major, _)| major < 1) {
                    return Err(PyValueError::new_err(
                        "DLPack producer returned an unnegotiated ABI version",
                    ));
                }
                const IS_COPIED: u64 = 1 << 1;
                if ptr.cast::<VersionOne>().as_ref().flags & IS_COPIED != 0 {
                    return Err(PyValueError::new_err(
                        "DLPack producer copied despite copy=False",
                    ));
                }
            }
            return Ok(tensor);
        }
    }
    Err(PyValueError::new_err(
        "expected an unused DLPack managed-tensor capsule",
    ))
}

#[pyclass(frozen)]
struct Producer {
    device: DlpackDevice,
    source: Mutex<Option<DlpackSource>>,
}

#[pymethods]
impl Producer {
    fn __dlpack_device__(&self) -> (i32, i32) {
        (self.device.0, self.device.1)
    }

    #[pyo3(signature = (*, stream = None, max_version = None, dl_device = None, copy = None))]
    fn __dlpack__<'py>(
        &self,
        py: Python<'py>,
        stream: Option<isize>,
        max_version: Option<(u32, u32)>,
        dl_device: Option<(i32, i32)>,
        copy: Option<bool>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if dl_device.is_some_and(|device| device != self.__dlpack_device__()) || copy == Some(true)
        {
            return Err(PyValueError::new_err(
                "DLPack handoff cannot copy or change the device",
            ));
        }
        let source = self
            .source
            .lock()
            .map_err(|_| PyRuntimeError::new_err("DLPack source lock poisoned"))?
            .take()
            .ok_or_else(|| PyValueError::new_err("this DLPack source was already consumed"))?;
        let tensor = source.export(DlpackRequest {
            stream,
            max_version,
        })?;
        if tensor.abi() == DlpackAbi::Versioned && max_version.is_none_or(|(major, _)| major < 1) {
            return Err(PyValueError::new_err(
                "consumer did not negotiate versioned DLPack",
            ));
        }
        into_capsule(py, tensor)
    }
}

/// Adapt a consuming Rust transfer to a Python array-protocol producer.
pub fn producer(py: Python<'_>, source: DlpackSource) -> PyResult<Bound<'_, PyAny>> {
    Producer {
        device: source.device(),
        source: Mutex::new(Some(source)),
    }
    .into_bound_py_any(py)
}

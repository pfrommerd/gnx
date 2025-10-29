use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use std::sync::Arc;

#[derive(Clone)]
enum BytesImpl {
    Rc(Arc<Vec<u8>>),
    Borrowed(&'static [u8]),
    // Use arc to allow for cloning
    // without needing access to the GIL.
    PyBytes(Arc<Py<PyBytes>>),
}

impl AsRef<[u8]> for ImBytes {
    fn as_ref(&self) -> &[u8] {
        match &self.inner {
            BytesImpl::Rc(rc) => rc.as_ref(),
            BytesImpl::Borrowed(b) => b,
            BytesImpl::PyBytes(py_bytes) => {
                Python::attach(|py| py_bytes.bind_borrowed(py).extract::<&[u8]>().unwrap())
            }
        }
    }
}

#[derive(Clone)]
pub struct ImBytes {
    inner: BytesImpl,
}

impl ImBytes {
    pub fn as_slice(&self) -> &[u8] {
        self.as_ref()
    }

    pub fn len(&self) -> usize {
        self.as_ref().len()
    }

    pub fn is_empty(&self) -> bool {
        self.as_ref().is_empty()
    }
}

impl From<Vec<u8>> for ImBytes {
    fn from(b: Vec<u8>) -> Self {
        ImBytes {
            inner: BytesImpl::Rc(Arc::new(b)),
        }
    }
}

impl From<&'static [u8]> for ImBytes {
    fn from(b: &'static [u8]) -> Self {
        ImBytes {
            inner: BytesImpl::Borrowed(b),
        }
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for ImBytes {
    type Error = PyErr;
    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if let Ok(py_bytes) = obj.cast::<PyBytes>() {
            Ok(ImBytes {
                inner: BytesImpl::PyBytes(Arc::new(py_bytes.into())),
            })
        } else {
            Err(PyTypeError::new_err("Expected bytes!"))
        }
    }
}

impl std::fmt::Debug for ImBytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.inner {
            BytesImpl::Rc(rc) => write!(f, "{:?}", *rc),
            BytesImpl::Borrowed(b) => write!(f, "{:?}", b),
            BytesImpl::PyBytes(py_bytes) => write!(f, "{:?}", py_bytes),
        }
    }
}

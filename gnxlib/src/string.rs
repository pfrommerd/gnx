use pyo3::exceptions::PyTypeError;
use pyo3::types::PyString;
use pyo3::{prelude::*, BoundObject};

use std::sync::OnceLock;
use std::{borrow::Borrow, sync::Arc};

use gnx::util::LifetimeFree;

// We store both a Rust and Python representation of the string
// and convert between them as needed.
pub struct OwnedString {
    rust: OnceLock<String>,
    py: OnceLock<Py<PyString>>,
}

impl From<String> for OwnedString {
    fn from(s: String) -> Self {
        OwnedString {
            rust: OnceLock::from(s),
            py: OnceLock::new(),
        }
    }
}

impl From<Py<PyString>> for OwnedString {
    fn from(py_string: Py<PyString>) -> Self {
        OwnedString {
            rust: OnceLock::new(),
            py: OnceLock::from(py_string),
        }
    }
}

impl AsRef<str> for OwnedString {
    fn as_ref(&self) -> &str {
        let rust_str = self.rust.get_or_init(|| {
            // Get the python string and convert it to a Rust string
            let py_string = self.py.get().unwrap();
            Python::attach(|py| py_string.bind(py).to_string_lossy().into_owned())
        });
        rust_str.as_str()
    }
}

#[derive(Clone)]
enum StringImpl {
    Owned(Arc<OwnedString>),
    Borrowed(&'static str),
}

#[derive(Clone)]
pub struct ImString(StringImpl);
unsafe impl LifetimeFree for ImString {}

impl From<String> for ImString {
    fn from(s: String) -> Self {
        ImString(StringImpl::Owned(Arc::new(OwnedString::from(s))))
    }
}

impl From<&'static str> for ImString {
    fn from(s: &'static str) -> Self {
        ImString(StringImpl::Borrowed(s))
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for ImString {
    type Error = PyErr;
    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if let Ok(py_string) = obj.cast::<PyString>() {
            Ok(ImString(StringImpl::Owned(Arc::new(OwnedString::from(
                py_string.into_bound().unbind(),
            )))))
        } else {
            Err(PyTypeError::new_err("Expected bytes!"))
        }
    }
}

impl ImString {
    pub fn as_str(&self) -> &str {
        self.as_ref()
    }
}

impl AsRef<str> for ImString {
    fn as_ref<'s>(&'s self) -> &'s str {
        match &self.0 {
            StringImpl::Borrowed(b) => b,
            StringImpl::Owned(owned) => owned.as_ref().as_ref(),
        }
    }
}

impl Borrow<str> for ImString {
    fn borrow(&self) -> &str {
        self.as_ref()
    }
}

impl PartialEq for ImString {
    fn eq(&self, other: &Self) -> bool {
        self.as_ref() == other.as_ref()
    }
}

impl Eq for ImString {}

impl std::hash::Hash for ImString {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.as_ref().hash(state);
    }
}
impl std::fmt::Debug for ImString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.as_ref().fmt(f)
    }
}

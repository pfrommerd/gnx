use gnx::buffer::TransferPolicy;
use gnx::{Eval, ExecOpts};
use gnx_xla::XlaDevice;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use uuid::Uuid;

use crate::tracer::PyArray;

#[pyclass(name = "XlaDevice", module = "gnxlib.xla", frozen)]
pub struct PyXlaDevice(XlaDevice);

#[pymethods]
impl PyXlaDevice {
    #[getter]
    fn platform(&self) -> &'static str {
        self.0.platform()
    }

    #[getter]
    fn host_uuid(&self) -> Uuid {
        self.0.host().0
    }

    #[getter]
    fn hardware_kind(&self) -> &str {
        self.0.hardware_kind()
    }

    #[getter]
    fn device_ordinal(&self) -> u64 {
        self.0.ordinal()
    }

    #[pyo3(signature = (roots, *, allow_copy = false))]
    async fn eval(&self, roots: Py<PyAny>, allow_copy: bool) -> PyResult<()> {
        let roots = Python::attach(|py| crate::parse_roots(roots.bind(py)))?;
        let transfers = if allow_copy {
            TransferPolicy::AllowCopy
        } else {
            TransferPolicy::RequireZeroCopy
        };
        self.0
            .eval(roots.as_ref(), ExecOpts { transfers })
            .await
            .map_err(crate::eval_error)
    }

    /// Materialize a packed host copy and return its little-endian bytes.
    fn to_host_bytes<'py>(
        &self,
        py: Python<'py>,
        array: &PyArray,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let data = array
            .0
            .data()
            .ok_or_else(|| PyValueError::new_err("array has not been evaluated"))?
            .clone();
        let host = py.detach(|| self.0.download(&data))?;
        Ok(PyBytes::new(py, host.bytes()))
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

/// Discover the process's single-device CPU XLA backend.
#[pyfunction]
pub fn xla_devices(py: Python<'_>) -> PyResult<Vec<PyXlaDevice>> {
    Ok(vec![PyXlaDevice(py.detach(XlaDevice::cpu)?)])
}

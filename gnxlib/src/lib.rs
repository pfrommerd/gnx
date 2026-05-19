pub mod bytes;
pub mod graph;
pub mod leaf;
pub mod string;

use pyo3::prelude::*;

/// A Python module implemented in Rust.

#[pymodule]
pub mod gnxlib {
    use crate::graph::PyGraph;
    use gnx::backend::{Backend, Device};
    use pyo3::prelude::*;
    use uuid::Uuid;
    use std::sync::Arc;

    #[pyclass(name = "Backend", eq)]
    #[derive(PartialEq, Eq)]
    struct PyBackend(Backend);

    #[pymethods]
    impl PyBackend {
        fn name(&self) -> &str {
            self.0.name()
        }

        fn __repr__(&self) -> String {
            format!("Backend(name='{}')", self.name())
        }
    }

    #[pyclass(name = "Device", eq, frozen, hash)]
    #[derive(PartialEq, Eq, Hash)]
    struct PyDevice(Device);

    #[pymethods]
    impl PyDevice {
        #[getter]
        fn uuid(&self) -> &Uuid {
            self.0.uuid()
        }
        #[getter]
        fn platform(&self) -> &str {
            self.0.platform()
        }
        #[getter]
        fn hardware_id(&self) -> &str {
            self.0.hardware_id()
        }
        #[getter]
        fn hardware_kind(&self) -> &str {
            self.0.hardware_kind()
        }

        fn __str__(&self) -> String {
            format!("{}", self.0)
        }
    }

    #[pyfunction]
    fn enable_jax_backend() -> PyBackend {
        PyBackend(gnx_jax::register())
    }

    #[pyfunction]
    fn backends() -> Vec<PyBackend> {
        gnx::backend::backends()
            .into_iter()
            .map(PyBackend)
            .collect()
    }

    #[pyfunction]
    fn devices() -> Result<Vec<PyDevice>, PyErr> {
        Ok(gnx::backend::devices()?.into_iter().map(PyDevice).collect())
    }

    // Contains a shared graph reference.
    #[pyclass(name = "Graph", skip_from_py_object)]
    #[derive(Clone)]
    pub struct GraphContainer(pub Arc<PyGraph>);

    #[pymethods]
    impl GraphContainer {
        #[staticmethod]
        fn from_value(value: PyGraph) -> Self {
            GraphContainer(Arc::new(value))
        }
        // Will return a copy of the graph
        fn to_value(&self) -> PyGraph {
            (*self.0).clone()
        }
    }
}
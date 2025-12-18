pub mod bytes;
pub mod graph;
pub mod leaf;
pub mod string;


use pyo3::prelude::*;

/// A Python module implemented in Rust.

#[pymodule]
mod gnxlib {
    use gnx::backend::{Backend, Device};
    use uuid::Uuid;
    use pyo3::prelude::*;

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
        fn uuid(&self) -> &Uuid { self.0.uuid() }
        #[getter]
        fn platform(&self) -> &str { self.0.platform() }
        #[getter]
        fn hardware_id(&self) -> &str { self.0.hardware_id() }
        #[getter]
        fn hardware_kind(&self) -> &str { self.0.hardware_kind() }

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

    // use crate::*;
    // #[pyclass(name = "Graph")]
    // struct GraphWrapper(PyGraph);

    // #[pymethods]
    // impl GraphWrapper {
    //     // #[getter]
    //     // fn graphdef(&self) -> PyGraphDef {}
    //     #[staticmethod]
    //     fn from_value(value: PyGraph) -> Self {
    //         GraphWrapper(value)
    //     }

    //     fn value(&self) -> &PyGraph {
    //         &self.0
    //     }
    //     fn __repr__(&self) -> String {
    //         format!("{:?}", self.0)
    //     }
    //     fn __str__(&self) -> String {
    //         format!("{}", self.0)
    //     }
    // }
}

pub mod bytes;
pub mod graph;
pub mod leaf;
pub mod string;
mod tracer;

use std::sync::Arc;

use gnx::expr::HExpr;
use gnx::{Function, Jit, TryCallable, TryJitable};
use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::gc::{PyTraverseError, PyVisit};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::graph::{GraphParser, PyGraph};

/// A Python module implemented in Rust.

#[pymodule]
pub mod _native {
    use gnx::backend::{Backend, Device};
    use pyo3::prelude::*;
    use uuid::Uuid;

    #[pymodule_export]
    use super::{jit, PyJit, PyTrace};
    #[pymodule_export]
    use crate::graph::GraphContainer;
    #[pymodule_export]
    use crate::tracer::{new_ref, placeholder, PyArray, PyArrayRef};

    #[pyclass(name = "Backend", module = "gnxlib.backend", eq)]
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

    #[pyclass(name = "Device", module = "gnxlib.backend", eq, frozen, hash)]
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
}

struct PythonFunction(Arc<Py<PyAny>>);

impl Function<PyGraph> for PythonFunction {
    type Output = PyGraph;
}

impl TryCallable<PyGraph> for PythonFunction {
    type Error = PyErr;

    fn try_invoke(&self, input: PyGraph) -> PyResult<PyGraph> {
        Python::attach(|py| {
            let PyGraph::Tuple(call) = input else {
                unreachable!("JIT inputs contain positional and keyword arguments");
            };
            let [args, kwargs]: [PyGraph; 2] = call.try_into().ok().expect("JIT call graph arity");
            let args = args.into_pyobject_bound(py)?.cast_into::<PyTuple>()?;
            let kwargs = kwargs.into_pyobject_bound(py)?.cast_into::<PyDict>()?;
            let result = self.0.bind(py).call(args, Some(&kwargs))?;
            result.extract()
        })
    }
}

impl TryJitable<PyGraph> for PythonFunction {
    fn try_invoke_owned(&self, input: PyGraph) -> PyResult<PyGraph> {
        self.try_invoke(input)
    }
}

/// A read-only high-level trace, without live argument or capture bindings.
#[pyclass(name = "Trace", module = "gnxlib", frozen)]
pub struct PyTrace(Arc<HExpr>);

#[pymethods]
impl PyTrace {
    #[getter]
    fn input_count(&self) -> usize {
        self.0.inputs().len()
    }

    #[getter]
    fn output_count(&self) -> usize {
        self.0.outputs().len()
    }

    #[getter]
    fn capture_count(&self) -> usize {
        self.0.closure().len()
    }

    fn __str__(&self) -> String {
        self.0.to_string()
    }

    fn __repr__(&self) -> String {
        format!(
            "<gnxlib.Trace inputs={} outputs={} captures={}>",
            self.input_count(),
            self.output_count(),
            self.capture_count(),
        )
    }
}

#[pyclass(name = "Jit", module = "gnxlib")]
pub struct PyJit {
    // Both Arcs share a single owned Python reference, visited once by the GC.
    function: Option<Arc<Py<PyAny>>>,
    jit: Option<Arc<Jit<PyGraph, PythonFunction>>>,
}

#[pymethods]
impl PyJit {
    #[getter]
    fn __wrapped__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.function
            .as_ref()
            .map(|function| function.clone_ref(py))
            .ok_or_else(cleared)
    }

    #[pyo3(signature = (*args, **kwargs))]
    fn __call__<'py>(
        &self,
        py: Python<'py>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let output = self.with_arguments(py, args, kwargs, TryCallable::try_invoke)?;
        output.into_pyobject_bound(py)
    }

    /// Inspect the trace for these arguments without calling it or updating cells.
    #[pyo3(signature = (*args, **kwargs))]
    fn trace(
        &self,
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyTrace> {
        self.with_arguments(py, args, kwargs, Jit::try_trace)
            .map(PyTrace)
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        if let Some(function) = &self.function {
            visit.call(function.as_ref())?;
        }
        Ok(())
    }

    fn __clear__(&mut self) {
        self.jit = None;
        self.function = None;
    }
}

impl PyJit {
    fn with_arguments<T: Send>(
        &self,
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
        f: impl FnOnce(&Jit<PyGraph, PythonFunction>, PyGraph) -> PyResult<T> + Send,
    ) -> PyResult<T> {
        let input = GraphParser::parse_args(args, kwargs)?;
        let jit = self.jit.as_ref().ok_or_else(cleared)?.clone();
        // A competing trace may need Python to finish. Never wait for its slot
        // while attached. The miss callback reattaches on this same thread.
        py.detach(move || f(&jit, input))
    }
}

fn cleared() -> PyErr {
    PyRuntimeError::new_err("JIT function has been cleared")
}

/// Trace a Python callable with symbolic GNX arrays and cells. This decorator
/// caches traces, not computed values, and does not execute a backend.
/// The Python body runs on cache misses. Arguments and returns may contain
/// the same values as Graph.from_value, including types with __graph__ hooks.
/// Flatten hooks run before cache lookup and may run on cache hits.
#[pyfunction]
pub fn jit(py: Python<'_>, function: Py<PyAny>) -> PyResult<PyJit> {
    if !function.bind(py).is_callable() {
        return Err(PyTypeError::new_err("jit expects a callable"));
    }
    let function = Arc::new(function);
    let jit = Arc::new(gnx::jit(PythonFunction(function.clone())));
    Ok(PyJit {
        function: Some(function),
        jit: Some(jit),
    })
}

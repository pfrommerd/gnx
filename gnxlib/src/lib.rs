pub mod bytes;
pub mod graph;
pub mod leaf;
pub mod string;
mod tracer;
#[cfg(feature = "xla")]
mod xla;

use std::sync::Arc;

use gnx::expr::{Capture, HExpr, ObjectTracer};
use gnx::{Function, Jit, TryCallable, TryJitable};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::gc::{PyTraverseError, PyVisit};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::graph::{parse_roots, GraphParser, PyGraph};

/// A Python module implemented in Rust.

#[pymodule]
pub mod _native {
    use gnx::buffer::TransferPolicy;
    use gnx::{Eval, ExecOpts};
    use gnx_jax::JaxDevice;
    use pyo3::prelude::*;
    use uuid::Uuid;

    #[pymodule_export]
    use super::{expr, jit, PyCapture, PyJit, PyTrace};
    #[pymodule_export]
    use crate::graph::GraphContainer;
    #[pymodule_export]
    use crate::tracer::{from_host_bytes, new_ref, placeholder, PyArray, PyArrayRef};
    #[cfg(feature = "xla")]
    #[pymodule_export]
    use crate::xla::{xla_devices, PyXlaDevice};

    #[pyclass(name = "JaxDevice", module = "gnxlib.jax", eq, frozen, hash)]
    #[derive(PartialEq, Eq, Hash)]
    struct PyJaxDevice(JaxDevice);

    #[pymethods]
    impl PyJaxDevice {
        #[staticmethod]
        fn from_native(device: Bound<'_, PyAny>) -> PyResult<Self> {
            JaxDevice::from_native(device).map(Self)
        }

        #[getter]
        fn uuid(&self) -> &Uuid {
            self.0.uuid()
        }

        #[getter]
        fn platform(&self) -> &str {
            self.0.platform()
        }

        #[getter]
        fn hardware_kind(&self) -> &str {
            self.0.hardware_kind()
        }

        #[getter]
        fn hardware_id(&self) -> &str {
            self.0.hardware_id()
        }

        #[getter]
        fn native<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
            self.0.native(py).clone()
        }

        /// Materialize the pending values in a graph in place.
        #[pyo3(signature = (roots, *, allow_copy = false))]
        async fn eval(&self, roots: Py<PyAny>, allow_copy: bool) -> PyResult<()> {
            let roots = Python::attach(|py| super::parse_roots(roots.bind(py)))?;
            let transfers = if allow_copy {
                TransferPolicy::AllowCopy
            } else {
                TransferPolicy::RequireZeroCopy
            };
            self.0
                .eval(roots.as_ref(), ExecOpts { transfers })
                .await
                .map_err(super::eval_error)
        }

        fn __repr__(&self) -> String {
            self.0.to_string()
        }
    }

    /// Return JAX device handles for the selected platform or JAX's default.
    #[pyfunction(signature = (backend = None))]
    fn jax_devices(py: Python<'_>, backend: Option<&str>) -> PyResult<Vec<PyJaxDevice>> {
        Ok(gnx_jax::devices(py, backend)?
            .into_iter()
            .map(PyJaxDevice)
            .collect())
    }
}

fn eval_error(error: gnx::EvalError) -> PyErr {
    use gnx::{EvalError, EvalPlanError};
    match error {
        EvalError::Backend(error) => error.into(),
        EvalError::Capture(_)
        | EvalError::Plan(EvalPlanError::UnboundPlaceholder | EvalPlanError::CellEffect) => {
            PyValueError::new_err(error.to_string())
        }
        _ => PyRuntimeError::new_err(error.to_string()),
    }
}

/// Capture a graph without executing or materializing it.
#[pyfunction]
fn expr(roots: &Bound<'_, PyAny>) -> PyResult<PyCapture> {
    let roots = parse_roots(roots)?;
    gnx::expr(roots.as_ref())
        .map(PyCapture)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

#[pyclass(name = "Capture", module = "gnxlib", frozen)]
pub struct PyCapture(Capture);

#[pymethods]
impl PyCapture {
    #[getter]
    fn closure<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        binding_tuple(py, self.0.closure())
    }

    #[getter]
    fn inputs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        binding_tuple(py, self.0.inputs())
    }

    #[getter]
    fn outputs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        binding_tuple(py, self.0.outputs())
    }

    fn __str__(&self) -> String {
        self.0.to_string()
    }

    fn __repr__(&self) -> String {
        format!(
            "<gnxlib.Capture inputs={} outputs={} captures={}>",
            self.0.inputs().len(),
            self.0.outputs().len(),
            self.0.closure().len()
        )
    }
}

fn binding_tuple<'py>(py: Python<'py>, bindings: &[ObjectTracer]) -> PyResult<Bound<'py, PyTuple>> {
    let values = bindings
        .iter()
        .map(|binding| tracer::wrap_tracer(py, binding.clone()))
        .collect::<PyResult<Vec<_>>>()?;
    PyTuple::new(py, values)
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

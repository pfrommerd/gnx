//! Python bindings for the function transformations: `map` (vectorization)
//! and automatic differentiation (`grad`, `value_and_grad`, `hessian`, `jvp`,
//! `vjp`).
//!
//! Every Python function sees the input graph `(args, kwargs)`, as for `jit`.
//! Transformed callables are wrapped in an outer `jit`, so that they cache
//! their traces and `trace()` shows the recorded transformation op.

use std::sync::Arc;

use gnx::array::Array;
use gnx::expr::LoweringError;
use gnx::graph::util::{Dag, DagChild, DagNode, OrderMap};
use gnx::graph::Key;
use gnx::transforms::{ApplyError, TransformError};
use gnx::{Function, Jit, TryCallable, TryJitable};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::gc::{PyTraverseError, PyVisit};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyInt, PyList, PyString, PyTuple};

use crate::graph::{GraphParser, PyGraph};
use crate::{PyTrace, PythonFunction};

// ----- errors -----------------------------------------------------------------

/// Signature (shape/dtype/arity) errors are `TypeError`s; invalid
/// specifications and other transformation errors are `ValueError`s.
pub(crate) fn transform_error(error: TransformError) -> PyErr {
    match error {
        TransformError::Signature(_) | TransformError::Interface(_) => {
            PyTypeError::new_err(error.to_string())
        }
        _ => PyValueError::new_err(error.to_string()),
    }
}

pub(crate) fn lowering_error(error: LoweringError) -> PyErr {
    match error {
        LoweringError::RegionSignature(message) => {
            PyTypeError::new_err(format!("invalid region signature: {message}"))
        }
        error => PyTypeError::new_err(error.to_string()),
    }
}

fn apply_error(error: ApplyError<PyErr>) -> PyErr {
    match error {
        ApplyError::Body(error) => error,
        ApplyError::Transform(error) => transform_error(error),
    }
}

// ----- graph helpers ----------------------------------------------------------

/// Split the `(args, kwargs)` call graph.
fn call_parts(input: PyGraph) -> (Vec<PyGraph>, PyGraph) {
    let PyGraph::Tuple(call) = input else {
        unreachable!("transformed inputs contain positional and keyword arguments");
    };
    let [args, kwargs]: [PyGraph; 2] = call.try_into().ok().expect("call graph arity");
    let PyGraph::Tuple(args) = args else {
        unreachable!("positional arguments are a tuple");
    };
    (args, kwargs)
}

fn call_graph(args: Vec<PyGraph>, kwargs: PyGraph) -> PyGraph {
    PyGraph::Tuple(vec![PyGraph::Tuple(args), kwargs])
}

fn tuple_items(graph: PyGraph) -> Vec<PyGraph> {
    match graph {
        PyGraph::Tuple(items) => items,
        _ => unreachable!("expected a tuple graph"),
    }
}

fn list_dag<T>(children: impl IntoIterator<Item = Dag<T>>) -> Dag<T> {
    Dag::Node(DagNode::List(
        children.into_iter().map(DagChild::Owned).collect(),
    ))
}

/// Convert a JAX-style axis spec (a pytree prefix of ints and `None`) to a Dag.
fn axes_dag(spec: &Bound<'_, PyAny>, what: &str) -> PyResult<Dag<usize>> {
    if spec.is_none() {
        Ok(Dag::Leaf(None))
    } else if spec.is_instance_of::<PyInt>() && !spec.is_instance_of::<PyBool>() {
        let axis: isize = spec.extract()?;
        usize::try_from(axis)
            .map(Dag::from)
            .map_err(|_| PyValueError::new_err(format!("{what} must be non-negative, got {axis}")))
    } else if let Ok(tuple) = spec.cast::<PyTuple>() {
        Ok(list_dag(
            tuple
                .iter()
                .map(|child| axes_dag(&child, what))
                .collect::<PyResult<Vec<_>>>()?,
        ))
    } else if let Ok(list) = spec.cast::<PyList>() {
        Ok(list_dag(
            list.iter()
                .map(|child| axes_dag(&child, what))
                .collect::<PyResult<Vec<_>>>()?,
        ))
    } else if let Ok(dict) = spec.cast::<PyDict>() {
        let mut children = OrderMap::new();
        for (key, child) in dict.iter() {
            let key = if key.is_exact_instance_of::<PyString>() {
                Key::from(key.extract::<String>()?)
            } else if key.is_exact_instance_of::<PyInt>() {
                Key::from(key.extract::<isize>()?)
            } else {
                return Err(PyTypeError::new_err(format!(
                    "{what} dict keys must be str or int"
                )));
            };
            children.insert(key, DagChild::Owned(axes_dag(&child, what)?));
        }
        Ok(Dag::Node(DagNode::Map(children)))
    } else {
        Err(PyTypeError::new_err(format!(
            "{what} must be an int, None, or a tuple/list/dict of them"
        )))
    }
}

/// Parse `argnums` into (positions, whether it was a single int).
fn parse_argnums(argnums: &Bound<'_, PyAny>) -> PyResult<(Arc<[usize]>, bool)> {
    let single = argnums.is_instance_of::<PyInt>();
    let numbers: Vec<isize> = if single {
        vec![argnums.extract()?]
    } else if let Ok(tuple) = argnums.cast::<PyTuple>() {
        tuple.extract()?
    } else {
        return Err(PyTypeError::new_err(
            "argnums must be an int or a tuple of ints",
        ));
    };
    let mut positions = Vec::with_capacity(numbers.len());
    for number in numbers {
        let position = usize::try_from(number).map_err(|_| {
            PyValueError::new_err(format!("argnums must be non-negative, got {number}"))
        })?;
        if positions.contains(&position) {
            return Err(PyValueError::new_err(format!(
                "argnums contains duplicate position {position}"
            )));
        }
        positions.push(position);
    }
    if positions.is_empty() {
        return Err(PyValueError::new_err("argnums must not be empty"));
    }
    Ok((positions.into(), single))
}

// ----- argnums --------------------------------------------------------------

/// A Python function called with its differentiated arguments split out:
/// the input graph is `((args at argnums...), ((other args...), kwargs))`.
/// This keeps the `wrt` spec independent of the number of arguments.
struct ArgnumsFunction {
    function: PythonFunction,
    argnums: Arc<[usize]>,
}

impl ArgnumsFunction {
    /// `(args, kwargs)` -> `(selected, (rest, kwargs))`.
    fn split(argnums: &[usize], input: PyGraph) -> PyResult<PyGraph> {
        let (args, kwargs) = call_parts(input);
        if let Some(&max) = argnums.iter().max() {
            if max >= args.len() {
                return Err(PyTypeError::new_err(format!(
                    "differentiating with respect to argnums={argnums:?} requires at least {} \
                     positional arguments, but got {}",
                    max + 1,
                    args.len()
                )));
            }
        }
        let mut args: Vec<Option<PyGraph>> = args.into_iter().map(Some).collect();
        let selected = argnums
            .iter()
            .map(|&i| args[i].take().expect("distinct argnums"))
            .collect();
        let rest = args.into_iter().flatten().collect();
        Ok(PyGraph::Tuple(vec![
            PyGraph::Tuple(selected),
            PyGraph::Tuple(vec![PyGraph::Tuple(rest), kwargs]),
        ]))
    }

    fn merge(&self, input: PyGraph) -> PyGraph {
        let [selected, others]: [PyGraph; 2] = tuple_items(input)
            .try_into()
            .ok()
            .expect("split call graph");
        let [rest, kwargs]: [PyGraph; 2] = tuple_items(others)
            .try_into()
            .ok()
            .expect("split call graph");
        let mut selected: Vec<_> = tuple_items(selected).into_iter().map(Some).collect();
        let mut rest = tuple_items(rest).into_iter();
        let count = selected.len() + rest.len();
        let args = (0..count)
            .map(
                |position| match self.argnums.iter().position(|&i| i == position) {
                    Some(k) => selected[k].take().expect("distinct argnums"),
                    None => rest.next().expect("remaining positional argument"),
                },
            )
            .collect();
        call_graph(args, kwargs)
    }
}

impl Function<PyGraph> for ArgnumsFunction {
    type Output = PyGraph;
}

impl TryCallable<PyGraph> for ArgnumsFunction {
    type Error = PyErr;

    fn try_invoke(&self, input: PyGraph) -> PyResult<PyGraph> {
        self.function.try_invoke(self.merge(input))
    }
}

impl TryJitable<PyGraph> for ArgnumsFunction {
    fn try_invoke_owned(&self, input: PyGraph) -> PyResult<PyGraph> {
        self.try_invoke(input)
    }
}

/// Differentiate the selected arguments (`true`), not the rest (`false`).
fn argnums_wrt() -> Dag<bool> {
    list_dag([Dag::from(true), Dag::from(false)])
}

/// The cotangents of the selected arguments from a cotangent of the split
/// input graph: one value for an int `argnums`, a tuple otherwise.
fn selected_cotangents(cotangents: PyGraph, single: bool) -> PyGraph {
    let [selected, _]: [PyGraph; 2] = tuple_items(cotangents)
        .try_into()
        .ok()
        .expect("split cotangent graph");
    if single {
        tuple_items(selected).pop().expect("one selected argument")
    } else {
        selected
    }
}

// ----- transformed callables ----------------------------------------------------

type GraphFnBox = Box<dyn Fn(PyGraph) -> PyResult<PyGraph> + Send + Sync>;

/// A transformation applied to the `(args, kwargs)` call graph.
pub(crate) struct GraphFn(GraphFnBox);

impl Function<PyGraph> for GraphFn {
    type Output = PyGraph;
}

impl TryCallable<PyGraph> for GraphFn {
    type Error = PyErr;

    fn try_invoke(&self, input: PyGraph) -> PyResult<PyGraph> {
        (self.0)(input)
    }
}

impl TryJitable<PyGraph> for GraphFn {
    fn try_invoke_owned(&self, input: PyGraph) -> PyResult<PyGraph> {
        self.try_invoke(input)
    }
}

/// A transformed Python function (`map`, `grad`, `value_and_grad`,
/// `hessian`). Calls are traced and cached like `jit`.
#[pyclass(name = "Transformed", module = "gnxlib")]
pub struct PyTransformed {
    name: &'static str,
    // Both Arcs share a single owned Python reference, visited once by the GC.
    function: Option<Arc<Py<PyAny>>>,
    jit: Option<Arc<Jit<PyGraph, GraphFn>>>,
}

#[pymethods]
impl PyTransformed {
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
        let input = GraphParser::parse_args(args, kwargs)?;
        let jit = self.jit.as_ref().ok_or_else(cleared)?.clone();
        py.detach(move || jit.try_invoke(input))?
            .into_pyobject_bound(py)
    }

    /// Inspect the trace for these arguments without calling it or updating cells.
    #[pyo3(signature = (*args, **kwargs))]
    fn trace(
        &self,
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyTrace> {
        let input = GraphParser::parse_args(args, kwargs)?;
        let jit = self.jit.as_ref().ok_or_else(cleared)?.clone();
        py.detach(move || jit.try_trace(input)).map(PyTrace)
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        let wrapped = self
            .function
            .as_ref()
            .and_then(|function| function.bind(py).repr().ok())
            .map(|repr| repr.to_string())
            .unwrap_or_else(|| "<cleared>".into());
        format!("<gnxlib.Transformed {} of {wrapped}>", self.name)
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

fn cleared() -> PyErr {
    PyRuntimeError::new_err("transformed function has been cleared")
}

fn callable(py: Python<'_>, function: Py<PyAny>, name: &str) -> PyResult<Arc<Py<PyAny>>> {
    if !function.bind(py).is_callable() {
        return Err(PyTypeError::new_err(format!("{name} expects a callable")));
    }
    Ok(Arc::new(function))
}

fn transformed(
    name: &'static str,
    function: Arc<Py<PyAny>>,
    body: impl Fn(PyGraph) -> PyResult<PyGraph> + Send + Sync + 'static,
) -> PyTransformed {
    PyTransformed {
        name,
        function: Some(function),
        jit: Some(Arc::new(gnx::jit(GraphFn(Box::new(body))))),
    }
}

/// Vectorize a function over leading (or chosen) axes of its arguments, like
/// `jax.vmap`. `in_axes` is an int or None for every argument, or a tuple with
/// one entry (a pytree prefix of ints and None) per positional argument.
/// Keyword arguments are always mapped over axis 0. `out_axes` is a pytree
/// prefix of ints for the result (None is not supported).
#[pyfunction(signature = (f, in_axes, out_axes, axis_size = None))]
pub fn map(
    py: Python<'_>,
    f: Py<PyAny>,
    in_axes: &Bound<'_, PyAny>,
    out_axes: &Bound<'_, PyAny>,
    axis_size: Option<usize>,
) -> PyResult<PyTransformed> {
    let function = callable(py, f, "map")?;
    let (in_dag, arity) =
        match in_axes {
            spec if spec.is_instance_of::<PyTuple>() || spec.is_instance_of::<PyList>() => {
                let Dag::Node(DagNode::List(args)) = axes_dag(spec, "in_axes")? else {
                    unreachable!("sequence specs are list nodes");
                };
                let arity = args.len();
                (
                    list_dag([Dag::Node(DagNode::List(args)), Dag::from(0)]),
                    Some(arity),
                )
            }
            spec if spec.is_none() || spec.is_instance_of::<PyInt>() => {
                (axes_dag(spec, "in_axes")?, None)
            }
            _ => return Err(PyTypeError::new_err(
                "in_axes must be an int, None, or a tuple with one entry per positional argument",
            )),
        };
    let out_dag = axes_dag(out_axes, "out_axes")?;
    let mut mapped = gnx::map(PythonFunction(function.clone()))
        .in_axes(in_dag)
        .out_axes(out_dag);
    if let Some(size) = axis_size {
        mapped = mapped.axis_size(size);
    }
    Ok(transformed("map", function, move |input| {
        if let Some(arity) = arity {
            let PyGraph::Tuple(call) = &input else {
                unreachable!("call graph is a tuple");
            };
            if let Some(PyGraph::Tuple(args)) = call.first() {
                if args.len() != arity {
                    return Err(PyValueError::new_err(format!(
                        "map in_axes has {arity} entries but got {} positional arguments",
                        args.len()
                    )));
                }
            }
        }
        mapped.try_invoke(input).map_err(apply_error)
    }))
}

/// The gradient of a function returning one floating-point scalar array,
/// with respect to the positional arguments at `argnums` (like `jax.grad`).
#[pyfunction]
pub fn grad(py: Python<'_>, f: Py<PyAny>, argnums: &Bound<'_, PyAny>) -> PyResult<PyTransformed> {
    let function = callable(py, f, "grad")?;
    let (argnums, single) = parse_argnums(argnums)?;
    let builder = gnx::grad(ArgnumsFunction {
        function: PythonFunction(function.clone()),
        argnums: argnums.clone(),
    })
    .wrt(argnums_wrt());
    Ok(transformed("grad", function, move |input| {
        let input = ArgnumsFunction::split(&argnums, input)?;
        let cotangents = builder.try_invoke(input).map_err(apply_error)?;
        Ok(selected_cotangents(cotangents, single))
    }))
}

/// Like `grad`, returning `(value, gradient)`.
#[pyfunction]
pub fn value_and_grad(
    py: Python<'_>,
    f: Py<PyAny>,
    argnums: &Bound<'_, PyAny>,
) -> PyResult<PyTransformed> {
    let function = callable(py, f, "value_and_grad")?;
    let (argnums, single) = parse_argnums(argnums)?;
    let builder = gnx::value_and_grad(ArgnumsFunction {
        function: PythonFunction(function.clone()),
        argnums: argnums.clone(),
    })
    .wrt(argnums_wrt());
    Ok(transformed("value_and_grad", function, move |input| {
        let input = ArgnumsFunction::split(&argnums, input)?;
        let (value, cotangents) = builder.try_invoke(input).map_err(apply_error)?;
        Ok(PyGraph::Tuple(vec![
            value,
            selected_cotangents(cotangents, single),
        ]))
    }))
}

/// The Hessian of a function returning one floating-point scalar array, with
/// respect to the single array argument at position `argnums` (an int). For
/// an argument of shape `S` the result has shape `S + S`.
#[pyfunction(signature = (f, argnums = 0))]
pub fn hessian(py: Python<'_>, f: Py<PyAny>, argnums: usize) -> PyResult<PyTransformed> {
    let function = callable(py, f, "hessian")?;
    let argnums: Arc<[usize]> = Arc::new([argnums]);
    let builder = gnx::hessian(ArgnumsFunction {
        function: PythonFunction(function.clone()),
        argnums: argnums.clone(),
    })
    .wrt(argnums_wrt());
    Ok(transformed("hessian", function, move |input| {
        let input = ArgnumsFunction::split(&argnums, input)?;
        let hessian: Array = builder.try_invoke(input).map_err(apply_error)?;
        Ok(PyGraph::Tracer(hessian.tracer().clone().into()))
    }))
}

// ----- jvp / vjp ----------------------------------------------------------------

fn positional(sequence: &Bound<'_, PyAny>, what: &str) -> PyResult<PyGraph> {
    let Ok(tuple) = sequence.cast::<PyTuple>() else {
        return Err(PyTypeError::new_err(format!("{what} must be a tuple")));
    };
    let args = GraphParser::parse_args(tuple, None)?;
    Ok(args)
}

/// Forward-mode differentiation (`jax.jvp`): evaluate `f(*primals)` and its
/// directional derivative along `tangents`. Returns `(out, tangent_out)`.
/// Tangents must have the primals' structure; non-floating-point results get
/// zero tangents.
#[pyfunction]
pub fn jvp<'py>(
    py: Python<'py>,
    f: Py<PyAny>,
    primals: &Bound<'py, PyAny>,
    tangents: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyTuple>> {
    let function = callable(py, f, "jvp")?;
    let primals = positional(primals, "primals")?;
    let tangents = positional(tangents, "tangents")?;
    let builder = gnx::jvp(PythonFunction(function));
    let (out, tangent) = py
        .detach(move || builder.try_invoke((primals, tangents)))
        .map_err(apply_error)?;
    PyTuple::new(
        py,
        [
            out.into_pyobject_bound(py)?,
            tangent.into_pyobject_bound(py)?,
        ],
    )
}

/// Reverse-mode differentiation (`jax.vjp`): returns `(out, vjp_fn)`, where
/// `vjp_fn(cotangent)` maps a cotangent with the structure of `out` to a tuple
/// with one cotangent per primal. Each `vjp_fn` call records one fused
/// reverse-mode op; the primal is traced once for `out` and once more for the
/// first `vjp_fn` call.
#[pyfunction(signature = (f, *primals))]
pub fn vjp<'py>(
    py: Python<'py>,
    f: Py<PyAny>,
    primals: &Bound<'py, PyTuple>,
) -> PyResult<Bound<'py, PyTuple>> {
    let function = callable(py, f, "vjp")?;
    let primals = GraphParser::parse_args(primals, None)?;
    let forward = PythonFunction(function.clone());
    let input = primals.clone();
    let out = py.detach(move || forward.try_invoke(input))?;
    let builder = gnx::vjp(PythonFunction(function.clone()));
    let pullback = PyVjpFunction {
        function: Some(function),
        builder: Some(Arc::new(builder)),
        primals,
    };
    PyTuple::new(
        py,
        [
            out.into_pyobject_bound(py)?,
            Bound::new(py, pullback)?.into_any(),
        ],
    )
}

type VjpBuilder = gnx::transforms::ad::Vjp<PyGraph, PythonFunction>;

/// The pullback returned by `vjp`.
#[pyclass(name = "VjpFunction", module = "gnxlib")]
pub struct PyVjpFunction {
    function: Option<Arc<Py<PyAny>>>,
    builder: Option<Arc<VjpBuilder>>,
    primals: PyGraph,
}

#[pymethods]
impl PyVjpFunction {
    fn __call__<'py>(
        &self,
        py: Python<'py>,
        cotangent: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let cotangent: PyGraph = cotangent.extract()?;
        let builder = self.builder.as_ref().ok_or_else(cleared)?.clone();
        let primals = self.primals.clone();
        let (_, cotangents) = py
            .detach(move || builder.try_invoke((primals, cotangent)))
            .map_err(apply_error)?;
        let (args, _) = call_parts(cotangents);
        PyGraph::Tuple(args).into_pyobject_bound(py)
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        if let Some(function) = &self.function {
            visit.call(function.as_ref())?;
        }
        Ok(())
    }

    fn __clear__(&mut self) {
        self.builder = None;
        self.function = None;
    }
}

//! Python bindings for structured control flow: `cond`, `switch`,
//! `while_loop` and `scan`, with graph (pytree) operands.
//!
//! Each Python callable is traced into a region `Capture` in its own
//! `TraceContext`, with placeholders that have the metadata of the operand
//! leaves. Values used from outside the region become its captures.

use gnx::array::{Array, ArrayInfo, DType, Dim, Shape};
use gnx::buffer::{DataHandle, HostBuffer};
use gnx::expr::{Capture, CellTrace, ObjectTracer, Trace, TraceContext, VarInfo};
use gnx::graph::util::{GraphExt, IterSource};
use gnx::graph::{Builder, Graph, GraphContext, GraphEq, GraphEqContext, Of, SourceError};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyInt, PyTuple};

use crate::graph::{PyBuilder, PyGraph};
use crate::tracer::extract_tracer;
use crate::transforms::lowering_error;

// ----- graph helpers ------------------------------------------------------------

fn split(graph: &PyGraph) -> PyResult<(PyBuilder, Vec<ObjectTracer>)> {
    let filter = &Of::<ObjectTracer>::filter();
    let builder = graph
        .builder::<ObjectTracer, _, SourceError>(filter, &mut GraphContext::new())
        .map_err(|error| PyTypeError::new_err(error.to_string()))?;
    Ok((builder, graph.collect_leaves(filter)))
}

fn rebuild(
    builder: &PyBuilder,
    leaves: impl IntoIterator<Item = ObjectTracer>,
) -> PyResult<PyGraph> {
    let mut leaves = leaves.into_iter();
    let graph = Builder::<ObjectTracer>::build(
        builder.clone(),
        IterSource::new(&mut leaves),
        &mut GraphContext::new(),
    )
    .map_err(|error| PyTypeError::new_err(error.to_string()))?;
    debug_assert!(leaves.next().is_none());
    Ok(graph)
}

fn same_structure(a: &PyBuilder, b: &PyBuilder) -> bool {
    PyGraph::builders_eq::<ObjectTracer>(a, b, &mut GraphEqContext::new())
}

/// The value leaves of a graph; control-flow carries can't be cells.
fn values(leaves: Vec<ObjectTracer>, what: &str) -> PyResult<Vec<Trace>> {
    leaves
        .into_iter()
        .map(|leaf| match leaf {
            ObjectTracer::Ref(trace) => Ok(trace),
            ObjectTracer::Cell(_) => Err(PyTypeError::new_err(format!(
                "{what} cannot contain an ArrayRef"
            ))),
        })
        .collect()
}

fn placeholder(leaf: &ObjectTracer) -> ObjectTracer {
    match leaf {
        ObjectTracer::Ref(trace) => Trace::placeholder(trace.info().clone()).into(),
        ObjectTracer::Cell(cell) => CellTrace::placeholder(cell.type_info().clone()).into(),
    }
}

/// A scalar selector: a GNX array, or a Python bool/int made concrete.
fn selector(value: &Bound<'_, PyAny>, dtype: DType) -> PyResult<Trace> {
    if let Some(ObjectTracer::Ref(trace)) = extract_tracer(value) {
        return Ok(trace);
    }
    let bytes = if value.is_instance_of::<PyBool>() && dtype == DType::Bool {
        vec![u8::from(value.extract::<bool>()?)]
    } else if value.is_instance_of::<PyInt>() && dtype == DType::I32 {
        value.extract::<i32>()?.to_le_bytes().to_vec()
    } else {
        let expected = if dtype == DType::Bool {
            "bool"
        } else {
            "int32"
        };
        return Err(PyTypeError::new_err(format!(
            "expected a scalar {expected} array"
        )));
    };
    let info = ArrayInfo::new(Shape::fixed([]), dtype);
    let host = HostBuffer::copy_from_slice(info, &bytes)?;
    Ok(Array::from_data(DataHandle::new(host))
        .tracer()
        .trace_ref()
        .clone())
}

// ----- regions --------------------------------------------------------------------

/// A traced region and the structure of its result.
struct Region {
    capture: Capture,
    output: PyBuilder,
}

/// Trace `function(*operands)` with operands of the given structure.
fn trace_branch(
    py: Python<'_>,
    function: &Bound<'_, PyAny>,
    builder: &PyBuilder,
    leaves: &[ObjectTracer],
) -> PyResult<Region> {
    if !function.is_callable() {
        return Err(PyTypeError::new_err("branches must be callable"));
    }
    trace_region_with(leaves, |placeholders| {
        let operands = rebuild(builder, placeholders.iter().cloned())?
            .into_pyobject_bound(py)?
            .cast_into::<PyTuple>()?;
        let result: PyGraph = function.call1(operands)?.extract()?;
        split(&result)
    })
}

/// Like [`trace_region`], creating one placeholder per template leaf.
fn trace_region_with(
    templates: &[ObjectTracer],
    call: impl FnOnce(&[ObjectTracer]) -> PyResult<(PyBuilder, Vec<ObjectTracer>)>,
) -> PyResult<Region> {
    let guard = TraceContext::enter();
    let inputs: Vec<_> = templates.iter().map(placeholder).collect();
    let (output, outputs) = call(&inputs)?;
    let capture = Capture::from_context(guard.context(), &inputs, &outputs)
        .map_err(|error| PyValueError::new_err(format!("could not capture a region: {error}")))?;
    Ok(Region { capture, output })
}

fn results<'py>(
    py: Python<'py>,
    builder: &PyBuilder,
    traces: Vec<Trace>,
) -> PyResult<Bound<'py, PyAny>> {
    rebuild(builder, traces.into_iter().map(ObjectTracer::from))?.into_pyobject_bound(py)
}

fn branches<'py>(
    py: Python<'py>,
    functions: &[Bound<'py, PyAny>],
    operands: &Bound<'py, PyTuple>,
) -> PyResult<(Vec<Region>, Vec<ObjectTracer>)> {
    let operands: PyGraph = operands.extract()?;
    let (builder, leaves) = split(&operands)?;
    let regions = functions
        .iter()
        .map(|function| trace_branch(py, function, &builder, &leaves))
        .collect::<PyResult<Vec<_>>>()?;
    if let Some((first, rest)) = regions.split_first() {
        if rest
            .iter()
            .any(|region| !same_structure(&first.output, &region.output))
        {
            return Err(PyTypeError::new_err(
                "branches must return results of the same structure",
            ));
        }
    }
    Ok((regions, leaves))
}

// ----- operations -----------------------------------------------------------------

/// Call `true_fun(*operands)` if `pred` else `false_fun(*operands)`, like
/// `jax.lax.cond`. `pred` is a scalar bool array (or Python bool). Both
/// branches are traced and must return the same structure of arrays.
#[pyfunction(signature = (pred, true_fun, false_fun, *operands))]
pub fn cond<'py>(
    py: Python<'py>,
    pred: &Bound<'py, PyAny>,
    true_fun: Bound<'py, PyAny>,
    false_fun: Bound<'py, PyAny>,
    operands: &Bound<'py, PyTuple>,
) -> PyResult<Bound<'py, PyAny>> {
    let pred = selector(pred, DType::Bool)?;
    let (regions, leaves) = branches(py, &[true_fun, false_fun], operands)?;
    let traces = gnx::transforms::cf::cond(pred, &regions[0].capture, &regions[1].capture, &leaves)
        .map_err(lowering_error)?;
    results(py, &regions[0].output, traces)
}

/// Call `branches[index](*operands)`, like `jax.lax.switch`. `index` is a
/// scalar int32 array (or Python int), clamped to the valid range.
#[pyfunction(signature = (index, branches, *operands))]
pub fn switch<'py>(
    py: Python<'py>,
    index: &Bound<'py, PyAny>,
    branches: &Bound<'py, PyAny>,
    operands: &Bound<'py, PyTuple>,
) -> PyResult<Bound<'py, PyAny>> {
    let index = selector(index, DType::I32)?;
    let functions = branches.try_iter()?.collect::<PyResult<Vec<_>>>()?;
    if functions.is_empty() {
        return Err(PyValueError::new_err("switch needs at least one branch"));
    }
    let (regions, leaves) = self::branches(py, &functions, operands)?;
    let captures: Vec<_> = regions.iter().map(|region| &region.capture).collect();
    let traces = gnx::transforms::cf::switch(index, &captures, &leaves).map_err(lowering_error)?;
    results(py, &regions[0].output, traces)
}

/// Repeat `val = body_fun(val)` while `cond_fun(val)`, like
/// `jax.lax.while_loop`. `cond_fun` returns a scalar bool array; `body_fun`
/// returns the structure, shapes and dtypes of `init_val`.
#[pyfunction]
pub fn while_loop<'py>(
    py: Python<'py>,
    cond_fun: Bound<'py, PyAny>,
    body_fun: Bound<'py, PyAny>,
    init_val: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let init: PyGraph = init_val.extract()?;
    let (builder, leaves) = split(&init)?;
    let carry = values(leaves.clone(), "while_loop carry")?;
    let trace = |function: &Bound<'py, PyAny>, predicate: bool| {
        if !function.is_callable() {
            return Err(PyTypeError::new_err("while_loop expects callables"));
        }
        trace_region_with(&leaves, |placeholders| {
            let value = rebuild(&builder, placeholders.iter().cloned())?.into_pyobject_bound(py)?;
            let result = function.call1((value,))?;
            if predicate && result.is_instance_of::<PyBool>() {
                let constant = selector(&result, DType::Bool)?;
                return split(&PyGraph::Tracer(constant.into()));
            }
            split(&result.extract()?)
        })
    };
    let predicate = trace(&cond_fun, true)?;
    let body = trace(&body_fun, false)?;
    if !same_structure(&builder, &body.output) {
        return Err(PyTypeError::new_err(
            "while_loop body_fun must return the structure of init_val",
        ));
    }
    let traces =
        gnx::transforms::cf::while_loop_captured(&predicate.capture, &body.capture, &carry)
            .map_err(lowering_error)?;
    results(py, &builder, traces)
}

/// The metadata of one slice along the leading axis.
fn slice_info(trace: &Trace) -> PyResult<VarInfo> {
    let error = || PyTypeError::new_err("scan xs must be arrays with a fixed leading axis");
    let array = ArrayInfo::from_info(trace.info()).ok_or_else(error)?;
    let [Dim::Fixed(_), tail @ ..] = array.shape().dims() else {
        return Err(error());
    };
    Ok(VarInfo::new(ArrayInfo::new(
        tail.iter().copied().collect::<gnx::Shape>(),
        array.dtype(),
    )))
}

/// Scan `f(carry, x) -> (carry, y)` over the leading axis of `xs`, like
/// `jax.lax.scan`. Returns `(final_carry, stacked_ys)`. `xs` may be None when
/// `length` is given.
#[pyfunction(signature = (f, init, xs = None, length = None))]
pub fn scan<'py>(
    py: Python<'py>,
    f: Bound<'py, PyAny>,
    init: &Bound<'py, PyAny>,
    xs: Option<&Bound<'py, PyAny>>,
    length: Option<usize>,
) -> PyResult<Bound<'py, PyTuple>> {
    if !f.is_callable() {
        return Err(PyTypeError::new_err("scan expects a callable"));
    }
    let init: PyGraph = init.extract()?;
    let xs: PyGraph = match xs {
        Some(xs) => xs.extract()?,
        None => py.None().into_bound(py).extract()?,
    };
    let (carry_builder, carry_leaves) = split(&init)?;
    let (xs_builder, xs_leaves) = split(&xs)?;
    let carry = values(carry_leaves, "scan carry")?;
    let xs = values(xs_leaves, "scan xs")?;
    let mut templates: Vec<ObjectTracer> = carry.iter().cloned().map(ObjectTracer::from).collect();
    for x in &xs {
        templates.push(Trace::placeholder(slice_info(x)?).into());
    }
    let carry_len = carry.len();
    let mut ys_builder = None;
    let region = trace_region_with(&templates, |placeholders| {
        let (carry, x) = placeholders.split_at(carry_len);
        let carry = rebuild(&carry_builder, carry.iter().cloned())?.into_pyobject_bound(py)?;
        let x = rebuild(&xs_builder, x.iter().cloned())?.into_pyobject_bound(py)?;
        let result = f.call1((carry, x))?;
        let pair = result
            .cast::<PyTuple>()
            .ok()
            .filter(|pair| pair.len() == 2)
            .ok_or_else(|| PyTypeError::new_err("scan f must return a (carry, y) tuple"))?;
        let (new_builder, mut outputs) = split(&pair.get_item(0)?.extract()?)?;
        if !same_structure(&carry_builder, &new_builder) {
            return Err(PyTypeError::new_err(
                "scan f must return a carry with the structure of init",
            ));
        }
        let (y_builder, ys) = split(&pair.get_item(1)?.extract()?)?;
        outputs.extend(ys);
        ys_builder = Some(y_builder);
        Ok((new_builder, outputs))
    })?;
    let ys_builder = ys_builder.expect("scan body was traced");
    let (carry, ys) =
        gnx::transforms::cf::scan(&region.capture, &carry, &xs, length).map_err(lowering_error)?;
    PyTuple::new(
        py,
        [
            results(py, &carry_builder, carry)?,
            results(py, &ys_builder, ys)?,
        ],
    )
}

//! The jit-based core shared by the transformation wrappers ([`crate::map`],
//! [`crate::ad`]): split argument graphs into tracer leaves, trace the wrapped
//! function once per signature, and rebuild result graphs.

use std::sync::Arc;

use gnx_expr::{
    Capture, CellTrace, Invocation, Kind, ObjectTracer, Op, Trace, TraceContext, ValueInfo,
};
use gnx_graph::util::GraphExt;
use gnx_graph::{Builder, Graph, Key, Of};
use gnx_math_base::ops::{ones_op, zeros_op};

use crate::body::Body;
use crate::cache::{TraceEntry, TracerBuilder, kind, rebuild_graph, split_graph};
use crate::spec::Resolved;
use crate::{TransformError, TransformResult, util};

/// An argument graph split into its structure and tracer leaves.
pub(crate) struct Leaves<I: Graph> {
    pub builder: TracerBuilder<I>,
    pub objects: Vec<ObjectTracer>,
    pub paths: Vec<Vec<Key>>,
}

impl<I: Graph> Leaves<I> {
    pub fn of<A>(graph: &A) -> Self
    where
        A: Graph<Builder<ObjectTracer> = TracerBuilder<I>>,
    {
        let (builder, objects) = split_graph(graph);
        let paths: Vec<_> = graph
            .collect_leaves_with_paths(&Of::<ObjectTracer>::filter())
            .into_iter()
            .map(|(path, _)| path)
            .collect();
        assert_eq!(
            paths.len(),
            objects.len(),
            "leaf paths follow builder order"
        );
        Self {
            builder,
            objects,
            paths,
        }
    }

    pub fn kinds(&self) -> Vec<Kind> {
        self.objects.iter().map(kind).collect()
    }

    /// Rebuild a graph of this structure from new leaves.
    pub fn rebuild(
        &self,
        leaves: impl IntoIterator<Item = ObjectTracer>,
    ) -> <TracerBuilder<I> as Builder<ObjectTracer>>::Graph {
        rebuild_graph(&self.builder, leaves)
    }
}

/// The tracer leaves of a graph, without its structure.
pub(crate) fn objects<G: Graph>(graph: &G) -> Vec<ObjectTracer> {
    split_graph(graph).1
}

/// The traced body with the flagged inputs moved into its closure, bound to
/// the corresponding argument leaves.
pub(crate) fn body<O: Graph>(
    entry: &TraceEntry<O>,
    objects: &[ObjectTracer],
    captured: &[bool],
) -> Body {
    let expr = entry.expr.capture_inputs(captured);
    let captures = entry
        .closure
        .iter()
        .chain(
            objects
                .iter()
                .zip(captured)
                .filter(|(_, captured)| **captured)
                .map(|(object, _)| object),
        )
        .cloned()
        .collect();
    Body::new(Arc::new(expr), captures)
}

/// A value leaf; transformations never take cells as explicit operands.
pub(crate) fn value(object: &ObjectTracer, what: &str) -> TransformResult<Trace> {
    match object {
        ObjectTracer::Ref(trace) => Ok(trace.clone()),
        ObjectTracer::Cell(_) => Err(TransformError::invalid(format!(
            "{what} cannot be a mutable cell"
        ))),
    }
}

pub(crate) fn constant(op: Op, info: ValueInfo) -> Trace {
    Invocation::invoke(op, vec![], Vec::<Trace>::new(), vec![info])
        .pop()
        .expect("one result")
        .into_trace_ref()
}

pub(crate) fn zeros(info: &ValueInfo) -> Trace {
    constant(zeros_op(), info.clone())
}

pub(crate) fn ones(info: &ValueInfo) -> Trace {
    constant(ones_op(), info.clone())
}

/// A zero cotangent in the place of an argument leaf.
pub(crate) fn zeros_like(object: &ObjectTracer) -> ObjectTracer {
    match object {
        ObjectTracer::Ref(trace) => zeros(trace.info()).into(),
        ObjectTracer::Cell(cell) => CellTrace::new(zeros(cell.type_info())).into(),
    }
}

/// Which argument leaves to differentiate: those selected by `wrt` that are
/// floating-point values. Selecting another leaf directly is an error; a
/// selection that only covers it (through an ancestor) treats it as constant.
pub(crate) fn differentiated(
    objects: &[ObjectTracer],
    paths: &[Vec<Key>],
    wrt: &[Resolved<bool>],
) -> TransformResult<Vec<bool>> {
    objects
        .iter()
        .zip(paths)
        .zip(wrt)
        .map(|((object, path), selected)| {
            if selected.value != Some(true) {
                return Ok(false);
            }
            let differentiable =
                matches!(object, ObjectTracer::Ref(trace) if util::is_inexact(trace.info()));
            if !differentiable && selected.exact {
                return Err(TransformError::invalid(format!(
                    "wrt selects a leaf that is not a floating-point array ({path:?})"
                )));
            }
            Ok(differentiable)
        })
        .collect()
}

/// Trace a closure of placeholder values into a region capture.
pub(crate) fn trace_region(
    infos: &[ValueInfo],
    build: impl FnOnce(&[Trace]) -> TransformResult<Vec<Trace>>,
) -> TransformResult<Capture> {
    let guard = TraceContext::enter();
    let args: Vec<_> = infos.iter().cloned().map(Trace::placeholder).collect();
    let outputs = build(&args)?;
    let args: Vec<ObjectTracer> = args.into_iter().map(ObjectTracer::from).collect();
    let outputs: Vec<ObjectTracer> = outputs.into_iter().map(ObjectTracer::from).collect();
    Capture::from_context(guard.context(), &args, &outputs)
        .map_err(|error| TransformError::invalid(format!("could not capture a region: {error:?}")))
}

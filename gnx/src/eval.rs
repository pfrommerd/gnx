use std::future::Future;

use gnx_expr::{Capture, ObjectTracer};
use gnx_graph::util::GraphExt;
use gnx_graph::{Graph, Of};
use gnx_math::{Device, ExecOpts};

pub use gnx_expr::{CaptureError, EvalPlanError};

/// Capture the pending expressions reachable from a graph without executing
/// them. Cell roots contribute their current home values.
pub fn expr<G: Graph>(roots: &G) -> Result<Capture, CaptureError> {
    let roots = roots.collect_leaves(Of::<ObjectTracer>::filter());
    Capture::from_objects(roots.iter().map(ObjectTracer::as_ref))
}

#[derive(Debug, thiserror::Error)]
pub enum EvalError {
    #[error(transparent)]
    Capture(#[from] CaptureError),
    #[error(transparent)]
    Plan(#[from] EvalPlanError),
    #[error("device execution failed: {0}")]
    Backend(#[source] std::io::Error),
}

/// Materialize the existing tracers in a graph, without changing their
/// identity. Concrete roots are left where they already reside.
pub trait Eval {
    fn eval<G: Graph>(
        &self,
        roots: &G,
        opts: ExecOpts,
    ) -> impl Future<Output = Result<(), EvalError>> + Send;
}

impl<D: Device + ?Sized> Eval for D {
    fn eval<G: Graph>(
        &self,
        roots: &G,
        opts: ExecOpts,
    ) -> impl Future<Output = Result<(), EvalError>> + Send {
        let prepared = (|| Ok::<_, EvalError>(expr(roots)?.prepare()?))();
        async move {
            let (expr, args, commit) = prepared?.into_parts();
            if expr.outputs().is_empty() {
                return Ok(commit.publish([])?);
            }
            let outputs = self
                .execute(expr, args, opts)
                .await
                .map_err(EvalError::Backend)?;
            Ok(commit.publish(outputs)?)
        }
    }
}

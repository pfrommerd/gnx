mod capture;
mod cell;
mod context;
mod trace_ref;

pub use capture::{Capture, CaptureError, TracerKey};
pub use cell::{CellKey, CellTrace, CellTracer, TraceCell};
pub use context::{ContextID, TraceContext, TraceContextGuard};
pub use trace_ref::{Trace, TraceData, Tracer};

use std::fmt::Debug;
use std::sync::Arc;

use crate::{Expr, Op};

pub use super::value::{ConcreteValue, Generic, Traceable, Value, ValueInfo};

/// A traced immutable value or shared mutable cell handle.
#[derive(Clone, Debug)]
pub enum ObjectTracer {
    Ref(Trace),
    Cell(CellTrace),
}

/// A borrowed, copyable view of an immutable trace or a cell handle. Creating
/// this view never reads cell contents or records a trace node.
#[derive(Clone, Copy, Debug)]
pub enum ObjectTracerRef<'a> {
    Ref(&'a Trace),
    Cell(&'a CellTrace),
}

impl ObjectTracerRef<'_> {
    /// Clone the handle while preserving the original value/cell identity.
    pub fn to_owned(self) -> ObjectTracer {
        match self {
            Self::Ref(value) => ObjectTracer::Ref(value.clone()),
            Self::Cell(cell) => ObjectTracer::Cell(cell.clone()),
        }
    }
}

impl ObjectTracer {
    pub fn as_ref(&self) -> ObjectTracerRef<'_> {
        match self {
            Self::Ref(value) => ObjectTracerRef::Ref(value),
            Self::Cell(cell) => ObjectTracerRef::Cell(cell),
        }
    }

    /// A cell's home is a value snapshot, not an observable read of a foreign cell.
    pub fn resolve(&self) -> Trace {
        match self {
            Self::Ref(r) => r.clone(),
            Self::Cell(c) => c.read_home(),
        }
    }
}

impl<T: Traceable> From<Tracer<T>> for ObjectTracer {
    fn from(t: Tracer<T>) -> Self {
        Self::Ref(t.into_trace_ref())
    }
}
impl<T: Traceable> From<CellTracer<T>> for ObjectTracer {
    fn from(c: CellTracer<T>) -> Self {
        Self::Cell(c.into())
    }
}
impl From<Trace> for ObjectTracer {
    fn from(r: Trace) -> Self {
        Self::Ref(r)
    }
}
impl From<CellTrace> for ObjectTracer {
    fn from(c: CellTrace) -> Self {
        Self::Cell(c)
    }
}

/// An immutable computation argument. Read a cell explicitly with get() before
/// passing its value to a computation; writes always use the cell API.
///
/// ```compile_fail
/// use gnx_expr::{CellTrace, TraceInput, ValueInfo};
/// let cell = CellTrace::placeholder(ValueInfo::new(0usize));
/// let argument: TraceInput = cell.into(); // No implicit read or access annotation.
/// ```
///
/// ```compile_fail
/// use gnx_expr::{Generic, TraceInput, ObjectTracer, Tracer, ValueInfo};
/// let value = Tracer::<Generic>::placeholder(ValueInfo::new(0usize));
/// let object = ObjectTracer::from(value);
/// let argument: TraceInput = object.into(); // Match the object explicitly first.
/// ```
#[derive(Clone)]
pub struct TraceInput(Trace);
impl TraceInput {
    fn into_value(self) -> Trace {
        self.0
    }
}
impl From<Trace> for TraceInput {
    fn from(value: Trace) -> Self {
        Self(value)
    }
}
impl<T: Traceable> From<Tracer<T>> for TraceInput {
    fn from(value: Tracer<T>) -> Self {
        Self(value.into_trace_ref())
    }
}

pub enum TraceOutput {
    Ref(ValueInfo),
}

/// A node in the trace DAG. Computations can only consume immutable value
/// references. Cell accesses have dedicated, typed operands and ordering edges;
/// they are not registered dialect operations.
pub enum Invocation {
    Apply {
        context_id: ContextID,
        op: Op,
        subexprs: Vec<(Arc<Expr>, Vec<Trace>)>,
        inputs: Vec<Trace>,
        outputs: Vec<ValueInfo>,
    },
    ReadCell {
        context_id: ContextID,
        cell: CellTrace,
        aliasing_parents: Vec<Arc<Invocation>>,
    },
    WriteCell {
        context_id: ContextID,
        cell: CellTrace,
        value: Trace,
        aliasing_parents: Vec<Arc<Invocation>>,
    },
}

impl Debug for Invocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Apply { context_id, op, .. } => f
                .debug_struct("Apply")
                .field("op", op)
                .field("ctx", context_id)
                .finish(),
            Self::ReadCell { context_id, .. } => {
                f.debug_struct("ReadCell").field("ctx", context_id).finish()
            }
            Self::WriteCell { context_id, .. } => f
                .debug_struct("WriteCell")
                .field("ctx", context_id)
                .finish(),
        }
    }
}

impl Invocation {
    pub fn context_id(&self) -> &ContextID {
        match self {
            Self::Apply { context_id, .. }
            | Self::ReadCell { context_id, .. }
            | Self::WriteCell { context_id, .. } => context_id,
        }
    }
    pub fn subexprs(&self) -> &[(Arc<Expr>, Vec<Trace>)] {
        match self {
            Self::Apply { subexprs, .. } => subexprs,
            _ => &[],
        }
    }
    pub fn inputs(&self) -> &[Trace] {
        match self {
            Self::Apply { inputs, .. } => inputs,
            _ => &[],
        }
    }
    /// Immutable data predecessors, including nested captures and write payloads.
    pub fn value_parents(&self) -> impl Iterator<Item = &Trace> {
        let value = match self {
            Self::WriteCell { value, .. } => Some(value),
            _ => None,
        };
        self.inputs()
            .iter()
            .chain(self.subexprs().iter().flat_map(|(_, captures)| captures))
            .chain(value)
    }
    pub fn aliasing_parents(&self) -> &[Arc<Invocation>] {
        match self {
            Self::ReadCell {
                aliasing_parents, ..
            }
            | Self::WriteCell {
                aliasing_parents, ..
            } => aliasing_parents,
            Self::Apply { .. } => &[],
        }
    }
    pub fn outputs(&self) -> &[ValueInfo] {
        match self {
            Self::Apply { outputs, .. } => outputs,
            Self::ReadCell { cell, .. } => std::slice::from_ref(cell.type_info()),
            Self::WriteCell { .. } => &[],
        }
    }

    pub fn invoke<I>(
        op: Op,
        subexprs: Vec<(Arc<Expr>, Vec<TraceInput>)>,
        inputs: Vec<I>,
        outputs: Vec<ValueInfo>,
    ) -> Vec<Tracer<Generic>>
    where
        I: Into<TraceInput>,
    {
        for (expr, closure) in &subexprs {
            assert!(expr.check().is_ok(), "invalid immutable computation body");
            assert_eq!(
                closure.len(),
                expr.closure().len(),
                "subexpression capture arity mismatch"
            );
        }
        let inputs = inputs.into_iter().map(|i| i.into().into_value()).collect();
        let subexprs = subexprs
            .into_iter()
            .map(|(expr, closure)| {
                (
                    expr,
                    closure.into_iter().map(TraceInput::into_value).collect(),
                )
            })
            .collect();
        let invocation = Arc::new(Self::Apply {
            context_id: TraceContext::current_id(),
            op,
            subexprs,
            inputs,
            outputs,
        });
        Self::returned_values(invocation)
            .into_iter()
            .map(Tracer::from)
            .collect()
    }

    pub fn invoke_with_subexprs(
        op: Op,
        subexprs: Vec<(Arc<Expr>, Vec<TraceInput>)>,
        inputs: Vec<TraceInput>,
        outputs: Vec<TraceOutput>,
    ) -> Vec<ObjectTracer> {
        Self::invoke(
            op,
            subexprs,
            inputs,
            outputs
                .into_iter()
                .map(|o| match o {
                    TraceOutput::Ref(info) => info,
                })
                .collect(),
        )
        .into_iter()
        .map(ObjectTracer::from)
        .collect()
    }

    pub(crate) fn read_cell(cell: CellTrace) -> Trace {
        let context = TraceContext::current();
        let aliasing_parents = context
            .as_ref()
            .map(|ctx| ctx.aliasing_parents(false))
            .unwrap_or_default();
        let invocation = Arc::new(Self::ReadCell {
            context_id: TraceContext::current_id(),
            cell,
            aliasing_parents,
        });
        let value = Self::returned_values(invocation.clone()).pop().unwrap();
        if let Some(ctx) = context {
            ctx.note_aliased_read(invocation);
        }
        value
    }

    pub(crate) fn write_cell(cell: CellTrace, value: Trace) {
        let context = TraceContext::current();
        let aliasing_parents = context
            .as_ref()
            .map(|ctx| ctx.aliasing_parents(true))
            .unwrap_or_default();
        let invocation = Arc::new(Self::WriteCell {
            context_id: TraceContext::current_id(),
            cell,
            value,
            aliasing_parents,
        });
        if let Some(ctx) = context {
            ctx.note_aliased_write(invocation);
        }
    }

    fn returned_values(invocation: Arc<Self>) -> Vec<Trace> {
        invocation
            .outputs()
            .iter()
            .enumerate()
            .map(|(index, info)| Trace::returned(invocation.clone(), index, info.clone()))
            .collect()
    }
}

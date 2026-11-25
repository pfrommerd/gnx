mod apply;
mod capture;
mod cell;
mod context;
mod trace_ref;

pub use apply::{Apply, ApplyLocal, ApplyPlan, ApplyProjection, ApplyRegion};
pub use capture::{Capture, CaptureError, TracerKey};
pub use cell::{CellKey, CellTrace, CellTracer, TraceCell};
pub use context::{ContextID, TraceContext, TraceContextGuard};
pub use trace_ref::{Trace, TraceData, Tracer};

use std::fmt::Debug;
use std::sync::{Arc, OnceLock, Weak};

use crate::{HExprHandle, Op};

pub use crate::value::{ConcreteValue, Generic, Traceable, Value};
pub use crate::var::{Info, VarInfo};

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
/// use gnx_expr::{CellTrace, TraceInput, VarInfo};
/// let cell = CellTrace::placeholder(VarInfo::new(0usize));
/// let argument: TraceInput = cell.into(); // No implicit read or access annotation.
/// ```
///
/// ```compile_fail
/// use gnx_expr::{Generic, TraceInput, ObjectTracer, Tracer, VarInfo};
/// let value = Tracer::<Generic>::placeholder(VarInfo::new(0usize));
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

/// A node in the trace DAG. Nested bodies are always high-level expressions.
/// Cell effects have typed operands and ordering edges.
pub enum Invocation {
    Apply(Apply),
    ReadCell {
        context_id: ContextID,
        cell: CellTrace,
        aliasing_parents: Vec<Arc<Invocation>>,
        result_slots: OnceLock<Box<[Weak<TraceData>]>>,
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
            Self::Apply(apply) => f
                .debug_struct("Apply")
                .field("op", &apply.op)
                .field("ctx", &apply.context_id)
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
    fn slots(&self) -> Option<&OnceLock<Box<[Weak<TraceData>]>>> {
        match self {
            Self::Apply(apply) => Some(&apply.result_slots),
            Self::ReadCell { result_slots, .. } => Some(result_slots),
            Self::WriteCell { .. } => None,
        }
    }

    pub(super) fn result_slots(&self) -> &[Weak<TraceData>] {
        self.slots()
            .and_then(OnceLock::get)
            .map_or(&[], Box::as_ref)
    }

    fn register_consumers(invocation: &Arc<Self>) {
        for parent in invocation.value_parents() {
            parent.add_consumer(invocation);
        }
    }

    pub fn context_id(&self) -> &ContextID {
        match self {
            Self::Apply(apply) => &apply.context_id,
            Self::ReadCell { context_id, .. } | Self::WriteCell { context_id, .. } => context_id,
        }
    }
    /// Immutable data predecessors, including nested captures and write payloads.
    pub fn value_parents(&self) -> impl Iterator<Item = &Trace> {
        match self {
            Self::Apply(apply) => apply.value_parents.as_slice(),
            Self::ReadCell { .. } => &[],
            Self::WriteCell { value, .. } => std::slice::from_ref(value),
        }
        .iter()
    }
    pub fn aliasing_parents(&self) -> &[Arc<Invocation>] {
        match self {
            Self::Apply(apply) => &apply.aliasing_parents,
            Self::ReadCell {
                aliasing_parents, ..
            }
            | Self::WriteCell {
                aliasing_parents, ..
            } => aliasing_parents,
        }
    }
    pub fn outputs(&self) -> &[VarInfo] {
        match self {
            Self::Apply(apply) => &apply.outputs,
            Self::ReadCell { cell, .. } => std::slice::from_ref(cell.type_info()),
            Self::WriteCell { .. } => &[],
        }
    }

    /// Apply `op` to values, with value results. Each region is given with
    /// the values bound to its closure.
    pub fn invoke<I>(
        op: Op,
        subexprs: Vec<(HExprHandle, Vec<TraceInput>)>,
        inputs: Vec<I>,
        outputs: Vec<VarInfo>,
    ) -> Vec<Tracer<Generic>>
    where
        I: Into<TraceInput>,
    {
        let object = |input: TraceInput| ObjectTracer::Ref(input.into_value());
        let regions = subexprs
            .into_iter()
            .map(|(body, closure)| {
                ApplyRegion::new(body, closure.into_iter().map(object).collect())
            })
            .collect();
        let inputs = (inputs.into_iter())
            .map(|input| object(input.into()))
            .collect();
        Self::apply(op, regions, inputs, outputs)
            .into_iter()
            .map(Tracer::from)
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
            result_slots: OnceLock::new(),
        });
        let value = Self::returned_values(invocation.clone()).pop().unwrap();
        if let Some(ctx) = context {
            ctx.note_aliased_read(invocation);
        }
        value
    }

    pub(crate) fn write_cell(cell: CellTrace, mut value: Trace) {
        value.make_internal();
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
        Self::register_consumers(&invocation);
        if let Some(ctx) = context {
            ctx.note_aliased_write(invocation);
        }
    }

    fn returned_values(invocation: Arc<Self>) -> Vec<Trace> {
        let results: Vec<_> = invocation
            .outputs()
            .iter()
            .enumerate()
            .map(|(index, info)| Trace::returned(&invocation, index, info.clone()))
            .collect();
        if let Some(slots) = invocation.slots() {
            // Results retain their producer, so the producer only keeps weak
            // slots, initialized as a batch before exposing those results.
            let weak = results
                .iter()
                .map(|trace| Arc::downgrade(&trace.data))
                .collect();
            slots.set(weak).expect("invocation results initialize once");
        }
        Self::register_consumers(&invocation);
        results
    }
}

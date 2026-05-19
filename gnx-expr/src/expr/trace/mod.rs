mod cell;
mod context;

pub use cell::{CellKey, TraceCell, TraceCellRef, TracerCell};
pub use context::{CellUpdate, ContextID, TraceContext, TraceContextGuard};

use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::Deref;
use std::sync::{Arc, OnceLock, Weak};

use castaway::LifetimeFree;

use crate::expr::Op;
use crate::util::DgArc;

pub use super::value::{
    ConcreteValue, Generic, Traceable, Value, ValueInfo,
};

/// Where a traced value is produced relative to an [`Invocation`].
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum Position {
    Return(usize),
    ClosureUpdate(usize),
    InputUpdate(usize),
}

impl Position {
    pub fn return_index(self) -> Option<usize> {
        match self {
            Position::Return(i) => Some(i),
            _ => None,
        }
    }
}

// An abstract trace value cannot be updated with a concrete value.
enum AbstractTrace {
    Placeholder,
    Computed {
        invocation: Arc<Invocation>,
        position: Position,
    },
}

struct UpdatableTrace {
    invocation: DgArc<Invocation>,
    position: Position,
    value: OnceLock<Value>,
}

enum TraceInner {
    Abstract(AbstractTrace),
    Updatable(UpdatableTrace),
    Constant(Value),
}

pub struct Trace {
    inner: TraceInner,
    info: ValueInfo,
    context_id: ContextID,
}

/// Shared handle to an immutable [`Trace`].
#[repr(transparent)]
#[derive(Clone)]
pub struct TraceRef(pub(crate) Arc<Trace>);

impl TraceRef {
    pub fn new(inner: Arc<Trace>) -> Self {
        TraceRef(inner)
    }
}

impl Deref for TraceRef {
    type Target = Trace;

    fn deref(&self) -> &Trace {
        &self.0
    }
}

impl From<Arc<Trace>> for TraceRef {
    fn from(value: Arc<Trace>) -> Self {
        TraceRef(value)
    }
}

impl From<TraceRef> for Arc<Trace> {
    fn from(value: TraceRef) -> Self {
        value.0
    }
}

impl AsRef<Trace> for TraceRef {
    fn as_ref(&self) -> &Trace {
        self
    }
}

impl Debug for TraceRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&*self.0, f)
    }
}

unsafe impl LifetimeFree for TraceRef {}

pub type WeakTraceRef = Weak<Trace>;

impl Debug for Trace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.inner {
            TraceInner::Abstract(AbstractTrace::Placeholder) => {
                f.debug_struct("Placeholder")
                    .field("ctx", &self.context_id)
                    .finish()
            }
            TraceInner::Abstract(AbstractTrace::Computed {
                invocation,
                position,
            }) => f
                .debug_struct("Computed")
                .field("call", &invocation)
                .field("position", position)
                .field("ctx", &self.context_id)
                .finish(),
            TraceInner::Updatable(UpdatableTrace {
                invocation,
                position,
                value,
            }) => {
                if let Some(value) = value.get() {
                    f.debug_struct("Value").field("value", value).finish()
                } else if let Some(invocation) = invocation.get() {
                    f.debug_struct("Computed")
                        .field("call", &invocation)
                        .field("position", position)
                        .finish()
                } else {
                    f.debug_struct("Invalid").finish()
                }
            }
            TraceInner::Constant(value) => {
                f.debug_struct("Constant").field("value", value).finish()
            }
        }
    }
}

/// Operand to an [`Invocation`]: immutable trace or mutable cell.
#[derive(Clone, Debug)]
pub enum TraceOperand {
    Ref(TraceRef),
    Cell(TraceCellRef),
}

impl TraceOperand {
    pub fn from_ref(t: impl AsRef<TraceRef>) -> Self {
        TraceOperand::Ref(t.as_ref().clone())
    }

    pub fn from_cell(c: TraceCellRef) -> Self {
        TraceOperand::Cell(c)
    }

    /// Effective trace value for dependency walking and context checks.
    pub fn resolve(&self) -> TraceRef {
        match self {
            TraceOperand::Ref(r) => r.clone(),
            TraceOperand::Cell(c) => c.get(),
        }
    }
}

pub struct Invocation {
    op: Op,
    context_id: ContextID,
    closure: Vec<TraceOperand>,
    inputs: Vec<TraceOperand>,
    outputs: usize,
}

impl Debug for Invocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Invocation")
            .field("op", &self.op)
            .field("ctx", &self.context_id)
            .field("inputs", &self.inputs)
            .finish()
    }
}

impl Invocation {
    pub fn op(&self) -> &Op {
        &self.op
    }

    pub fn context_id(&self) -> ContextID {
        self.context_id
    }

    pub fn closure(&self) -> &[TraceOperand] {
        &self.closure
    }

    pub fn inputs(&self) -> &[TraceOperand] {
        &self.inputs
    }

    pub fn outputs(&self) -> usize {
        self.outputs
    }

    pub fn invoke(
        op: Op,
        closure: Vec<TraceOperand>,
        inputs: Vec<TraceOperand>,
        outputs: Vec<ValueInfo>,
    ) -> Vec<Tracer<Generic>> {
        if operands_abstract(&closure, &inputs) {
            Self::invoke_abstract(op, closure, inputs, outputs)
        } else {
            Self::invoke_updatable(op, closure, inputs, outputs)
        }
    }

    pub fn invoke_abstract(
        op: Op,
        closure: Vec<TraceOperand>,
        inputs: Vec<TraceOperand>,
        outputs: Vec<ValueInfo>,
    ) -> Vec<Tracer<Generic>> {
        let context_id = TraceContext::current_id();
        let invocation = Arc::new(Invocation {
            op,
            context_id,
            closure: closure.clone(),
            inputs: inputs.clone(),
            outputs: outputs.len(),
        });
        bind_cell_operands(&invocation, &closure, &inputs, context_id, true);
        outputs
            .into_iter()
            .enumerate()
            .map(|(index, info)| {
                Tracer::new(produced_trace(
                    invocation.clone(),
                    Position::Return(index),
                    info,
                    context_id,
                    true,
                ))
            })
            .collect()
    }

    pub fn invoke_updatable(
        op: Op,
        closure: Vec<TraceOperand>,
        inputs: Vec<TraceOperand>,
        outputs: Vec<ValueInfo>,
    ) -> Vec<Tracer<Generic>> {
        let context_id = TraceContext::current_id();
        let invocation = Arc::new(Invocation {
            op,
            context_id,
            closure: closure.clone(),
            inputs: inputs.clone(),
            outputs: outputs.len(),
        });
        bind_cell_operands(&invocation, &closure, &inputs, context_id, false);
        outputs
            .into_iter()
            .enumerate()
            .map(|(index, info)| {
                Tracer::new(produced_trace(
                    invocation.clone(),
                    Position::Return(index),
                    info,
                    context_id,
                    false,
                ))
            })
            .collect()
    }
}

fn operands_abstract(closure: &[TraceOperand], inputs: &[TraceOperand]) -> bool {
    closure.iter().chain(inputs.iter()).any(|op| match op {
        TraceOperand::Ref(r) => r.is_abstract(),
        TraceOperand::Cell(c) => c.get().is_abstract(),
    })
}

fn produced_trace(
    invocation: Arc<Invocation>,
    position: Position,
    info: ValueInfo,
    context_id: ContextID,
    abstract_: bool,
) -> TraceRef {
    if abstract_ {
        TraceRef::new(Arc::new(Trace {
            inner: TraceInner::Abstract(AbstractTrace::Computed {
                invocation,
                position,
            }),
            info,
            context_id,
        }))
    } else {
        TraceRef::new(Arc::new(Trace {
            inner: TraceInner::Updatable(UpdatableTrace {
                invocation: invocation.into(),
                position,
                value: OnceLock::new(),
            }),
            info,
            context_id,
        }))
    }
}

fn bind_cell_operands(
    invocation: &Arc<Invocation>,
    closure: &[TraceOperand],
    inputs: &[TraceOperand],
    context_id: ContextID,
    abstract_: bool,
) {
    for (i, op) in closure.iter().enumerate() {
        if let TraceOperand::Cell(cell) = op {
            let info = cell.get().info().clone();
            let trace = produced_trace(
                invocation.clone(),
                Position::ClosureUpdate(i),
                info,
                context_id,
                abstract_,
            );
            cell.set(trace);
        }
    }
    for (i, op) in inputs.iter().enumerate() {
        if let TraceOperand::Cell(cell) = op {
            let info = cell.get().info().clone();
            let trace = produced_trace(
                invocation.clone(),
                Position::InputUpdate(i),
                info,
                context_id,
                abstract_,
            );
            cell.set(trace);
        }
    }
}

impl Trace {
    pub fn placeholder(info: ValueInfo, context_id: ContextID) -> Self {
        Trace {
            inner: TraceInner::Abstract(AbstractTrace::Placeholder),
            info,
            context_id,
        }
    }

    pub fn concrete(value: Value, info: ValueInfo, context_id: ContextID) -> Self {
        Trace {
            inner: TraceInner::Constant(value),
            info,
            context_id,
        }
    }

    pub fn context_id(&self) -> ContextID {
        self.context_id
    }

    fn is_abstract(&self) -> bool {
        matches!(self.inner, TraceInner::Abstract(_))
    }

    pub fn producer(&self) -> Option<(Arc<Invocation>, Position)> {
        match &self.inner {
            TraceInner::Abstract(AbstractTrace::Computed {
                invocation,
                position,
            }) => Some((invocation.clone(), *position)),
            TraceInner::Updatable(u) => u
                .invocation
                .get()
                .map(|inv| (inv, u.position)),
            TraceInner::Abstract(AbstractTrace::Placeholder) | TraceInner::Constant(_) => None,
        }
    }

    pub fn is_placeholder(&self) -> bool {
        matches!(
            self.inner,
            TraceInner::Abstract(AbstractTrace::Placeholder)
        )
    }

    pub fn info(&self) -> &ValueInfo {
        &self.info
    }

    /// Concrete evaluation only; effectful programs use [`TraceCell::set`].
    pub fn update(&self, value: Value) {
        match &self.inner {
            TraceInner::Updatable(updatable) => {
                updatable
                    .value
                    .set(value)
                    .ok()
                    .expect("Value has already been set");
                updatable.invocation.downgrade();
            }
            TraceInner::Abstract(_) => panic!("Cannot update an abstract Tracer"),
            TraceInner::Constant(_) => panic!("Cannot update a constant Tracer"),
        }
    }

    pub fn try_concrete(&self) -> Option<&Value> {
        match &self.inner {
            TraceInner::Updatable(updatable) => updatable.value.get(),
            TraceInner::Constant(value) => Some(value),
            _ => None,
        }
    }

    pub fn try_constant(&self) -> Option<&Value> {
        match &self.inner {
            TraceInner::Constant(value) => Some(value),
            _ => None,
        }
    }
}

/// Tracing handle referencing a [`Trace`].
pub struct Tracer<T: Traceable> {
    trace: TraceRef,
    _phantom: PhantomData<T>,
}

impl<T: Traceable> Tracer<T> {
    pub(crate) fn new(trace: TraceRef) -> Self {
        Tracer {
            trace,
            _phantom: PhantomData,
        }
    }

    pub fn placeholder(info: T::Info) -> Self {
        let context_id = TraceContext::current_id();
        Tracer::new(TraceRef::new(Arc::new(Trace::placeholder(
            ValueInfo::new(info),
            context_id,
        ))))
    }

    pub fn concrete(value: T::Concrete, info: T::Info) -> Self {
        let context_id = TraceContext::current_id();
        Tracer::new(TraceRef::new(Arc::new(Trace::concrete(
            Value::new(value),
            ValueInfo::new(info),
            context_id,
        ))))
    }

    pub fn trace_ref(&self) -> &TraceRef {
        &self.trace
    }

    pub fn is_abstract(&self) -> bool {
        self.trace.is_abstract()
    }

    pub fn context_id(&self) -> ContextID {
        self.trace.context_id()
    }

    pub fn info(&self) -> &T::Info {
        self.trace.info().downcast_ref().unwrap()
    }

    pub fn generic(self) -> Tracer<Generic> {
        Tracer {
            trace: self.trace,
            _phantom: PhantomData,
        }
    }

    pub fn cast<U: Traceable>(self) -> Tracer<U> {
        Tracer {
            trace: self.trace,
            _phantom: PhantomData,
        }
    }

    pub fn try_concrete(&self) -> Option<&T::Concrete> {
        self.trace.try_concrete().map(|value| {
            value
                .downcast_ref()
                .expect("Tracer contained incorrect type")
        })
    }
}

impl<T: Traceable> AsRef<TraceRef> for Tracer<T> {
    fn as_ref(&self) -> &TraceRef {
        &self.trace
    }
}

impl<T: Traceable> Clone for Tracer<T> {
    fn clone(&self) -> Self {
        Tracer {
            trace: self.trace.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<T: Traceable> Debug for Tracer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tracer").field("trace", &self.trace).finish()
    }
}

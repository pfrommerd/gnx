mod cell;
mod context;
mod capture;

pub use cell::{CellKey, TraceCell, TraceCellRef, TracerCell};
pub use context::{CellUpdate, ContextID, TraceContext, TraceContextGuard};
pub use capture::{Capture, TracerKey};

use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::{Arc, OnceLock};

use crate::Op;
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

impl Debug for TraceRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.0.inner {
            TraceInner::Abstract(AbstractTrace::Placeholder) => {
                f.debug_struct("Placeholder")
                    .field("ctx", &self.0.context_id)
                    .finish()
            }
            TraceInner::Abstract(AbstractTrace::Computed {
                invocation,
                position,
            }) => f
                .debug_struct("Computed")
                .field("call", &invocation)
                .field("position", position)
                .field("ctx", &self.0.context_id)
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

impl TraceRef {
    pub fn placeholder(info: ValueInfo, context_id: ContextID) -> Self {
        TraceRef(Arc::new(Trace {
            inner: TraceInner::Abstract(AbstractTrace::Placeholder),
            info,
            context_id,
        }))
    }

    pub fn concrete(value: Value, info: ValueInfo, context_id: ContextID) -> Self {
        TraceRef(Arc::new(Trace {
            inner: TraceInner::Constant(value),
            info,
            context_id,
        }))
    }

    pub fn produced(abstract_: bool, invocation: Arc<Invocation>, position: Position,
                    info: ValueInfo, context_id: ContextID) -> Self {
        if abstract_ {
            TraceRef(Arc::new(Trace {
                inner: TraceInner::Abstract(AbstractTrace::Computed {
                    invocation,
                    position,
                }),
                info,
                context_id,
            }))
        } else {
            TraceRef(Arc::new(Trace {
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

    pub fn context_id(&self) -> ContextID { self.0.context_id }

    fn is_abstract(&self) -> bool {
        matches!(self.0.inner, TraceInner::Abstract(_))
    }

    pub fn producer(&self) -> Option<(Arc<Invocation>, Position)> {
        match &self.0.inner {
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
            self.0.inner,
            TraceInner::Abstract(AbstractTrace::Placeholder)
        )
    }

    pub fn info(&self) -> &ValueInfo {
        &self.0.info
    }

    /// Concrete evaluation only; effectful programs use [`TraceCell::set`].
    pub fn update(&self, value: Value) {
        match &self.0.inner {
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
        match &self.0.inner {
            TraceInner::Updatable(updatable) => updatable.value.get(),
            TraceInner::Constant(value) => Some(value),
            _ => None,
        }
    }

    pub fn try_constant(&self) -> Option<&Value> {
        match &self.0.inner {
            TraceInner::Constant(value) => Some(value),
            _ => None,
        }
    }
}

/// A traced value referenced by an [`Invocation`]: immutable trace or mutable cell.
#[derive(Clone, Debug)]
pub enum TraceObject {
    Ref(TraceRef),
    Cell(TraceCellRef),
}

/// Closure or input lists for [`Invocation::invoke`] and [`crate::Capture::from_context`].
pub type TraceOperands = Vec<TraceObject>;

impl TraceObject {
    /// Effective trace value for dependency walking and context checks.
    pub fn resolve(&self) -> TraceRef {
        match self {
            TraceObject::Ref(r) => r.clone(),
            TraceObject::Cell(c) => c.get(),
        }
    }
}

impl<T: Traceable> From<Tracer<T>> for TraceObject {
    fn from(t: Tracer<T>) -> Self {
        TraceObject::Ref(t.into_trace_ref())
    }
}

impl<T: Traceable> From<TracerCell<T>> for TraceObject {
    fn from(c: TracerCell<T>) -> Self {
        TraceObject::Cell(c.into())
    }
}

impl From<TraceRef> for TraceObject {
    fn from(r: TraceRef) -> Self {
        TraceObject::Ref(r)
    }
}

impl From<TraceCellRef> for TraceObject {
    fn from(c: TraceCellRef) -> Self {
        TraceObject::Cell(c)
    }
}

pub struct Invocation {
    op: Op,
    context_id: ContextID,
    closure: Vec<TraceObject>,
    inputs: Vec<TraceObject>,
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

    pub fn closure(&self) -> &[TraceObject] {
        &self.closure
    }

    pub fn inputs(&self) -> &[TraceObject] {
        &self.inputs
    }

    pub fn outputs(&self) -> usize {
        self.outputs
    }

    pub fn invoke(
        op: Op,
        closure: Vec<TraceObject>,
        inputs: Vec<TraceObject>,
        outputs: Vec<ValueInfo>,
    ) -> Vec<Tracer<Generic>> {
        let abstract_ = closure.iter().chain(inputs.iter()).any(|op| match op {
            TraceObject::Ref(r) => r.is_abstract(),
            TraceObject::Cell(c) => c.get().is_abstract(),
        });
        let context_id = TraceContext::current_id();
        let invocation = Arc::new(Invocation {
            op,
            context_id,
            closure: closure.clone(),
            inputs: inputs.clone(),
            outputs: outputs.len(),
        });
        // For any cells in closure, inputs set their new values to the output of the invocation.
        for (i, op) in closure.iter().enumerate() {
            if let TraceObject::Cell(cell) = op {
                let info = cell.get().info().clone();
                let trace = TraceRef::produced(
                    abstract_,
                    invocation.clone(),
                    Position::ClosureUpdate(i),
                    info,
                    context_id,
                );
                cell.set(trace);
            }
        }
        for (i, op) in inputs.iter().enumerate() {
            if let TraceObject::Cell(cell) = op {
                let info = cell.get().info().clone();
                let trace = TraceRef::produced(
                    abstract_,
                    invocation.clone(),
                    Position::InputUpdate(i),
                    info,
                    context_id,
                );
                cell.set(trace);
            }
        }
        // Return the outputs as tracers.
        outputs
            .into_iter()
            .enumerate()
            .map(|(index, info)| {
                Tracer::from(TraceRef::produced(
                    abstract_,
                    invocation.clone(),
                    Position::Return(index),
                    info,
                    context_id,
                ))
            }).collect()
    }
}

/// Tracing handle referencing a [`Trace`].
#[repr(transparent)]
pub struct Tracer<T: Traceable> {
    trace: TraceRef,
    _phantom: PhantomData<T>,
}

impl From<&TraceRef> for &Tracer<Generic> {
    fn from(r: &TraceRef) -> Self {
        // SAFETY: The Tracer<Generic> and TraceRef have the same layout.
        unsafe { std::mem::transmute(r) }
    }
}
impl From<TraceRef> for Tracer<Generic> {
    fn from(t: TraceRef) -> Self {
        Tracer { trace: t, _phantom: PhantomData }
    }
}

impl<T: Traceable> Tracer<T> {
    pub fn unchecked_new(trace: TraceRef) -> Self {
        Tracer {
            trace,
            _phantom: PhantomData,
        }
    }



    pub fn placeholder(info: T::Info) -> Self {
        let context_id = TraceContext::current_id();
        Tracer::unchecked_new(TraceRef::placeholder(
            ValueInfo::new(info),
            context_id,
        ))
    }

    pub fn concrete(value: T::Concrete, info: T::Info) -> Self {
        let context_id = TraceContext::current_id();
        Tracer::unchecked_new(TraceRef::concrete(
            Value::new(value),
            ValueInfo::new(info),
            context_id,
        ))
    }

    pub fn info(&self) -> &T::Info {
        self.trace.info().downcast().unwrap()
    }

    pub fn trace_ref(&self) -> &TraceRef { &self.trace }
    pub fn into_trace_ref(self) -> TraceRef { self.trace }
    pub fn is_abstract(&self) -> bool { self.trace.is_abstract() }
    pub fn context_id(&self) -> ContextID { self.trace.context_id() }

    pub unsafe fn cast_unchecked<U: Traceable>(self) -> Tracer<U> {
        Tracer::unchecked_new(self.trace)
    }

    pub unsafe fn cast_ref_unchecked<U: Traceable>(&self) -> &Tracer<U> {
        // SAFETY: The Tracer<U> and Tracer<T> have the same layout.
        unsafe { std::mem::transmute(self) }
    }

    pub fn into_generic(self) -> Tracer<Generic> {
        // SAFETY: It is safe to cast to Generic.
        unsafe { self.cast_unchecked() }
    }
    pub fn generic(&self) -> &Tracer<Generic> {
        // SAFETY: It is safe to cast to Generic.
        unsafe { self.cast_ref_unchecked() }
    }

    pub fn try_cast_ref<U: Traceable>(&self) -> Result<&Tracer<U>, &Self> {
        match self.trace.info().downcast::<U::Info>() {
            // SAFETY: It is safe to cast to U since we just checked that the info matches.
            Ok(_) => Ok(unsafe { self.cast_ref_unchecked() }),
            Err(_) => Err(self),
        }
    }

    pub fn try_cast<U: Traceable>(self) -> Result<Tracer<U>, Self> {
        match self.trace.info().downcast::<U::Info>() {
            Ok(_) => Ok(Tracer::unchecked_new(self.trace)),
            Err(_) => Err(self),
        }
    }

    pub fn try_concrete(&self) -> Option<&T::Concrete> {
        self.trace.try_concrete().map(|value| {
            value
                .downcast()
                .expect("Tracer contained incorrect type")
        })
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

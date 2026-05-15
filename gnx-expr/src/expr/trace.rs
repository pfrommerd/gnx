use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::{Arc, OnceLock, Weak};

use crate::expr::{Op, Effect};
use crate::util::DgArc;

pub use super::value::{
    ConcreteValue, Generic, Traceable, Value, ValueInfo,
};

// An abstract trace value cannot be updated with a concrete value.
// These do *not* have an ArcCell and so do not need to
// close the underlying arc in order to access the invocation.
enum AbstractTrace {
    Placeholder,
    Returned {
        invocation: Arc<Invocation>,
        ret_idx: usize,
    },
}

struct UpdatableTrace {
    // Will be set to None if the trace is a concrete value.
    invocation: DgArc<Invocation>,
    ret_idx: usize,
    // The concrete value for this tracer.
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
}

pub type TraceRef = Arc<Trace>;
pub type WeakTraceRef = Weak<Trace>;

// For debugging purposes, make this a bit cleaner
// than the auto-derived Debug impl would be
impl Debug for Trace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.inner {
            TraceInner::Abstract(AbstractTrace::Placeholder) => {
                f.debug_struct("Placeholder").finish()
            }
            TraceInner::Abstract(AbstractTrace::Returned {
                invocation,
                ret_idx,
            }) => f
                .debug_struct("Returned")
                .field("call", &invocation)
                .field("ret_idx", ret_idx)
                .finish(),
            TraceInner::Updatable(UpdatableTrace {
                invocation,
                ret_idx,
                value,
            }) => {
                if let Some(value) = value.get() {
                    f.debug_struct("Value").field("value", value).finish()
                } else if let Some(invocation) = invocation.get() {
                    f.debug_struct("Returned")
                        .field("call", &invocation)
                        .field("ret_idx", ret_idx)
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


pub struct Invocation {
    op: Op,
    closure: Vec<TraceRef>,
    inputs: Vec<TraceRef>,
    outputs: usize,
}

impl Debug for Invocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Invocation")
            .field("op", &self.op)
            .field("inputs", &self.inputs)
            .finish()
    }
}

impl Invocation {
    pub fn op(&self) -> &Op {
        &self.op
    }
    pub fn closure(&self) -> &[TraceRef] { &self.closure }
    pub fn inputs(&self) -> &[TraceRef] { &self.inputs }
    pub fn outputs(&self) -> usize { self.outputs }

    #[rustfmt::skip]
    pub fn invoke(
        op: Op,
        closure: Vec<impl AsRef<TraceRef>>,
        inputs: Vec<impl AsRef<TraceRef>>,
        outputs: Vec<ValueInfo>,
    ) -> Vec<Tracer<Generic>> {
        let closure_refs: Vec<TraceRef> = closure.into_iter().map(|x| x.as_ref().clone()).collect();
        let input_refs: Vec<TraceRef> = inputs.into_iter().map(|x| x.as_ref().clone()).collect();
        let is_abstract = input_refs.iter().any(|x| x.is_abstract()) || closure_refs.iter().any(|x| x.is_abstract());
        let invocation = Arc::new(Invocation { op, closure: closure_refs, inputs: input_refs, outputs: outputs.len() });
        let outputs: Vec<TraceRef> = if is_abstract {
            outputs.into_iter().enumerate().map(|(index, info)| {
                Arc::new(Trace {
                    inner: TraceInner::Abstract(AbstractTrace::Returned {
                        invocation: invocation.clone(),
                        ret_idx: index,
                    }),
                    info,
                })
            }).collect()
        } else {
            outputs.into_iter().enumerate().map(|(index, info)| {
                Arc::new(Trace {
                    inner: TraceInner::Updatable(UpdatableTrace {
                        invocation: invocation.clone().into(),
                        ret_idx: index,
                        value: OnceLock::new(),
                    }),
                    info,
                })
            }).collect()
        };
        // Each output tracer starts with a single-frame stack.
        outputs.into_iter().map(Tracer::new).collect()
    }
}

impl Trace {
    pub fn placeholder(info: ValueInfo) -> Self {
        Trace { inner: TraceInner::Abstract(AbstractTrace::Placeholder), info }
    }
    pub fn concrete(value: Value, info: ValueInfo) -> Self {
        Trace { inner: TraceInner::Constant(value), info }
    }

    fn is_abstract(&self) -> bool {
        matches!(self.inner, TraceInner::Abstract(_))
    }

    /// If this trace is the result of a primitive invocation, returns that invocation and result index.
    pub fn producer(&self) -> Option<(Arc<Invocation>, usize)> {
        match &self.inner {
            TraceInner::Abstract(AbstractTrace::Returned {
                invocation,
                ret_idx,
            }) => Some((invocation.clone(), *ret_idx)),
            TraceInner::Updatable(u) => u.invocation.get().map(|inv| (inv, u.ret_idx)),
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

    pub fn update(&self, value: Value) {
        match &self.inner {
            TraceInner::Updatable(updatable) => {
                updatable
                    .value
                    .set(value)
                    .ok()
                    .expect("Value has already been set");
                // Downgrade the invocation to allow
                // for dropping the associated graph.
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
    fn new(trace: TraceRef) -> Self {
        Tracer {
            trace,
            _phantom: PhantomData,
        }
    }

    pub fn placeholder(info: T::Info) -> Self {
        Tracer::new(Arc::new(Trace::placeholder(ValueInfo::new(info))))
    }

    // Create a tracer from a concrete value.
    pub fn concrete(value: T::Concrete, info: T::Info) -> Self {
        Tracer::new(Arc::new(Trace::concrete(
            Value::new(value),
            ValueInfo::new(info),
        )))
    }

    pub fn trace_ref(&self) -> &TraceRef {
        &self.trace
    }

    pub fn is_abstract(&self) -> bool {
        self.trace.is_abstract()
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

impl<T: Traceable> AsRef<TraceRef> for &Tracer<T> {
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
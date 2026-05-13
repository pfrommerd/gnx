use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::{Arc, OnceLock, RwLock, Weak};

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

#[derive(Debug)]
pub struct InputTracer {
    pub trace: TraceRef,
    pub effect: Effect,
}


pub struct Invocation {
    op: Op,
    closure: Vec<InputTracer>,
    inputs: Vec<InputTracer>,
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
    pub fn closure(&self) -> &[InputTracer] { &self.closure }
    pub fn inputs(&self) -> &[InputTracer] { &self.inputs }
    pub fn outputs(&self) -> usize { self.outputs }

    #[rustfmt::skip]
    pub fn invoke(
        op: Op,
        closure: Vec<(Tracer<Generic>, Effect)>,
        inputs: Vec<(Tracer<Generic>, Effect)>,
        outputs: Vec<ValueInfo>,
    ) -> Vec<Tracer<Generic>> {
        let is_abstract = inputs.iter().any(|(x, _)| x.is_abstract()) || closure.iter().any(|(x, _)| x.is_abstract());
        let closure_refs: Vec<InputTracer> = closure.into_iter().map(|(x, e)| InputTracer {
            trace: x.trace_ref(), effect: e
        }).collect();
        let input_refs: Vec<InputTracer> = inputs.into_iter().map(|(x, e)| InputTracer {
            trace: x.trace_ref(), effect: e
        }).collect();
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
        outputs.into_iter().map(|o| Tracer::new_single(o)).collect()
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

/// Tracing handle with a **non-empty stack** of [`TraceRef`] overlays.
/// The **top** of the stack is the active trace for [`Self::invoke`].
///
/// Use [`Self::push`] / [`Self::pop`] when entering or leaving a nested tracing scope.
/// Cloning ([`Clone::clone`]) copies **only the top** frame; deeper frames are not shared
/// with the clone.
pub struct Tracer<T: Traceable> {
    /// Oldest frame first; the current trace is always [`Self::top`].
    stack: RwLock<Vec<TraceRef>>,
    base: TraceRef, // The base tracer (returned if stack is empty)
    _phantom: PhantomData<T>,
}

impl<T: Traceable> Tracer<T> {
    fn new_single(trace: TraceRef) -> Self {
        Tracer {
            stack: RwLock::new(vec![]),
            base: trace,
            _phantom: PhantomData,
        }
    }

    pub fn placeholder(info: T::Info) -> Self {
        Tracer::new_single(Arc::new(Trace::placeholder(ValueInfo::new(info))))
    }

    // Create a tracer from a concrete value.
    pub fn concrete(value: T::Concrete, info: T::Info) -> Self {
        Tracer::new_single(Arc::new(Trace::concrete(
            Value::new(value),
            ValueInfo::new(info),
        )))
    }


    /// Active [`TraceRef`] (top of the overlay stack).
    pub fn trace_ref(&self) -> TraceRef {
        let guard = self.stack.read().unwrap();
        if let Some(frame) = guard.last() {
            frame.clone()
        } else {
            self.base.clone()
        }
    }

    /// Number of overlayed traces (always >= 1).
    pub fn stack_depth(&self) -> usize {
        self.stack.read().unwrap().len() + 1
    }

    /// Push a new trace frame on top (e.g. when entering a nested traced function).
    pub fn push(&mut self, frame: TraceRef) {
        assert!(frame.is_abstract(), "Expected abstract trace, got concrete");
        assert!(frame.info() == self.base.info(), "Expected trace with info {:?}, got {:?}", self.base.info(), frame.info());
        let mut guard = self.stack.write().unwrap();
        guard.push(frame);
    }

    /// Pop the top frame. Returns `None` if only one frame remains.
    pub fn pop(&self) {
        let mut guard = self.stack.write().unwrap();
        guard.pop();
    }

    pub fn is_abstract(&self) -> bool {
        let guard = self.stack.read().unwrap();
        !guard.is_empty() || self.base.is_abstract()
    }

    pub fn info(&self) -> &T::Info {
        self.base.info().downcast_ref().unwrap()
    }

    // Cast to a generic tracer.
    pub fn generic(self) -> Tracer<Generic> {
        Tracer {
            stack: self.stack,
            base: self.base,
            _phantom: PhantomData,
        }
    }
    // Cast to a particular type of tracer.
    pub fn cast<U: Traceable>(self) -> Tracer<U> {
        Tracer {
            stack: self.stack,
            base: self.base,
            _phantom: PhantomData,
        }
    }

    // Only the *base* value can be updatable. All higher frames must be abstract.
    pub fn try_concrete(&self) -> Option<&T::Concrete> {
        if !self.stack.read().unwrap().is_empty() {
            return None;
        }
        self.base.try_concrete().map(|value| {
            value.downcast_ref()
                .expect("Tracer contained incorrect type")
        })
    }
}

impl<T: Traceable> Clone for Tracer<T> {
    fn clone(&self) -> Self {
        Tracer {
            stack: RwLock::new(vec![]),
            base: self.trace_ref(),
            _phantom: PhantomData,
        }
    }
}

impl<T: Traceable> Debug for Tracer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tracer")
            .field("stack_depth", &self.stack_depth())
            .field("current", &self.trace_ref())
            .finish()
    }
}
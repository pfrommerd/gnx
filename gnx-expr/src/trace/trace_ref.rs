use crate::{Generic, Invocation, Traceable, Value, ValueInfo};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::Arc;

use super::context::{ContextID, TraceContext};

enum TraceVariant {
    // a concrete value
    Concrete(Value),
    // an abstract placeholder, used for inputs
    Placeholder,
    // represents the returned value of an invocation
    Returned {
        inv: Arc<Invocation>,
        ret_idx: usize,
    },
}

pub struct TraceData {
    variant: TraceVariant,
    info: ValueInfo,
    context_id: ContextID,
}

/// Shared handle to an immutable [`TraceData`].
#[repr(transparent)]
#[derive(Clone)]
pub struct Trace(pub(crate) Arc<TraceData>);

impl Debug for Trace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.0.variant {
            TraceVariant::Placeholder => f
                .debug_struct("Placeholder")
                .field("ctx", &self.0.context_id)
                .finish(),
            TraceVariant::Returned { inv, ret_idx } => f
                .debug_struct("Returned")
                .field("call", &inv)
                .field("ret_idx", &ret_idx)
                .field("ctx", &self.0.context_id)
                .finish(),
            TraceVariant::Concrete(value) => f
                .debug_struct("Concrete")
                .field("value", value)
                .field("ctx", &self.0.context_id)
                .finish(),
        }
    }
}

impl Trace {
    pub fn concrete(value: Value, info: ValueInfo) -> Self {
        Trace(Arc::new(TraceData {
            variant: TraceVariant::Concrete(value),
            info,
            context_id: TraceContext::current_id(),
        }))
    }

    pub fn placeholder(info: ValueInfo) -> Self {
        Trace(Arc::new(TraceData {
            variant: TraceVariant::Placeholder,
            info,
            context_id: TraceContext::current_id(),
        }))
    }

    pub fn returned(invocation: Arc<Invocation>, ret_idx: usize, info: ValueInfo) -> Self {
        Trace(Arc::new(TraceData {
            variant: TraceVariant::Returned {
                inv: invocation,
                ret_idx,
            },
            info,
            context_id: TraceContext::current_id(),
        }))
    }

    pub fn context_id(&self) -> &ContextID {
        &self.0.context_id
    }

    // Returns the producer of this trace, if any.
    pub fn producer(&self) -> Option<&Arc<Invocation>> {
        match &self.0.variant {
            TraceVariant::Placeholder => None,
            TraceVariant::Returned { inv, .. } => Some(inv),
            TraceVariant::Concrete(_) => None,
        }
    }

    /// The producing DAG node and its immutable result position.
    pub fn producer_with_index(&self) -> Option<(&Arc<Invocation>, usize)> {
        match &self.0.variant {
            TraceVariant::Placeholder | TraceVariant::Concrete(_) => None,
            TraceVariant::Returned { inv, ret_idx } => Some((inv, *ret_idx)),
        }
    }

    pub fn is_placeholder(&self) -> bool {
        matches!(self.0.variant, TraceVariant::Placeholder)
    }

    pub fn info(&self) -> &ValueInfo {
        &self.0.info
    }

    pub fn try_concrete(&self) -> Option<&Value> {
        match &self.0.variant {
            TraceVariant::Concrete(value) => Some(value),
            _ => None,
        }
    }
}

/// Tracing handle referencing a [`TraceData`].
#[repr(transparent)]
pub struct Tracer<T: Traceable> {
    trace: Trace,
    _phantom: PhantomData<T>,
}

impl From<&Trace> for &Tracer<Generic> {
    fn from(r: &Trace) -> Self {
        // SAFETY: The Tracer<Generic> and Trace have the same layout.
        unsafe { std::mem::transmute(r) }
    }
}
impl From<Trace> for Tracer<Generic> {
    fn from(t: Trace) -> Self {
        Tracer {
            trace: t,
            _phantom: PhantomData,
        }
    }
}

impl<T: Traceable> Tracer<T> {
    pub fn placeholder(info: ValueInfo) -> Self {
        unsafe { Tracer::unchecked_new(Trace::placeholder(info)) }
    }

    pub unsafe fn unchecked_new(trace: Trace) -> Self {
        Tracer {
            trace,
            _phantom: PhantomData,
        }
    }

    pub fn info(&self) -> &T::Info {
        self.trace.info().downcast().unwrap()
    }

    pub fn trace_ref(&self) -> &Trace {
        &self.trace
    }
    pub fn into_trace_ref(self) -> Trace {
        self.trace
    }
    pub fn context_id(&self) -> &ContextID {
        self.trace.context_id()
    }

    pub unsafe fn cast_unchecked<U: Traceable>(self) -> Tracer<U> {
        unsafe { Tracer::unchecked_new(self.trace) }
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
            // SAFETY: We have checked that the info matches.
            Ok(_) => Ok(unsafe { Tracer::unchecked_new(self.trace) }),
            Err(_) => Err(self),
        }
    }

    pub fn try_concrete(&self) -> Option<&T::Concrete> {
        self.trace
            .try_concrete()
            .map(|value| value.downcast().expect("Tracer contained incorrect type"))
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
        f.debug_struct("Tracer")
            .field("trace", &self.trace)
            .finish()
    }
}

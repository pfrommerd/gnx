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
    // The state of a new cell created
    // by the specified invocation
    NewCellState {
        inv: Arc<Invocation>,
        ret_idx: usize,
    },
    // the cell has been updated by the specified invocation
    UpdatedCellState {
        inv: Arc<Invocation>,
    },
}

pub struct Trace {
    variant: TraceVariant,
    info: ValueInfo,
    context_id: ContextID,
}

/// Shared handle to an immutable [`Trace`].
#[repr(transparent)]
#[derive(Clone)]
pub struct TraceRef(pub(crate) Arc<Trace>);

impl Debug for TraceRef {
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
            TraceVariant::NewCellState { inv, ret_idx } => f
                .debug_struct("NewCellState")
                .field("call", &inv)
                .field("ret_idx", &ret_idx)
                .field("ctx", &self.0.context_id)
                .finish(),
            TraceVariant::UpdatedCellState {
                inv,
                subexpr,
                index,
            } => f
                .debug_struct("UpdatedCellState")
                .field("call", &inv)
                .field("subexpr", &subexpr)
                .field("index", &index)
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

impl TraceRef {
    pub fn concrete(value: Value, info: ValueInfo) -> Self {
        TraceRef(Arc::new(Trace {
            variant: TraceVariant::Concrete(value),
            info,
            context_id: TraceContext::current_id(),
        }))
    }

    pub fn placeholder(info: ValueInfo) -> Self {
        TraceRef(Arc::new(Trace {
            variant: TraceVariant::Placeholder,
            info,
            context_id: TraceContext::current_id(),
        }))
    }

    pub fn returned(invocation: Arc<Invocation>, ret_idx: usize, info: ValueInfo) -> Self {
        TraceRef(Arc::new(Trace {
            variant: TraceVariant::Returned {
                inv: invocation,
                ret_idx,
            },
            info,
            context_id: TraceContext::current_id(),
        }))
    }

    pub fn updated_cell(
        invocation: Arc<Invocation>,
        info: ValueInfo,
        subexpr: Option<usize>,
        index: usize,
    ) -> Self {
        TraceRef(Arc::new(Trace {
            variant: TraceVariant::UpdatedCellState {
                inv: invocation,
                subexpr,
                index,
            },
            info,
            context_id: TraceContext::current_id(),
        }))
    }

    pub fn context_id(&self) -> &ContextID {
        &self.0.context_id
    }

    // Returns the produce of this trace, if any.
    pub fn producer(&self) -> Option<&Arc<Invocation>> {
        match &self.0.variant {
            TraceVariant::Placeholder => None,
            TraceVariant::Returned { inv, .. } => Some(inv),
            TraceVariant::NewCellState { inv, .. } => Some(inv),
            TraceVariant::UpdatedCellState { inv, .. } => Some(inv),
            TraceVariant::Concrete(_) => None,
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
        Tracer {
            trace: t,
            _phantom: PhantomData,
        }
    }
}

impl<T: Traceable> Tracer<T> {
    pub fn placeholder(info: ValueInfo) -> Self {
        unsafe { Tracer::unchecked_new(TraceRef::placeholder(info)) }
    }

    pub unsafe fn unchecked_new(trace: TraceRef) -> Self {
        Tracer {
            trace,
            _phantom: PhantomData,
        }
    }

    pub fn info(&self) -> &T::Info {
        self.trace.info().downcast().unwrap()
    }

    pub fn trace_ref(&self) -> &TraceRef {
        &self.trace
    }
    pub fn into_trace_ref(self) -> TraceRef {
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

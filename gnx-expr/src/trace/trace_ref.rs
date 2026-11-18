use crate::{Generic, Invocation, Traceable, Value, ValueInfo};
use std::collections::BTreeMap;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use super::context::{ContextID, TraceContext};

pub struct TraceData {
    info: ValueInfo,
    context_id: ContextID,
    placeholder: bool,
    concrete: OnceLock<Value>,
    producer: Mutex<Option<Arc<Invocation>>>,
    // Keep the capture key stable even after detaching the producer.
    origin: Option<(Weak<Invocation>, usize)>,
    external_handles: AtomicUsize,
    consumers: Mutex<BTreeMap<usize, Weak<Invocation>>>,
}

/// Shared handle to a traced value. Materialization does not change its
/// identity, metadata, or immutable contents.
#[repr(C)]
pub struct Trace {
    pub(super) data: Arc<TraceData>,
    external: bool,
}

impl Clone for Trace {
    fn clone(&self) -> Self {
        Self::external(self.data.clone())
    }
}

impl Drop for Trace {
    fn drop(&mut self) {
        self.make_internal();
    }
}

impl Debug for Trace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(value) = self.try_concrete() {
            return f
                .debug_struct("Concrete")
                .field("value", value)
                .field("ctx", self.context_id())
                .finish();
        }
        if let Some((invocation, index)) = self.producer_with_index() {
            return f
                .debug_struct("Returned")
                .field("call", &invocation)
                .field("ret_idx", &index)
                .finish();
        }
        f.debug_struct("Placeholder")
            .field("ctx", self.context_id())
            .finish()
    }
}

impl Trace {
    fn external(data: Arc<TraceData>) -> Self {
        data.external_handles.fetch_add(1, Ordering::Relaxed);
        Self {
            data,
            external: true,
        }
    }

    pub(super) fn internal(data: Arc<TraceData>) -> Self {
        Self {
            data,
            external: false,
        }
    }

    pub(super) fn internal_clone(&self) -> Self {
        Self::internal(self.data.clone())
    }

    pub(super) fn make_internal(&mut self) {
        if self.external {
            self.external = false;
            self.data.external_handles.fetch_sub(1, Ordering::Relaxed);
        }
    }

    pub(super) fn add_consumer(&self, invocation: &Arc<Invocation>) {
        // Concrete data already cuts the DAG, so it needs no reverse edges.
        if self.try_concrete().is_some() || self.is_placeholder() {
            return;
        }
        let key = Arc::as_ptr(invocation) as usize;
        let mut consumers = self.data.consumers.lock().unwrap();
        if self.try_concrete().is_some() || consumers.contains_key(&key) {
            return;
        }
        // Periodically prune expired edges without scanning the full fan-out
        // on every insertion. Keeping a Weak also prevents pointer reuse.
        if consumers.len().is_power_of_two() {
            consumers.retain(|_, user| user.strong_count() != 0);
        }
        consumers.insert(key, Arc::downgrade(invocation));
    }

    pub(super) fn is_live_outside(&self, inside: impl Fn(usize) -> bool) -> bool {
        if self.data.external_handles.load(Ordering::Relaxed) != 0 {
            return true;
        }
        let mut consumers = self.data.consumers.lock().unwrap();
        consumers.retain(|_, user| user.strong_count() != 0);
        consumers.keys().any(|&user| !inside(user))
    }

    pub fn concrete(value: Value, info: ValueInfo) -> Self {
        Self::external(Arc::new(TraceData {
            info,
            context_id: TraceContext::current_id(),
            placeholder: false,
            concrete: OnceLock::from(value),
            producer: Mutex::new(None),
            origin: None,
            external_handles: AtomicUsize::new(0),
            consumers: Mutex::default(),
        }))
    }

    pub fn placeholder(info: ValueInfo) -> Self {
        Self::external(Arc::new(TraceData {
            info,
            context_id: TraceContext::current_id(),
            placeholder: true,
            concrete: OnceLock::new(),
            producer: Mutex::new(None),
            origin: None,
            external_handles: AtomicUsize::new(0),
            consumers: Mutex::default(),
        }))
    }

    pub(super) fn returned(invocation: &Arc<Invocation>, ret_idx: usize, info: ValueInfo) -> Self {
        Self::external(Arc::new(TraceData {
            info,
            context_id: invocation.context_id().clone(),
            placeholder: false,
            concrete: OnceLock::new(),
            producer: Mutex::new(Some(invocation.clone())),
            origin: Some((Arc::downgrade(invocation), ret_idx)),
            external_handles: AtomicUsize::new(0),
            consumers: Mutex::default(),
        }))
    }

    pub fn context_id(&self) -> &ContextID {
        &self.data.context_id
    }

    /// Snapshot the producer of this trace, if it is still deferred.
    pub fn producer(&self) -> Option<Arc<Invocation>> {
        self.data.producer.lock().unwrap().clone()
    }

    /// The producing DAG node and its immutable result position.
    pub fn producer_with_index(&self) -> Option<(Arc<Invocation>, usize)> {
        let (_, index) = self.origin()?;
        Some((self.producer()?, index))
    }

    pub(super) fn origin(&self) -> Option<(&Weak<Invocation>, usize)> {
        self.data
            .origin
            .as_ref()
            .map(|(producer, index)| (producer, *index))
    }

    pub fn is_placeholder(&self) -> bool {
        self.data.placeholder
    }

    pub fn info(&self) -> &ValueInfo {
        &self.data.info
    }

    pub fn try_concrete(&self) -> Option<&Value> {
        self.data.concrete.get()
    }

    /// Fill a pending result; the first publication wins. Detach the old
    /// producer so it can be released after leaving the publication lock.
    pub(crate) fn publish(&self, value: Value) -> Option<Arc<Invocation>> {
        let mut producer = self.data.producer.lock().unwrap();
        if producer.is_some() && self.data.concrete.set(value).is_ok() {
            self.data.consumers.lock().unwrap().clear();
            producer.take()
        } else {
            None
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

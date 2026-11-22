use std::borrow::Cow;
use std::convert::Infallible;
use std::sync::Arc;

use gnx_expr::{HExpr, Kind, ObjectTracer, VarInfo};
use gnx_graph::{Graph, GraphEq, GraphHash};

use crate::cache::{TraceCache, TraceEntry, TracerBuilder};
use crate::{Callable, Function, Jitable, TryCallable, TryJitable};

pub use gnx_derive::{jit, jit_fn, jittable, try_jit_fn};

/// Callable wrapper used by the function attribute and closure macro.
///
/// Cache the Rust-body trace by input graph structure, static data and ordered
/// leaf metadata.
/// Each dynamic slot has its own placeholder, independently of handle identity.
/// Each wrapper has one input signature and a typed cache for that signature.
/// Threads can share it when the callable and cached builders are `Send + Sync`.
/// The input and its owned form must use the same builder for tracer leaves;
/// the owned graph provides that builder's hashing and equality.
/// Graph hashing and equality for cache keys must not re-enter the same JIT.
/// Calls record high-level call nodes at any nesting level. No lowering or
/// backend compilation is performed here.
/// Rust side effects and ordinary Rust captures are evaluated on trace misses;
/// use graph inputs for data that should be reconsidered on each call.
/// Capture, call, or graph reconstruction errors currently panic.
///
/// ```
/// #[gnx_transforms::jit]
/// fn increment(value: u32) -> u32 { value + 1 }
/// let offset = 2;
/// let callable = gnx_transforms::jit_fn!(|value: u32| increment(value) + offset);
/// assert_eq!(callable(3), 6);
/// ```
///
/// Use `try_jit_fn!` for closures with a fallible body:
/// ```compile_fail
/// let _ = gnx_transforms::jit_fn!(|x: u32| -> Result<u32, &'static str> { Ok(x) });
/// ```
///
/// The attribute currently accepts no options:
/// ```compile_fail
/// #[gnx_transforms::jit(unsupported)]
/// fn example(value: u32) -> u32 { value }
/// ```
pub struct Jit<I: Graph, F: Function<I>> {
    func: F,
    name_hint: Option<Cow<'static, str>>,
    traces: TraceCache<I, F::Output>,
    try_traces: TraceCache<I, F::Output>,
}

impl<I: Graph, F: Function<I>> Jit<I, F> {
    const fn new(func: F, name_hint: Option<Cow<'static, str>>) -> Self {
        Self {
            func,
            name_hint,
            traces: TraceCache::new(),
            try_traces: TraceCache::new(),
        }
    }

    /// Wrap a function and label its traces for display.
    pub const fn new_with_hint(func: F, name_hint: Cow<'static, str>) -> Self {
        Self::new(func, Some(name_hint))
    }

    pub(crate) fn set_name_hint(&mut self, name_hint: Cow<'static, str>) {
        self.name_hint = Some(name_hint);
    }

    pub fn name_hint(&self) -> Option<&str> {
        self.name_hint.as_deref()
    }
}

impl<I: Graph, F: Function<I>> Function<I> for Jit<I, F> {
    type Output = F::Output;
}

impl<I, F> TryCallable<I> for Jit<I, F>
where
    I: Graph,
    I::Owned: GraphHash + GraphEq<Builder<ObjectTracer> = TracerBuilder<I>>,
    F: TryJitable<I>,
{
    type Error = F::Error;

    fn try_invoke(&self, input: I) -> Result<Self::Output, Self::Error> {
        self.try_invoke_compatible(input)
    }
}

impl<I, F> Callable<I> for Jit<I, F>
where
    I: Graph,
    I::Owned: GraphHash + GraphEq<Builder<ObjectTracer> = TracerBuilder<I>>,
    F: Jitable<I>,
{
    fn invoke(&self, input: I) -> Self::Output {
        self.traces
            .invoke(input, self.name_hint.as_ref(), |input| {
                Ok::<_, Infallible>(self.func.invoke_owned(input))
            })
            .unwrap_or_else(|never| match never {})
    }
}

impl<I, F> Jit<I, F>
where
    I: Graph,
    I::Owned: GraphHash + GraphEq<Builder<ObjectTracer> = TracerBuilder<I>>,
    F: Function<I>,
    F::Output: Graph<Owned = F::Output>,
{
    /// Return the high-level expression for this input signature without
    /// emitting a call. Uses the same trace cache as `TryCallable::try_invoke`.
    pub fn try_trace(&self, input: I) -> Result<Arc<HExpr>, F::Error>
    where
        F: TryJitable<I>,
    {
        self.try_traces
            .trace(input, self.name_hint.as_ref(), |input| {
                self.func.try_invoke_owned(input)
            })
    }

    // An attribute-generated static can accept shorter borrows because
    // tracing only passes the rebuilt, owned graph to its body.
    #[doc(hidden)]
    pub fn try_invoke_compatible<A>(&self, input: A) -> Result<F::Output, F::Error>
    where
        F: TryJitable<I>,
        A: Graph<Owned = I::Owned, Builder<ObjectTracer> = TracerBuilder<I>>,
    {
        self.try_traces
            .invoke(input, self.name_hint.as_ref(), |input| {
                self.func.try_invoke_owned(input)
            })
    }
}

/// Wrap a function with a cache typed to its input graph and owned output.
/// `Callable::invoke` traces a `Jitable` body. `TryCallable::try_invoke`
/// traces a `TryJitable` body. They cache separately because the body can
/// implement each interface differently. Body errors aren't cached.
pub const fn jit<I: Graph, F: Function<I>>(func: F) -> Jit<I, F> {
    Jit::new(func, None)
}

impl<I, F> Jit<I, F>
where
    I: Graph,
    I::Owned: GraphHash + GraphEq<Builder<ObjectTracer> = TracerBuilder<I>>,
    F: Function<I>,
    F::Output: Graph<Owned = F::Output>,
{
    /// Trace (or reuse) the body at an explicit signature: the input
    /// structure and one kind per tracer leaf. Shares `Callable::invoke`'s cache.
    pub(crate) fn traced(
        &self,
        builder: TracerBuilder<I>,
        kinds: Vec<(Kind, VarInfo)>,
    ) -> Arc<TraceEntry<F::Output>>
    where
        F: Jitable<I>,
    {
        self.traces
            .entry(builder, kinds, self.name_hint.as_ref(), |input| {
                Ok::<_, Infallible>(self.func.invoke_owned(input))
            })
            .unwrap_or_else(|never| match never {})
    }

    /// Like [`Jit::traced`] for a fallible body. Shares `try_invoke`'s cache.
    pub(crate) fn try_traced(
        &self,
        builder: TracerBuilder<I>,
        kinds: Vec<(Kind, VarInfo)>,
    ) -> Result<Arc<TraceEntry<F::Output>>, F::Error>
    where
        F: TryJitable<I>,
    {
        self.try_traces
            .entry(builder, kinds, self.name_hint.as_ref(), |input| {
                self.func.try_invoke_owned(input)
            })
    }
}

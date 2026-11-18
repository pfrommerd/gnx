use crate::{Callable, Jitable, TryCallable, TryJitable};

pub use gnx_derive::{jit, jit_fn, jittable};

use gnx_expr::{Capture, CellTrace, ContextID, Kind, ObjectTracer, Trace, TraceContext};
use gnx_graph::util::{GraphExt, IterSource};
use gnx_graph::{
    Builder, Error, Graph, GraphContext, GraphEq, GraphEqContext, GraphHash, GraphHashContext,
    GraphPartialEq, Of, SourceError,
};
use std::any::{Any, TypeId};
use std::collections::{HashMap, hash_map::DefaultHasher};
use std::convert::Infallible;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, TryLockError};

struct TraceEntry<O> {
    capture: Capture,
    output_builder: O,
}

struct TraceSlot<B, O> {
    input_builder: B,
    kinds: Vec<Kind>,
    entry: Mutex<Option<Arc<TraceEntry<O>>>>,
}

impl<B, O> TraceSlot<B, O> {
    fn get_or_trace<E>(
        &self,
        trace: impl FnOnce() -> Result<Arc<TraceEntry<O>>, E>,
    ) -> Result<Arc<TraceEntry<O>>, E> {
        // Nested traces must not wait on another slot: two tracing bodies can
        // call each other. If the slot is busy, trace locally without publishing.
        let mut entry = if TraceContext::current().is_some() {
            match self.entry.try_lock() {
                Ok(entry) => entry,
                Err(TryLockError::Poisoned(err)) => err.into_inner(),
                Err(TryLockError::WouldBlock) => return trace(),
            }
        } else {
            self.entry.lock().unwrap_or_else(|err| err.into_inner())
        };
        if let Some(cached) = entry.as_ref() {
            return Ok(cached.clone());
        }
        let traced = trace()?;
        if reusable_captures(&traced.capture) {
            *entry = Some(traced.clone());
        }
        Ok(traced)
    }
}

type TraceKey = (TypeId, &'static str, TraceMode, u64);
type TraceBuckets = HashMap<TraceKey, Vec<Arc<dyn Any + Send + Sync>>>;

struct TraceCache(Mutex<Option<TraceBuckets>>);

impl TraceCache {
    const fn new() -> Self {
        Self(Mutex::new(None))
    }

    fn slot<S: Any + Send + Sync>(
        &self,
        key: TraceKey,
        new_slot: S,
        matches: impl Fn(&S, &S) -> bool,
    ) -> Arc<S> {
        let mut cache = self.0.lock().unwrap();
        let mut compared = 0;
        loop {
            let bucket = cache
                .get_or_insert_with(HashMap::new)
                .entry(key)
                .or_default();
            if bucket.len() == compared {
                let slot = Arc::new(new_slot);
                bucket.push(slot.clone());
                return slot;
            }

            // Graph equality can invoke user code. Compare only new candidates,
            // outside the map lock, before deciding whether to insert a slot.
            let candidates = bucket[compared..].to_vec();
            compared = bucket.len();
            drop(cache);
            for candidate in candidates {
                let slot = candidate.downcast::<S>().expect("JIT trace slot type");
                if matches(&slot, &new_slot) {
                    return slot;
                }
            }
            cache = self.0.lock().unwrap();
        }
    }
}

// A callable may implement both traits with different bodies for the same input.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum TraceMode {
    Infallible,
    Fallible,
}

fn kind(object: &ObjectTracer) -> Kind {
    match object {
        ObjectTracer::Ref(value) => Kind::Value(value.info().clone()),
        ObjectTracer::Cell(cell) => Kind::Cell(cell.type_info().clone()),
    }
}

fn placeholder(kind: &Kind) -> ObjectTracer {
    match kind {
        Kind::Value(info) => Trace::placeholder(info.clone()).into(),
        Kind::Cell(info) => CellTrace::placeholder(info.clone()).into(),
    }
}

fn reusable_captures(capture: &Capture) -> bool {
    // Scoped captures can depend on an enclosing trace's variables. Never
    // retain them in a cache that can outlive that trace. Concrete captures
    // such as constants created by the Rust body are not scope-dependent.
    capture.closure().iter().all(|object| match object {
        ObjectTracer::Ref(value) => {
            value.try_concrete().is_some() || value.context_id() == &ContextID::ROOT
        }
        ObjectTracer::Cell(cell) => cell.context_id() == &ContextID::ROOT,
    })
}

fn same_builder<I: GraphEq>(
    left: &I::Builder<ObjectTracer>,
    right: &I::Builder<ObjectTracer>,
    leaves: &[ObjectTracer],
) -> bool
where
    I::Owned: GraphEq,
{
    if let Some(equal) = I::builders_eq::<ObjectTracer>(left, right, &mut GraphEqContext::new()) {
        return equal;
    }
    // For custom builders with no direct comparison, fill both skeletons with
    // the same source and compare the results while excluding dynamic handles.
    let filter = Of::<ObjectTracer>::filter();
    let build = |builder: &I::Builder<ObjectTracer>| {
        build_from_leaves(builder.clone(), leaves.iter().cloned())
    };
    match (build(left), build(right)) {
        (Ok(left), Ok(right)) => left.graph_eq_filtered(&right, filter),
        _ => false,
    }
}

fn build_from_leaves<B: Builder<ObjectTracer>>(
    builder: B,
    leaves: impl IntoIterator<Item = ObjectTracer>,
) -> Result<B::Graph, SourceError> {
    let mut leaves = leaves.into_iter();
    let graph = builder.build(IterSource::new(&mut leaves), &mut GraphContext::new())?;
    if leaves.next().is_some() {
        return Err(SourceError::custom(
            "JIT builder did not consume all leaves",
        ));
    }
    Ok(graph)
}

/// Callable wrapper used by the function attribute and closure macro.
///
/// Cache the Rust-body trace by input graph structure, static data and ordered
/// leaf metadata.
/// Each dynamic slot has its own placeholder, independently of handle identity.
/// Threads can share a wrapper when its callable is `Sync`. Cached graph
/// builders must be `Send + Sync` (use `Arc`, not `Rc`, for shared nodes).
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
/// The attribute currently accepts no options:
/// ```compile_fail
/// #[gnx_transforms::jit(unsupported)]
/// fn example(value: u32) -> u32 { value }
/// ```
pub struct Jit<F> {
    func: F,
    traces: TraceCache,
}

impl<I, F> Callable<I> for Jit<F>
where
    I: GraphEq + GraphHash,
    I::Owned: GraphEq,
    F: Jitable<I>,
    I::Builder<ObjectTracer>: Send + Sync,
    F::Output: Graph<Builder<ObjectTracer>: Send + Sync>,
{
    type Output = <F as Callable<I>>::Output;

    fn invoke(&self, input: I) -> Self::Output {
        self.invoke_with(input, TraceMode::Infallible, |input| {
            Ok::<_, Infallible>(self.func.invoke_owned(input))
        })
        .unwrap_or_else(|never| match never {})
    }
}

impl<I, F> TryCallable<I> for Jit<F>
where
    I: GraphEq + GraphHash,
    I::Owned: GraphEq,
    F: TryJitable<I>,
    I::Builder<ObjectTracer>: Send + Sync,
    F::Output: Graph<Builder<ObjectTracer>: Send + Sync>,
{
    type Output = <F as TryCallable<I>>::Output;
    type Error = <F as TryCallable<I>>::Error;

    fn try_invoke(&self, input: I) -> Result<Self::Output, Self::Error> {
        self.invoke_with(input, TraceMode::Fallible, |input| {
            self.func.try_invoke_owned(input)
        })
    }
}

impl<F> Jit<F> {
    const fn new(func: F) -> Self {
        Self {
            func,
            traces: TraceCache::new(),
        }
    }

    fn invoke_with<I, O, E>(
        &self,
        input: I,
        mode: TraceMode,
        body: impl FnOnce(I::Owned) -> Result<O, E>,
    ) -> Result<O, E>
    where
        I: GraphEq + GraphHash,
        I::Owned: GraphEq,
        I::Builder<ObjectTracer>: Send + Sync,
        O: Graph<Owned = O, Builder<ObjectTracer>: Send + Sync>,
    {
        type Slot<I, O> =
            TraceSlot<<I as Graph>::Builder<ObjectTracer>, <O as Graph>::Builder<ObjectTracer>>;
        let filter = &Of::<ObjectTracer>::filter();
        let input_builder = input
            .builder::<ObjectTracer, _, SourceError>(filter, &mut GraphContext::new())
            .expect("JIT could not build input graph");
        let arguments = input.collect_leaves(filter);
        let kinds: Vec<_> = arguments.iter().map(kind).collect();

        let mut hasher = DefaultHasher::new();
        I::hash_builder::<_, ObjectTracer>(
            &input_builder,
            &mut hasher,
            &mut GraphHashContext::new(),
        );
        kinds.hash(&mut hasher);
        let key = (
            TypeId::of::<(I::Owned, Slot<I, O>)>(),
            std::any::type_name::<I>(),
            mode,
            hasher.finish(),
        );
        let slot = self.traces.slot(
            key,
            Slot::<I, O> {
                input_builder,
                kinds,
                entry: Mutex::new(None),
            },
            |cached, new| {
                cached.kinds == new.kinds
                    && same_builder::<I>(&cached.input_builder, &new.input_builder, &arguments)
            },
        );
        let entry = slot.get_or_trace(|| {
            let guard = TraceContext::enter();
            let placeholders: Vec<_> = slot.kinds.iter().map(placeholder).collect();
            let traced_input =
                build_from_leaves(slot.input_builder.clone(), placeholders.iter().cloned())
                    .expect("JIT could not reconstruct placeholder arguments");
            let output = body(traced_input)?;
            let output_builder = output
                .builder::<ObjectTracer, _, SourceError>(filter, &mut GraphContext::new())
                .expect("JIT could not build output graph");
            let outputs = output.collect_leaves(filter);
            let capture = Capture::from_context(guard.context(), &placeholders, &outputs)
                .expect("JIT could not capture traced function");
            drop(guard);
            Ok(Arc::new(TraceEntry {
                capture,
                output_builder,
            }))
        })?;

        let results = entry.capture.call(&arguments);
        let output = build_from_leaves(entry.output_builder.clone(), results)
            .expect("JIT could not reconstruct function results");
        Ok(output)
    }
}

/// Wrap a function without imposing a callable bound at construction time.
/// Use `Callable::invoke` for infallible functions or `TryCallable::try_invoke`
/// for fallible functions. A body error is returned unchanged and not cached.
pub const fn jit<F>(func: F) -> Jit<F> {
    Jit::new(func)
}

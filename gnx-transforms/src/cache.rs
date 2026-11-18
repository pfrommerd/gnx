use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, OnceLock, TryLockError};

use dashmap::DashMap;
use gnx_expr::{
    Capture, CellTrace, ContextID, HExpr, Invocation, Kind, ObjectTracer, Trace, TraceContext,
};
use gnx_graph::anyhash::HasherWrite;
use gnx_graph::util::{GraphExt, IterSource};
use gnx_graph::{
    Builder, Graph, GraphContext, GraphEq, GraphEqContext, GraphHash, GraphHashContext, Of,
    SourceError,
};

pub(super) type TracerBuilder<I> = <I as Graph>::Builder<ObjectTracer>;
type Slots<I, O> = DashMap<Arc<TraceKey<<I as Graph>::Owned>>, Arc<TraceSlot<O>>>;

pub(super) struct TraceCache<I: Graph, O: Graph> {
    slots: OnceLock<Slots<I, O>>,
}

impl<I: Graph, O: Graph> TraceCache<I, O> {
    pub(super) const fn new() -> Self {
        Self {
            slots: OnceLock::new(),
        }
    }
}

impl<I, O> TraceCache<I, O>
where
    I: Graph,
    I::Owned: GraphHash + GraphEq<Builder<ObjectTracer> = TracerBuilder<I>>,
    O: Graph<Owned = O>,
{
    pub(super) fn invoke<A, E>(
        &self,
        input: A,
        body: impl FnOnce(I::Owned) -> Result<O, E>,
    ) -> Result<O, E>
    where
        A: Graph<Owned = I::Owned, Builder<ObjectTracer> = TracerBuilder<I>>,
    {
        let (entry, arguments) = self.get_or_trace(input, body)?;
        Ok(entry.call(&arguments))
    }

    pub(super) fn trace<E>(
        &self,
        input: I,
        body: impl FnOnce(I::Owned) -> Result<O, E>,
    ) -> Result<Arc<HExpr>, E> {
        let (entry, _) = self.get_or_trace(input, body)?;
        Ok(entry.expr.clone())
    }

    fn get_or_trace<A, E>(
        &self,
        input: A,
        body: impl FnOnce(I::Owned) -> Result<O, E>,
    ) -> Result<(Arc<TraceEntry<O>>, Vec<ObjectTracer>), E>
    where
        A: Graph<Owned = I::Owned, Builder<ObjectTracer> = TracerBuilder<I>>,
    {
        let (builder, arguments) = split_graph(&input);
        let kinds = arguments.iter().map(kind).collect();
        let key: Arc<TraceKey<I::Owned>> = Arc::new(TraceKey { builder, kinds });
        let slot = self.slot(&key);
        let entry = slot.get_or_trace(|| TraceEntry::trace(&key.builder, &key.kinds, body))?;
        Ok((entry, arguments))
    }

    fn slot(&self, key: &Arc<TraceKey<I::Owned>>) -> Arc<TraceSlot<O>> {
        let slots = self.slots.get_or_init(DashMap::new);
        if let Some(slot) = slots.get(key) {
            return Arc::clone(slot.value());
        }
        // No shard guard escapes this function or is held while tracing.
        slots.entry(key.clone()).or_default().value().clone()
    }
}

struct TraceSlot<O: Graph>(Mutex<Option<Arc<TraceEntry<O>>>>);

impl<O: Graph> Default for TraceSlot<O> {
    fn default() -> Self {
        Self(Mutex::new(None))
    }
}

impl<O: Graph<Owned = O>> TraceSlot<O> {
    fn get_or_trace<E>(
        &self,
        trace: impl FnOnce() -> Result<TraceEntry<O>, E>,
    ) -> Result<Arc<TraceEntry<O>>, E> {
        // Nested tracing bodies can call each other. Don't wait on a busy slot
        // in that case; use an unpublished trace instead.
        let mut cached = if TraceContext::current().is_some() {
            match self.0.try_lock() {
                Ok(cached) => cached,
                Err(TryLockError::Poisoned(err)) => err.into_inner(),
                Err(TryLockError::WouldBlock) => return trace().map(Arc::new),
            }
        } else {
            self.0.lock().unwrap_or_else(|err| err.into_inner())
        };
        if let Some(entry) = cached.as_ref() {
            return Ok(entry.clone());
        }
        let entry = Arc::new(trace()?);
        if entry.is_cacheable() {
            *cached = Some(entry.clone());
        }
        Ok(entry)
    }
}

struct TraceEntry<O: Graph> {
    expr: Arc<HExpr>,
    closure: Vec<ObjectTracer>,
    output_builder: TracerBuilder<O>,
}

impl<O: Graph<Owned = O>> TraceEntry<O> {
    fn is_cacheable(&self) -> bool {
        // A trace can outlive its caller's scope, but a captured local handle
        // cannot. Concrete values and root handles are safe to keep.
        self.closure.iter().all(|object| match object {
            ObjectTracer::Ref(value) => {
                value.try_concrete().is_some() || value.context_id() == &ContextID::ROOT
            }
            ObjectTracer::Cell(cell) => cell.context_id() == &ContextID::ROOT,
        })
    }
    fn trace<B: Builder<ObjectTracer>, E>(
        input_builder: &B,
        kinds: &[Kind],
        body: impl FnOnce(B::Graph) -> Result<O, E>,
    ) -> Result<Self, E> {
        let guard = TraceContext::enter();
        let placeholders: Vec<_> = kinds.iter().map(placeholder).collect();
        let input = rebuild_graph(input_builder, placeholders.iter().cloned());
        let output = body(input)?;
        let (output_builder, outputs) = split_graph(&output);
        let capture = Capture::from_context(guard.context(), &placeholders, &outputs)
            .expect("JIT could not capture traced function");
        let (expr, closure, ..) = capture.into_parts();
        Ok(Self {
            expr: Arc::new(expr),
            closure,
            output_builder,
        })
    }

    fn call(&self, arguments: &[ObjectTracer]) -> O {
        rebuild_graph(
            &self.output_builder,
            Invocation::defer_call(self.expr.clone(), &self.closure, arguments),
        )
    }
}

struct TraceKey<G: Graph> {
    builder: TracerBuilder<G>,
    kinds: Vec<Kind>,
}

impl<G: GraphHash> Hash for TraceKey<G> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        G::hash_builder::<_, ObjectTracer>(
            &self.builder,
            &mut GraphHashWriter(state),
            &mut GraphHashContext::new(),
        );
        self.kinds.hash(state);
    }
}

impl<G: GraphEq> PartialEq for TraceKey<G> {
    fn eq(&self, other: &Self) -> bool {
        self.kinds == other.kinds
            && G::builders_eq::<ObjectTracer>(
                &self.builder,
                &other.builder,
                &mut GraphEqContext::new(),
            )
    }
}

impl<G: GraphEq> Eq for TraceKey<G> {}

struct GraphHashWriter<'a, H>(&'a mut H);

impl<H: Hasher> HasherWrite for GraphHashWriter<'_, H> {
    fn write(&mut self, bytes: &[u8]) {
        self.0.write(bytes);
    }
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

fn split_graph<G: Graph>(graph: &G) -> (TracerBuilder<G>, Vec<ObjectTracer>) {
    let filter = &Of::<ObjectTracer>::filter();
    let builder = graph
        .builder::<ObjectTracer, _, SourceError>(filter, &mut GraphContext::new())
        .expect("JIT could not build graph template");
    (builder, graph.collect_leaves(filter))
}

fn rebuild_graph<B: Builder<ObjectTracer>>(
    builder: &B,
    leaves: impl IntoIterator<Item = ObjectTracer>,
) -> B::Graph {
    let mut leaves = leaves.into_iter();
    let graph = builder
        .clone()
        .build(IterSource::new(&mut leaves), &mut GraphContext::new())
        .expect("JIT could not rebuild graph");
    assert!(
        leaves.next().is_none(),
        "JIT builder did not consume all leaves"
    );
    graph
}

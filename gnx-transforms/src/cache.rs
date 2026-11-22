use std::borrow::Cow;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, OnceLock, TryLockError};

use dashmap::DashMap;
use gnx_expr::{
    Capture, CellTrace, ContextID, HExpr, Kind, ObjectTracer, Trace, TraceContext, VarInfo,
};
use gnx_graph::anyhash::HasherWrite;
use gnx_graph::util::{GraphExt, IterSource};
use gnx_graph::{
    Builder, Graph, GraphContext, GraphEq, GraphEqContext, GraphHash, GraphHashContext, Key, Of,
    SourceError,
};

pub(crate) type TracerBuilder<I> = <I as Graph>::Builder<ObjectTracer>;
type Slots<I, O> = DashMap<Arc<TraceKey<<I as Graph>::Owned>>, Arc<TraceSlot<O>>>;

pub(crate) struct TraceCache<I: Graph, O: Graph> {
    slots: OnceLock<Slots<I, O>>,
}

impl<I: Graph, O: Graph> TraceCache<I, O> {
    pub(crate) const fn new() -> Self {
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
    pub(crate) fn invoke<A, E>(
        &self,
        input: A,
        name_hint: Option<&Cow<'static, str>>,
        body: impl FnOnce(I::Owned) -> Result<O, E>,
    ) -> Result<O, E>
    where
        A: Graph<Owned = I::Owned, Builder<ObjectTracer> = TracerBuilder<I>>,
    {
        let (entry, arguments) = self.get_or_trace(input, name_hint, body)?;
        Ok(entry.call(&arguments))
    }

    pub(crate) fn trace<E>(
        &self,
        input: I,
        name_hint: Option<&Cow<'static, str>>,
        body: impl FnOnce(I::Owned) -> Result<O, E>,
    ) -> Result<Arc<HExpr>, E> {
        let (entry, _) = self.get_or_trace(input, name_hint, body)?;
        Ok(entry.expr.clone())
    }

    fn get_or_trace<A, E>(
        &self,
        input: A,
        name_hint: Option<&Cow<'static, str>>,
        body: impl FnOnce(I::Owned) -> Result<O, E>,
    ) -> Result<(Arc<TraceEntry<O>>, Vec<ObjectTracer>), E>
    where
        A: Graph<Owned = I::Owned, Builder<ObjectTracer> = TracerBuilder<I>>,
    {
        let (builder, arguments) = split_graph(&input);
        let kinds = arguments.iter().map(kind).collect();
        let entry = self.entry_with_key(builder, kinds, name_hint, body, true)?;
        Ok((entry, arguments))
    }

    /// The trace for an explicit signature: the input structure and one kind
    /// per tracer leaf, in leaf order. Transformations trace at signatures
    /// that differ from their arguments (e.g. `map` removes mapped axes).
    pub(crate) fn entry<E>(
        &self,
        builder: TracerBuilder<I>,
        kinds: Vec<(Kind, VarInfo)>,
        name_hint: Option<&Cow<'static, str>>,
        body: impl FnOnce(I::Owned) -> Result<O, E>,
    ) -> Result<Arc<TraceEntry<O>>, E> {
        self.entry_with_key(builder, kinds, name_hint, body, false)
    }
    fn entry_with_key<E>(
        &self,
        builder: TracerBuilder<I>,
        kinds: Vec<(Kind, VarInfo)>,
        name_hint: Option<&Cow<'static, str>>,
        body: impl FnOnce(I::Owned) -> Result<O, E>,
        abstract_ids: bool,
    ) -> Result<Arc<TraceEntry<O>>, E> {
        // Call sites instantiate a cached body's input symbols. Transforms
        // requesting an explicit signature consume its original namespace.
        let mut scope = gnx_expr::VarScope::new();
        let mut ids = std::collections::BTreeMap::new();
        let ctx = TraceContext::current();
        let key_kinds = kinds
            .iter()
            .map(|(k, info)| {
                (
                    *k,
                    info.remap_symbols(|s| {
                        *ids.entry(s).or_insert_with(|| {
                            assert!(
                                ctx.as_ref().is_some_and(|c| c.has_symbol(s)),
                                "symbol not declared in tracing scope"
                            );
                            if abstract_ids {
                                scope.create_symbol()
                            } else {
                                s
                            }
                        })
                    }),
                )
            })
            .collect();
        let key: Arc<TraceKey<I::Owned>> = Arc::new(TraceKey {
            builder,
            kinds: key_kinds,
            abstract_ids: abstract_ids && !ids.is_empty(),
        });
        let slot = self.slot(&key);
        slot.get_or_trace(|| TraceEntry::trace(&key.builder, &kinds, name_hint, body))
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

pub(crate) struct TraceEntry<O: Graph> {
    pub(crate) expr: Arc<HExpr>,
    pub(crate) closure: Vec<ObjectTracer>,
    pub(crate) output_builder: TracerBuilder<O>,
    /// Key path of each output tracer leaf, in leaf order.
    pub(crate) output_paths: Vec<Vec<Key>>,
}

impl<O: Graph<Owned = O>> TraceEntry<O> {
    fn is_cacheable(&self) -> bool {
        // A trace can outlive its caller's scope, but a captured local handle
        // cannot. Concrete values and root handles are safe to keep.
        self.expr.symbol_closure().is_empty()
            && self.closure.iter().all(|object| match object {
                ObjectTracer::Ref(value) => {
                    value.info().symbols().is_empty()
                        && (value.try_concrete().is_some()
                            || value.context_id() == &ContextID::ROOT)
                }
                ObjectTracer::Cell(cell) => {
                    cell.type_info().symbols().is_empty() && cell.context_id() == &ContextID::ROOT
                }
            })
    }
    fn trace<B: Builder<ObjectTracer>, E>(
        input_builder: &B,
        kinds: &[(Kind, VarInfo)],
        name_hint: Option<&Cow<'static, str>>,
        body: impl FnOnce(B::Graph) -> Result<O, E>,
    ) -> Result<Self, E> {
        let guard = TraceContext::enter();
        let placeholders: Vec<_> = kinds.iter().map(placeholder).collect();
        let input = rebuild_graph(input_builder, placeholders.iter().cloned());
        let output = body(input)?;
        let (output_builder, outputs) = split_graph(&output);
        let output_paths = output
            .collect_leaves_with_paths(&Of::<ObjectTracer>::filter())
            .into_iter()
            .map(|(path, _)| path)
            .collect();
        let capture = Capture::from_context(guard.context(), &placeholders, &outputs)
            .expect("JIT could not capture traced function");
        let (expr, closure, ..) = capture.into_parts();
        let expr = match name_hint {
            Some(hint) => expr.with_name_hint(hint.clone()),
            None => expr,
        };
        Ok(Self {
            expr: Arc::new(expr),
            closure,
            output_builder,
            output_paths,
        })
    }

    fn call(&self, arguments: &[ObjectTracer]) -> O {
        rebuild_graph(
            &self.output_builder,
            crate::cf::defer_call(self.expr.clone(), &self.closure, arguments),
        )
    }
}

struct TraceKey<G: Graph> {
    abstract_ids: bool,
    builder: TracerBuilder<G>,
    kinds: Vec<(Kind, VarInfo)>,
}

impl<G: GraphHash> Hash for TraceKey<G> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        G::hash_builder::<_, ObjectTracer>(
            &self.builder,
            &mut GraphHashWriter(state),
            &mut GraphHashContext::new(),
        );
        self.abstract_ids.hash(state);
        self.kinds.hash(state);
    }
}

impl<G: GraphEq> PartialEq for TraceKey<G> {
    fn eq(&self, other: &Self) -> bool {
        self.abstract_ids == other.abstract_ids
            && self.kinds == other.kinds
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

pub(crate) fn kind(object: &ObjectTracer) -> (Kind, VarInfo) {
    match object {
        ObjectTracer::Ref(value) => (Kind::Value, value.info().clone()),
        ObjectTracer::Cell(cell) => (Kind::Cell, cell.type_info().clone()),
    }
}

fn placeholder(kind: &(Kind, VarInfo)) -> ObjectTracer {
    match kind {
        (Kind::Value, info) => Trace::placeholder(info.clone()).into(),
        (Kind::Cell, info) => CellTrace::placeholder(info.clone()).into(),
    }
}

pub(crate) fn split_graph<G: Graph>(graph: &G) -> (TracerBuilder<G>, Vec<ObjectTracer>) {
    let filter = &Of::<ObjectTracer>::filter();
    let builder = graph
        .builder::<ObjectTracer, _, SourceError>(filter, &mut GraphContext::new())
        .expect("JIT could not build graph template");
    (builder, graph.collect_leaves(filter))
}

pub(crate) fn rebuild_graph<B: Builder<ObjectTracer>>(
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

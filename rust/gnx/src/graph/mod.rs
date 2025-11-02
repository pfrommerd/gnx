mod callable;
mod impls;
mod path;
mod visitors;

pub mod filter;
pub mod util;

pub use filter::{GraphCow, GraphFilter, NonLeafRepr};
pub use gnx_derive::{Leaf, impl_leaf};
pub use impls::{LeafDef, NodeDef};

pub use visitors::{Bound, View, ViewMut};
pub use visitors::{ChildrenConsumer, ChildrenMutVisitor, ChildrenVisitor};
pub use visitors::{GraphConsumer, GraphMutVisitor, GraphVisitor};

pub use callable::Callable;
pub use path::{GraphId, Key, KeyRef, Path};

use std::any::{Any, TypeId};
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::error::Error;

pub trait Graph {
    // The owned version of this graph node
    // Note that the GraphDef is of the Owned type!
    type GraphDef<I: Clone + 'static, R: NonLeafRepr>: GraphDef<I, Graph = Self::Owned>;
    // The "owned" graph needs to be static and cloneable
    type Owned: Graph + Clone + 'static;

    fn graph_def<I, L, F, M>(
        &self,
        filter: F,
        map: M,
        ctx: &mut GraphContext,
    ) -> Result<Self::GraphDef<I, F::NonLeafRepr>, GraphError>
    where
        I: Clone + 'static,
        F: GraphFilter<L>,
        M: FnMut(F::Ref<'_>) -> I;

    fn into_graph_def<I, L, F, M>(
        self,
        filter: F,
        map: M,
        ctx: &mut GraphContext,
    ) -> Result<Self::GraphDef<I, F::NonLeafRepr>, GraphError>
    where
        I: Clone + 'static,
        F: GraphFilter<L>,
        M: FnMut(GraphCow<F::Ref<'_>, L>) -> I;

    fn visit<L, F, V>(&self, filter: F, visitor: impl Into<V>) -> V::Output
    where
        F: GraphFilter<L>,
        V: GraphVisitor<L, F>;

    // visit_mut will visit all mutablely accessible children, mutably.
    // When a node contains immutable shared children (e.g. Rc<T>)
    // these can be traversed using visit_mut_inner, which takes &self
    // and acts like visit(), but will attempt to recursively
    // visit children like RefCell<T> mutably if possible.
    // This allows us to track mutation state
    fn mut_visit<L, F, M>(&mut self, filter: F, visitor: impl Into<M>) -> M::Output
    where
        F: GraphFilter<L>,
        M: GraphMutVisitor<L, F>;

    fn inner_mut_visit<L, F, M, O>(&self, filter: F, visitor: impl Into<M>) -> O
    where
        F: GraphFilter<L>,
        M: GraphVisitor<L, F, Output = O> + GraphMutVisitor<L, F, Output = O>,
    {
        self.visit(filter, visitor)
    }

    fn into_visit<L, V, M>(self, view: V, consumer: impl Into<M>) -> M::Output
    where
        V: GraphFilter<L>,
        M: GraphConsumer<L, V>;
}

pub trait Node: Graph {
    fn visit_children<L, F, V>(&self, filter: F, visitor: impl Into<V>) -> V::Output
    where
        F: GraphFilter<L>,
        V: ChildrenVisitor<L, F>;
    fn visit_children_mut<L, F, V>(&mut self, filter: F, visitor: impl Into<V>) -> V::Output
    where
        F: GraphFilter<L>,
        V: ChildrenMutVisitor<L, F>;
    fn consume_children<L, F, C>(self, filter: F, map: impl Into<C>) -> C::Output
    where
        F: GraphFilter<L>,
        C: ChildrenConsumer<L, F>;
}

pub trait GraphDef<I: Clone + 'static>: Clone + 'static {
    type Graph: Graph + 'static;

    fn build<L, B, S>(
        &self,
        builder: B,
        source: S,
        ctx: &mut GraphContext,
    ) -> Result<Self::Graph, S::Error>
    where
        B: LeafBuilder<I, L>,
        S: GraphSource<I, L>;

    fn into_build<L, B, S>(
        self,
        builder: B,
        source: S,
        ctx: &mut GraphContext,
    ) -> Result<Self::Graph, S::Error>
    where
        B: LeafBuilder<I, L>,
        S: GraphSource<I, L>;
}

pub trait GraphSource<I: Clone + 'static, L> {
    type Error: Error + From<GraphError>;
    // Whether this is a shared node
    fn id(&self) -> Option<GraphId>;
    // Try to construct a leaf given a value
    fn leaf(self, info: Cow<'_, I>) -> Result<L, Self::Error>;
    fn node(self) -> impl ChildrenSource<I, L, Error = Self::Error>;
}

pub trait ChildrenSource<I: Clone + 'static, L> {
    type Error: Error + From<&'static str>;

    #[rustfmt::skip]
    fn child(&mut self, key: KeyRef<'_>) -> Result<Option<
        impl GraphSource<I, L, Error=Self::Error>
    >, Self::Error>;
}

// LeafBuilder is the counterpart to GraphFilter.
// It allows constructing arbitrary Graph types given an associated Def and a value
pub trait LeafBuilder<I: Clone + 'static, L>: Copy {
    fn try_build<G: Graph>(&self, value: L) -> Result<G, GraphError>;
}

pub enum GraphError {
    MissingNode,
    UnsupportedLeafDef,
    ContextError,
}

// A GraphContext stores already-constructed graph nodes
// in a generic type-erased way
// The proper way of interacting with it is through the ctx_build! macro,
// which releases the &mut borrow on the context while building child nodes
#[derive(Default)]
pub struct GraphContext {
    // The boxes contain HashMap<GraphId, T>
    maps: HashMap<TypeId, Box<dyn Any>>,
    // All the GraphIds we've seen so far
    seen: HashSet<GraphId>,
}

impl GraphContext {
    pub fn new() -> Self {
        Self::default()
    }

    // For use by the ctx_build_shared macro
    // through which you should interact with the GraphContext
    // The macro released the &mut borrow while building child nodes
    pub fn _reserve<T: Clone + 'static>(&mut self, id: GraphId) -> Result<Option<T>, GraphError> {
        if self.seen.contains(&id) {
            return Err(GraphError::ContextError);
        }
        self.seen.insert(id);
        let map = self
            .maps
            .entry(TypeId::of::<T>())
            .or_insert_with(|| Box::new(HashMap::<GraphId, T>::new()));
        let map = map
            .downcast_mut::<HashMap<GraphId, T>>()
            .ok_or(GraphError::ContextError)?;
        if let Some(s) = map.get(&id) {
            Ok(Some(s.clone()))
        } else {
            Ok(None)
        }
    }
    pub fn _finish<T: Clone + 'static>(&mut self, id: GraphId, value: T) -> Result<(), GraphError> {
        let map = self
            .maps
            .entry(TypeId::of::<T>())
            .or_insert_with(|| Box::new(HashMap::<GraphId, T>::new()));
        // self.seen.insert(id);
        let map = map
            .downcast_mut::<HashMap<GraphId, T>>()
            .ok_or(GraphError::ContextError)?;
        map.insert(id, value);
        Ok(())
    }

    pub fn build<T: Clone + 'static, F: FnOnce(&mut Self) -> T>(
        &mut self,
        id: GraphId,
        builder: F,
    ) -> Result<T, GraphError> {
        self._reserve::<T>(id)?
            .ok_or(GraphError::ContextError)
            .or_else(|_| {
                let value = builder(self);
                self._finish::<T>(id, value.clone())?;
                Ok(value)
            })
    }
}

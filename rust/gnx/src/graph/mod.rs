mod callable;
mod context;
mod impls;
mod path;
mod views;
mod visitors;

pub mod filters;
pub mod util;

pub use context::*;
pub use filters::*;
pub use impls::*;
pub use views::*;
pub use visitors::*;

pub use gnx_derive::{Graph, Leaf};

pub use callable::Callable;
pub use path::{GraphId, Key, KeyRef, Path};

use castaway::LifetimeFree;
use std::error::Error;

pub trait Graph {
    // The owned version of this graph node
    // Note that the GraphDef is of the Owned type!
    type GraphDef<I: Leaf, R: StaticRepr>: GraphDef<I, Graph = Self::Owned>;
    // The "owned" graph is one that is both cloneable and static
    // regular graphs may not be cloneable (e.g. contain &mut references)
    type Owned: Graph + Clone + 'static;

    // to_owned and into_owned are
    // implemented in terms of graph_def by default
    // but can be overridden for efficiency gains
    fn graph_def<I: Leaf, L: Leaf, F, M>(
        &self,
        filter: F,
        map: impl Into<M>,
        ctx: &mut GraphContext,
    ) -> Result<Self::GraphDef<I, F::StaticRepr>, GraphError>
    where
        F: Filter<L>,
        M: FnMut(L::Ref<'_>) -> I;

    fn into_graph_def<I: Leaf, L: Leaf, F, M>(
        self,
        filter: F,
        map: impl Into<M>,
        ctx: &mut GraphContext,
    ) -> Result<Self::GraphDef<I, F::StaticRepr>, GraphError>
    where
        F: Filter<L>,
        M: FnMut(LeafCow<'_, L>) -> I;

    fn visit<L, F, V>(&self, filter: F, visitor: impl Into<V>) -> V::Output
    where
        L: Leaf,
        F: Filter<L>,
        V: GraphVisitor<L>;

    fn into_visit<L, F, C>(self, filter: F, consumer: impl Into<C>) -> C::Output
    where
        L: Leaf,
        F: Filter<L>,
        C: GraphConsumer<L>;

    // visit_mut will visit all mutablely accessible children, mutably.
    // When a node contains immutable shared children (e.g. Rc<T>)
    // these can be traversed using visit_mut_inner, which takes &self
    // and acts like visit(), but will attempt to recursively
    // visit children like RefCell<T> mutably if possible.
    // This allows us to track mutation state
    fn mut_visit<L, F, M>(&mut self, filter: F, visitor: impl Into<M>) -> M::Output
    where
        L: Leaf,
        F: Filter<L>,
        M: GraphMutVisitor<L>;

    fn inner_mut_visit<L, F, M, O>(&self, filter: F, visitor: impl Into<M>) -> O
    where
        L: Leaf,
        F: Filter<L>,
        M: GraphVisitor<L, Output = O> + GraphMutVisitor<L, Output = O>,
    {
        self.visit(filter, visitor)
    }
}

pub trait Node: Graph {
    fn visit_children<L, F, V>(&self, filter: F, visitor: impl Into<V>) -> V::Output
    where
        L: Leaf,
        F: Filter<L>,
        V: ChildrenVisitor<L>;
    fn visit_children_mut<L, F, V>(&mut self, filter: F, visitor: impl Into<V>) -> V::Output
    where
        L: Leaf,
        F: Filter<L>,
        V: ChildrenMutVisitor<L>;
    fn consume_children<L, F, C>(self, filter: F, consumer: impl Into<C>) -> C::Output
    where
        L: Leaf,
        F: Filter<L>,
        C: ChildrenConsumer<L>;
}

pub trait TypedGraph<I: Leaf>: Graph {}

pub trait Leaf: TypedGraph<Self> + Clone + 'static {
    type Ref<'l>: Copy
    where
        Self: 'l;
    type RefMut<'l>
    where
        Self: 'l;

    fn as_ref<'l>(&'l self) -> Self::Ref<'l>;
    fn as_mut<'l>(&'l mut self) -> Self::RefMut<'l>;

    fn clone_ref(v: Self::Ref<'_>) -> Self;
    fn clone_mut(v: Self::RefMut<'_>) -> Self;

    fn try_from_value<V>(g: V) -> Result<Self, V>;
    fn try_as_ref<'v, V>(graph: &'v V) -> Result<Self::Ref<'v>, &'v V>;
    fn try_as_mut<'v, V>(graph: &'v mut V) -> Result<Self::RefMut<'v>, &'v mut V>;
}

pub enum LeafCow<'l, L: Leaf + 'l> {
    Borrowed(L::Ref<'l>),
    Owned(L),
}

impl<'l, L: Leaf + 'l> LeafCow<'l, L> {
    fn as_ref<'s: 'l>(&'s self) -> L::Ref<'s> {
        match self {
            LeafCow::Borrowed(r) => *r,
            LeafCow::Owned(o) => o.as_ref(),
        }
    }
}

// The GraphDef type
pub trait GraphDef<I: Leaf>: TypedGraph<I> + Clone + 'static {
    type Graph: Graph + 'static;

    fn build<V: Value, S: GraphSource<I, V>>(
        &self,
        source: S,
        ctx: &mut GraphContext,
    ) -> Result<Self::Graph, S::Error>;

    fn into_build<V: Value, S: GraphSource<I, V>>(
        self,
        source: S,
        ctx: &mut GraphContext,
    ) -> Result<Self::Graph, S::Error>;
}

pub trait Value: Sized {
    fn convert_into<G: Graph>(self) -> Result<G, Self>;
}

pub trait GraphSource<I: Leaf, V: Value> {
    type Error: Error + From<GraphError>;
    // Whether this is a shared node
    fn id(&self) -> Option<GraphId>;
    // Try to construct a leaf given a value
    fn leaf(self, info: LeafCow<I>) -> Result<V, Self::Error>;
    fn node(self) -> impl ChildrenSource<I, V, Error = Self::Error>;
}

pub trait ChildrenSource<I: Leaf, V: Value> {
    type Error: Error + From<&'static str>;

    #[rustfmt::skip]
    fn child(&mut self, key: KeyRef<'_>) -> Result<Option<
        impl GraphSource<I, V, Error=Self::Error>
    >, Self::Error>;
}

pub enum GraphError {
    GraphDefUnsupported,
    ReprUnsupported,
    MissingNode,
    ContextError,
    InvalidType,
}

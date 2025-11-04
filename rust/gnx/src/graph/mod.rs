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
pub use path::*;
pub use views::*;
pub use visitors::*;

pub use gnx_derive::{Graph, Leaf};

pub use callable::Callable;

use std::error::Error;

pub trait Builder<L: Leaf>: Clone {
    type Graph: Graph + Clone + 'static;
    type Owned: Builder<L, Graph = Self::Graph> + 'static;

    fn build<S: GraphSource<(), L>>(
        self,
        source: S,
        ctx: &mut GraphContext,
    ) -> Result<Self::Graph, S::Error>;

    fn owned_builder(&self, ctx: &mut GraphContext) -> Self::Owned;
}

#[rustfmt::skip]
pub trait Graph {
    // The owned version of this graph node
    type Owned: Graph + Clone + 'static;

    // The Builder types
    type Builder<'g, L, F>: Builder<L, Graph = Self::Owned, Owned = Self::OwnedBuilder<L>>
        where Self: 'g, L: Leaf, F: Filter<L>;
    type OwnedBuilder<L: Leaf>: Builder<L, Graph = Self::Owned> + 'static;

    fn builder<'g, L: Leaf, F: Filter<L>>(&'g self, filter: F) -> Self::Builder<'g, L, F>;

    fn owned_graph(&self, ctx: &mut GraphContext) -> Self::Owned {
        let builder = self.builder(IgnoreAll::<()>::filter());
        builder.build(NullSource, ctx).unwrap()
    }

    fn visit<L: Leaf, F: Filter<L>, V: GraphVisitor<Self, L>>(
        &self, filter: F, visitor: impl Into<V>
    ) -> V::Output;

    fn into_visit<L: Leaf, F: Filter<L>, C: GraphConsumer<Self, L>>(
        self, filter: F, consumer: impl Into<C>
    ) -> C::Output;

    // visit_mut will visit all mutablely accessible children, mutably.
    // When a node contains immutable shared children (e.g. Rc<T>)
    // these can be traversed using visit_mut_inner, which takes &self
    // and acts like visit(), but will attempt to recursively
    // visit children like RefCell<T> mutably if possible.
    // This allows us to track mutation state
    fn mut_visit<L: Leaf, F: Filter<L>, V: GraphMutVisitor<Self, L>>(
        &mut self, filter: F, visitor: impl Into<V>
    ) -> V::Output;

    fn inner_mut_visit<L, F, V, O>(&self, filter: F, visitor: impl Into<V>) -> O
    where
        L: Leaf,
        F: Filter<L>,
        V: GraphVisitor<Self, L, Output = O> + GraphMutVisitor<Self, L, Output = O>,
    {
        self.visit(filter, visitor)
    }
}

pub trait Node: Graph {
    fn visit_children<L, F, V>(&self, filter: F, visitor: impl Into<V>) -> V::Output
    where
        L: Leaf,
        F: Filter<L>,
        V: ChildrenVisitor<Self, L>;
    fn visit_children_mut<L, F, V>(&mut self, filter: F, visitor: impl Into<V>) -> V::Output
    where
        L: Leaf,
        F: Filter<L>,
        V: ChildrenMutVisitor<Self, L>;
    fn consume_children<L, F, C>(self, filter: F, consumer: impl Into<C>) -> C::Output
    where
        L: Leaf,
        F: Filter<L>,
        C: ChildrenConsumer<Self, L>;
}

pub trait TypedGraph<L: Leaf>: Graph {}

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
    fn try_from_ref<'v, V>(graph: &'v V) -> Result<Self::Ref<'v>, &'v V>;
    fn try_from_mut<'v, V>(graph: &'v mut V) -> Result<Self::RefMut<'v>, &'v mut V>;

    fn try_into_value<V: 'static>(self) -> Result<V, Self>;
}

pub enum LeafCow<'l, L: Leaf + 'l> {
    Borrowed(L::Ref<'l>),
    Owned(L),
}

impl<'l, L: Leaf + 'l> LeafCow<'l, L> {
    pub fn as_ref<'s: 'l>(&'s self) -> L::Ref<'s> {
        match self {
            LeafCow::Borrowed(r) => *r,
            LeafCow::Owned(o) => o.as_ref(),
        }
    }
}

pub trait GraphSource<I, L> {
    type Error: Error + From<GraphError>;
    type ChildrenSource: ChildrenSource<I, L, Error = Self::Error>;
    // Whether this is a shared node
    fn id(&self) -> Option<GraphId>;
    // Try to construct a leaf given a value
    fn leaf(self, info: I) -> Result<L, Self::Error>;
    fn node(self) -> Result<Self::ChildrenSource, Self::Error>;
}

// TODO:
// impl<L: Leaf, G: TypedGraph<L>> GraphSource<(), L> for G {
// }

pub trait ChildrenSource<I, L> {
    type Error: Error + From<GraphError>;
    type ChildSource: GraphSource<I, L, Error = Self::Error>;
    #[rustfmt::skip]
    fn child(&mut self, key: KeyRef<'_>) -> Result<Option<Self::ChildSource>, Self::Error>;
    #[rustfmt::skip]
    fn next(&mut self) -> Result<Option<(Key, Self::ChildSource)>, Self::Error>;
}

pub struct NullSource;

impl<I, L> GraphSource<I, L> for NullSource {
    type Error = GraphError;
    type ChildrenSource = NullSource;

    fn id(&self) -> Option<GraphId> {
        None
    }

    fn leaf(self, _info: I) -> Result<L, Self::Error> {
        Err(GraphError::MissingChild)
    }

    fn node(self) -> Result<NullSource, Self::Error> {
        Ok(NullSource)
    }
}
impl<I, L> ChildrenSource<I, L> for NullSource {
    type Error = GraphError;
    type ChildSource = NullSource;

    fn child(&mut self, _key: KeyRef<'_>) -> Result<Option<NullSource>, Self::Error> {
        Ok(Some(NullSource))
    }

    fn next(&mut self) -> Result<Option<(Key, NullSource)>, Self::Error> {
        Err(GraphError::MissingChild)
    }
}

#[derive(Debug)]
pub enum GraphError {
    GraphDefUnsupported,
    ReprUnsupported,
    MissingChild,
    InvalidLeaf,
    ContextError,
}

impl std::fmt::Display for GraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GraphError::GraphDefUnsupported => write!(f, "Graph definition is unsupported"),
            GraphError::ReprUnsupported => write!(f, "Static representation is unsupported"),
            GraphError::MissingChild => write!(f, "Missing node in graph"),
            GraphError::ContextError => write!(f, "Graph context error"),
            GraphError::InvalidLeaf => write!(f, "Invalid leaf value"),
        }
    }
}

impl std::error::Error for GraphError {}

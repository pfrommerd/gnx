mod callable;
mod context;
mod impls;
mod path;
mod source;
mod views;
mod visitors;

pub mod filters;
pub mod util;

pub use context::*;
pub use filters::*;
pub use impls::*;
pub use path::*;
pub use source::*;
pub use views::*;
pub use visitors::*;

pub use gnx_derive::{Graph, Leaf};

pub use callable::Callable;

use std::{borrow::Borrow, error::Error};

pub trait Builder<L: Leaf>: Clone {
    type Graph: Graph + Clone + 'static;
    type Owned: Builder<L, Graph = Self::Graph> + 'static;

    fn build<S: GraphSource<(), L>>(
        self,
        source: S,
        ctx: &mut GraphContext,
    ) -> Result<Self::Graph, S::Error>;

    fn to_owned_builder(&self, ctx: &mut GraphContext) -> Self::Owned;
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

    fn visit<'g, L: Leaf, F: Filter<L>, V: GraphVisitor<'g, Self, L>>(
        &'g self, filter: F, visitor: V
    ) -> V::Output;

    fn mut_visit<'g, L: Leaf, F: Filter<L>, V: GraphMutVisitor<'g, Self, L>>(
        &'g mut self, filter: F, visitor: V
    ) -> V::Output;

    fn into_visit<'g, L: Leaf, F: Filter<L>, C: GraphConsumer<Self, L>>(
        self, filter: F, consumer: C
    ) -> C::Output;

    fn to_owned_graph(&self, ctx: &mut GraphContext) -> Self::Owned {
        let builder = self.builder(IgnoreAll::<()>::filter());
        builder.build(NullSource, ctx).unwrap()
    }
}

pub trait Node: Graph {
    fn visit_children<'g, L, F, V>(&'g self, filter: F, visitor: V) -> V::Output
    where
        L: Leaf,
        F: Filter<L>,
        V: ChildrenVisitor<'g, Self, L>;
    fn mut_visit_children<'g, L, F, V>(&'g mut self, filter: F, visitor: V) -> V::Output
    where
        L: Leaf,
        F: Filter<L>,
        V: ChildrenMutVisitor<'g, Self, L>;
    fn into_visit_children<L, F, C>(self, filter: F, consumer: C) -> C::Output
    where
        L: Leaf,
        F: Filter<L>,
        C: ChildrenConsumer<Self, L>;
}

pub trait TypedGraph<L: Leaf>: Graph {
    fn as_source<'g>(&'g self) -> AsSource<'g, Self, Of<L>> {
        AsSource::new(self, Of::<L>::filter())
    }
}

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

#[derive(Debug)]
pub enum GraphError {
    GraphDefUnsupported,
    ReprUnsupported,
    InvalidLeaf,
    ContextError,
    MissingChild,
    MissingLeaf,
    MissingNode,
    Other,
}

impl GraphError {
    fn is_missing(&self) -> bool {
        use GraphError::*;
        match self {
            MissingChild => true,
            MissingLeaf => true,
            MissingNode => true,
            _ => false,
        }
    }
}

impl std::fmt::Display for GraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GraphError::GraphDefUnsupported => write!(f, "Graph definition is unsupported"),
            GraphError::ReprUnsupported => write!(f, "Static representation is unsupported"),
            GraphError::ContextError => write!(f, "Graph context error"),
            GraphError::InvalidLeaf => write!(f, "Invalid leaf value"),
            GraphError::MissingChild => write!(f, "Missing child of node"),
            GraphError::MissingNode => write!(f, "Not a node!"),
            GraphError::MissingLeaf => write!(f, "Not a leaf!"),
            GraphError::Other => write!(f, "Other"),
        }
    }
}

impl std::error::Error for GraphError {}

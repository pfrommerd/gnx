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

use crate::util::Error;
pub use gnx_derive::{Graph, Leaf};

use std::borrow::Borrow;

pub trait Builder<L: Leaf>: Clone {
    type Graph: Graph + Clone + 'static;

    fn build<S: GraphSource<(), L>>(
        self,
        source: S,
        ctx: &mut GraphContext,
    ) -> Result<Self::Graph, S::Error>;
}

#[rustfmt::skip]
pub trait Graph: Clone {
    // The owned version of this graph
    type Owned: Graph + Clone + 'static;

    fn replace<'g, L: Leaf, F: Filter<L>, S: GraphSource<L::Ref<'g>, L>>(
        &'g self, replace: F, source: S, ctx: &mut GraphContext
    ) -> Result<Self::Owned, S::Error>;

    type Builder<L: Leaf>: Builder<L, Graph = Self::Owned> + 'static;
    fn builder<L: Leaf, F: Filter<L>>(
        &self, replace: F, ctx: &mut GraphContext
    ) -> Result<Self::Builder<L>, GraphError>;

    fn visit<'g, L: Leaf, F: Filter<L>, V: GraphVisitor<'g, Self, L>>(
        &'g self, filter: F, visitor: V
    ) -> V::Output;
    fn visit_into<'g, L: Leaf, F: Filter<L>, C: GraphConsumer<Self, L>>(
        self, filter: F, consumer: C
    ) -> C::Output;
}

pub trait Node: Graph {
    fn visit_children<'g, L, F, V>(&'g self, filter: F, visitor: V) -> V::Output
    where
        L: Leaf,
        F: Filter<L>,
        V: ChildrenVisitor<'g, Self, L>;
    fn visit_into_children<L, F, C>(self, filter: F, consumer: C) -> C::Output
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

pub trait Leaf: TypedGraph<Self, Owned = Self> + Clone + 'static {
    type Ref<'l>: Copy
    where
        Self: 'l;
    fn as_ref<'l>(&'l self) -> Self::Ref<'l>;
    fn clone_ref(v: Self::Ref<'_>) -> Self;
    fn try_from_value<V>(g: V) -> Result<Self, V>;
    fn try_from_ref<'v, V>(graph: &'v V) -> Result<Self::Ref<'v>, &'v V>;

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
    Custom(String),
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
            GraphError::Custom(v) => write!(f, "{}", v),
            GraphError::Other => write!(f, "Other"),
        }
    }
}
impl std::error::Error for GraphError {}

impl Error for GraphError {
    fn custom<T: std::fmt::Display>(msg: T) -> Self {
        GraphError::Custom(msg.to_string())
    }
}

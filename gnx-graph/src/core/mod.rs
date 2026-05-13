pub use gnx_derive::{Graph, Leaf};
use std::fmt::Display;

mod path;
mod source;
mod context;
mod visitors;
mod views;
mod filters;

pub use path::*;
pub use source::*;
pub use context::*;
pub use visitors::*;
pub use views::*;
pub use filters::*;

pub trait Error : Sized + std::error::Error {
    fn custom<T: Display>(msg: T) -> Self;


    fn invalid_type(unexp_value: Option<impl Display>, unexp_type: impl Display, expected: impl Display) -> Self {
        let _ = unexp_value;
        Self::custom(format!("Invalid type, got {unexp_type}, expected {expected}"))
    }
    fn invalid_leaf() -> Self {
        Self::custom("Invalid leaf")
    }
    fn invalid_id(id: GraphId) -> Self {
        Self::custom(format!("Invalid ID: {id}"))
    }

    fn missing_leaf() -> Self { Self::custom("Missing leaf") }
    fn missing_static_leaf() -> Self { Self::custom("Missing static leaf") }
    fn missing_node() -> Self { Self::custom("Missing node") }
    fn missing_child(key: Key) -> Self { Self::custom(format!("Missing child {key}")) }

    fn expected_leaf() -> Self { Self::custom("Expected leaf") }
    fn expected_static_leaf() -> Self { Self::custom("Expected static leaf") }
    fn expected_node() -> Self { Self::custom("Expected node") }
}

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
    fn builder<L: Leaf, F: Filter<L>, E: Error>(
        &self, replace: F, ctx: &mut GraphContext
    ) -> Result<Self::Builder<L>, E>;

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

pub trait OwnedGraph: Graph<Owned=Self> + Clone + 'static {}
impl<T: Graph<Owned=Self> + Clone + 'static> OwnedGraph for T {}

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
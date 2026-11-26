pub use gnx_base::{Error, GraphContext, GraphId, Key, KeyRef, Path};
pub use gnx_derive::{Graph, Leaf, impl_leaf};

mod filters;
mod source;
mod views;
mod visitors;

pub use filters::*;
pub use source::*;
pub use views::*;
pub use visitors::*;

pub trait Builder<L: Leaf>: Clone {
    type Graph: Graph + Clone + 'static;

    /// Rebuild the graph, requesting selected leaves in the same order as a
    /// depth-first visit of the original graph. Like the builder, the visit
    /// must descend into shared subgraphs only on their first occurrence.
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

    // Replace nodes in this graph according to the given filter and source.
    // This is more efficient than creating a builder() and then calling build().
    fn replace<'g, L: Leaf, F: Filter<L>, S: GraphSource<L::Ref<'g>, L>>(
        &'g self, replace: F, source: S, ctx: &mut GraphContext
    ) -> Result<Self::Owned, S::Error>;

    // Duplicate this graph by calling replace with a Nothing filter and a NullSource.
    fn duplicate(&self) -> Self::Owned {
        self.replace(Nothing::<()>::filter(), NullSource, &mut GraphContext::new()).unwrap()
    }

    // A 'builder' stores the graph structure information
    // and can be used to construct the owned graph from a source.
    type Builder<L: Leaf>: Builder<L, Graph = Self::Owned> + 'static;

    fn builder<L: Leaf, F: Filter<L>, E: Error>(
        &self, replace: F, ctx: &mut GraphContext
    ) -> Result<Self::Builder<L>, E>;

    // Visitor functions for visiting the graph and its children.
    fn visit<'g, L: Leaf, F: Filter<L>, V: GraphVisitor<'g, Self, L, F::Annotation<'g>>>(
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
        V: ChildrenVisitor<'g, Self, L, F::Annotation<'g>>;
    fn visit_into_children<L, F, C>(self, filter: F, consumer: C) -> C::Output
    where
        L: Leaf,
        F: Filter<L>,
        C: ChildrenConsumer<Self, L>;
}

// A typed view of a graph.
// All graph types get a blanket implementation for all leaf types.
pub trait TypedGraph<L: Leaf>: Graph {
    fn as_source<'g>(&'g self) -> AsSource<'g, Self, Of<L>> {
        AsSource::new(self, Of::<L>::filter())
    }
}

impl<L: Leaf, G: Graph> TypedGraph<L> for G {}

pub trait OwnedGraph: Graph<Owned = Self> + 'static {}
impl<G: Graph<Owned = Self> + 'static> OwnedGraph for G {}

pub trait Leaf: OwnedGraph {
    type Ref<'l>: Copy;
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

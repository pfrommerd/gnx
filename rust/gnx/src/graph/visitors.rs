use super::Filter;
use super::{Bound, Graph, GraphId, Key, KeyRef, Leaf, Node, View};

pub trait GraphVisitor<'g, G: Graph + ?Sized, L: Leaf> {
    type Output;
    fn visit_leaf(self, value: L::Ref<'g>) -> Self::Output;
    fn visit_static<S: Leaf>(self, value: S::Ref<'g>) -> Self::Output;
    fn visit_node<N: Node, F: Filter<L>>(self, node: View<'g, N, F>) -> Self::Output;
    fn visit_shared<S: Graph, F: Filter<L>>(
        self,
        id: GraphId,
        shared: View<'g, S, F>,
    ) -> Self::Output;
}
pub trait GraphConsumer<G: Graph + ?Sized, L: Leaf> {
    type Output;
    fn consume_leaf(self, value: L) -> Self::Output;
    fn consume_static<S: Leaf>(self, value: S) -> Self::Output;
    fn consume_node<N: Node, F: Filter<L>>(self, node: Bound<N, F>) -> Self::Output;
    // Once we hit a "Shared," we may have to switch to
    // (1) to visiting the inner graph definition,
    // (2) cloning and continue mapping
    // (3) stop e.g. if we have seen this GraphId before
    fn consume_shared<S: Graph, F: Filter<L>>(
        self,
        id: GraphId,
        shared: View<'_, S, F>,
    ) -> Self::Output;
}

pub trait ChildrenVisitor<'g, G: Node + ?Sized, L: Leaf> {
    type Output;
    fn visit_child<C: Graph, F: Filter<L>>(
        &mut self,
        key: KeyRef<'g>,
        child: View<'g, C, F>,
    ) -> &mut Self;
    fn finish(self) -> Self::Output;
}
pub trait ChildrenConsumer<G: Node + ?Sized, L: Leaf> {
    type Output;
    fn consume_child<C: Graph, F: Filter<L>>(&mut self, key: Key, child: Bound<C, F>) -> &mut Self;
    fn finish(self) -> Self::Output;
}

pub struct GenericVisitor<'g, G: Graph, L: Leaf, V: GraphVisitor<'g, G, L>>(
    pub V,
    std::marker::PhantomData<(&'g G, L)>,
);
pub struct GenericConsumer<G: Graph, L: Leaf, V: GraphConsumer<G, L>>(
    pub V,
    std::marker::PhantomData<(G, L)>,
);

#[rustfmt::skip]
impl<'g, G, L, V> From<V> for GenericVisitor<'g, G, L, V>
        where G: Graph, L: Leaf, V: GraphVisitor<'g, G, L> {
    fn from(visitor: V) -> Self {
        Self(visitor, std::marker::PhantomData)
    }
}
#[rustfmt::skip]
impl<'g, G, L, V> From<V> for GenericConsumer<G, L, V>
        where G: Graph, L: Leaf, V: GraphConsumer<G, L> {
    fn from(visitor: V) -> Self {
        Self(visitor, std::marker::PhantomData)
    }
}

#[rustfmt::skip]
impl<'g, G, OG, L, V> GraphVisitor<'g, G, L> for GenericVisitor<'g, OG, L, V>
    where G: Graph, OG: Graph, L: Leaf, V: GraphVisitor<'g, OG, L>,
{
    type Output = V::Output;
    fn visit_leaf(self, value: L::Ref<'g>) -> Self::Output {
        self.0.visit_leaf(value)
    }
    fn visit_static<S: Leaf>(self, value: S::Ref<'g>)
        -> Self::Output { self.0.visit_static::<S>(value) }
    fn visit_node<N: Node, F: Filter<L>>(self, node: View<'g, N, F>)
        -> Self::Output { self.0.visit_node(node) }
    fn visit_shared<S: Graph, F: Filter<L>>(
        self, id: GraphId, shared: View<'g, S, F>,
    ) -> Self::Output { self.0.visit_shared(id, shared) }
}
#[rustfmt::skip]
impl<G, OG, L, V> GraphConsumer<G, L> for GenericConsumer<OG, L, V>
    where G: Graph, OG: Graph, L: Leaf, V: GraphConsumer<OG, L>,
{
    type Output = V::Output;
    fn consume_leaf(self, value: L)
        -> Self::Output { self.0.consume_leaf(value) }
    fn consume_static<S: Leaf>(self, value: S)
        -> Self::Output { self.0.consume_static(value) }
    fn consume_node<N: Node, F: Filter<L>>(self, node: Bound<N, F>)
        -> Self::Output { self.0.consume_node(node) }
    fn consume_shared<S: Graph, F: Filter<L>>(
        self, id: GraphId, shared: View<'_, S, F>,
    ) -> Self::Output { self.0.consume_shared(id, shared) }
}

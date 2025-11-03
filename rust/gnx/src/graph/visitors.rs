use super::Filter;
use super::{Bound, Graph, GraphId, Key, KeyRef, Leaf, Node, View, ViewMut};

pub trait GraphVisitor<L: Leaf> {
    type Output;
    // None if the node is logically a leaf,
    // but it has no value under the current View
    fn visit_leaf(self, value: Option<L::Ref<'_>>) -> Self::Output;
    fn visit_node<N: Node, F: Filter<L>>(self, node: View<'_, N, F>) -> Self::Output;
    fn visit_shared<S: Graph, F: Filter<L>>(
        self,
        id: GraphId,
        shared: View<'_, S, F>,
    ) -> Self::Output;
}
pub trait GraphMutVisitor<L: Leaf> {
    type Output;
    // None if the node is logically a leaf,
    // but it has no value under the current View
    fn visit_leaf_mut(self, value: Option<L::RefMut<'_>>) -> Self::Output;
    fn visit_node_mut<N: Node, F: Filter<L>>(self, node: ViewMut<'_, N, F>) -> Self::Output;
    fn visit_shared_mut<S: Graph, F: Filter<L>>(
        self,
        id: GraphId,
        shared: View<'_, S, F>,
    ) -> Self::Output;
}
pub trait GraphConsumer<L: Leaf> {
    type Output;
    fn consume_leaf(self, value: Option<L>) -> Self::Output;
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

pub trait ChildrenVisitor<L: Leaf> {
    type Output;
    fn visit_child<G: Graph, F: Filter<L>>(
        &mut self,
        key: KeyRef<'_>,
        child: View<'_, G, F>,
    ) -> &mut Self;
    fn finish(self) -> Self::Output;
}
pub trait ChildrenMutVisitor<L: Leaf> {
    type Output;
    fn visit_child_mut<G: Graph, F: Filter<L>>(
        &mut self,
        key: KeyRef<'_>,
        child: ViewMut<'_, G, F>,
    ) -> &mut Self;
    fn finish(self) -> Self::Output;
}
pub trait ChildrenConsumer<L: Leaf> {
    type Output;
    fn consume_child<G: Graph, F: Filter<L>>(&mut self, key: Key, child: Bound<G, F>) -> &mut Self;
    fn finish(self) -> Self::Output;
}

use super::Filter;
use super::{Bound, Graph, GraphId, Key, KeyRef, Leaf, Node, View, ViewMut};

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

// Note that a MutVisitor does *not* get a builder,
// as a MutVisitor may potentially mutate the structure of the graph,
// meaning we cannot pass a builder that borrows from the same structure.
pub trait GraphMutVisitor<'g, G: Graph + ?Sized, L: Leaf> {
    type Output;
    // None if the node is logically a leaf,
    // but it has no value under the current View
    fn visit_leaf_mut(self, value: L::RefMut<'g>) -> Self::Output;
    fn visit_static_mut<S: Leaf>(self, value: S::RefMut<'g>) -> Self::Output;
    fn visit_node_mut<N: Node, F: Filter<L>>(self, node: ViewMut<'g, N, F>) -> Self::Output;
    fn visit_shared_mut<S: Graph, F: Filter<L>>(
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
    // Consume a mutable reference (i.e. *not* shared but not owned)
    fn consume_mut<S: Graph, F: Filter<L>>(self, view: ViewMut<'_, S, F>) -> Self::Output;
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
pub trait ChildrenMutVisitor<'g, G: Node + ?Sized, L: Leaf> {
    type Output;
    fn visit_child_mut<C: Graph, F: Filter<L>>(
        &mut self,
        key: KeyRef<'g>,
        child: ViewMut<'g, C, F>,
    ) -> &mut Self;
    fn finish(self) -> Self::Output;
}
pub trait ChildrenConsumer<G: Node + ?Sized, L: Leaf> {
    type Output;
    fn consume_child<C: Graph, F: Filter<L>>(&mut self, key: Key, child: Bound<C, F>) -> &mut Self;
    fn finish(self) -> Self::Output;
}

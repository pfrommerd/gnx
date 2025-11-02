use super::GraphFilter;
use super::{Graph, GraphId, Key, KeyRef, Node};

pub trait GraphVisitor<L, V: GraphFilter<L>> {
    type Output;
    // None if the node is logically a leaf,
    // but it has no value under the current View
    fn visit_leaf(self, value: Option<V::Ref<'_>>) -> Self::Output;
    fn visit_node<N: Node>(self, node: View<'_, N, V>) -> Self::Output;
    fn visit_shared<S: Graph>(self, id: GraphId, shared: View<'_, S, V>) -> Self::Output;
}
pub trait GraphMutVisitor<L, V: GraphFilter<L>> {
    type Output;
    // None if the node is logically a leaf,
    // but it has no value under the current View
    fn visit_leaf_mut(self, value: Option<V::RefMut<'_>>) -> Self::Output;
    fn visit_node_mut<N: Node>(self, node: ViewMut<'_, N, V>) -> Self::Output;
    fn visit_shared_mut<S: Graph>(self, id: GraphId, shared: View<'_, S, V>) -> Self::Output;
}
pub trait GraphConsumer<L, V: GraphFilter<L>> {
    type Output;
    fn consume_leaf(self, value: Option<L>) -> Self::Output;
    fn consume_node<N: Node>(self, node: Bound<N, V>) -> Self::Output;
    // Once we hit a "Shared," we may have to switch to
    // (1) to visiting the inner graph definition,
    // (2) cloning and continue mapping
    // (3) stop e.g. if we have seen this GraphId before
    fn consume_shared<S: Graph>(self, id: GraphId, shared: View<'_, S, V>) -> Self::Output;
}

pub trait ChildrenVisitor<L, V: GraphFilter<L>> {
    type Output;
    fn visit_child<G: Graph>(&mut self, key: KeyRef<'_>, child: View<'_, G, V>) -> &mut Self;
    fn finish(self) -> Self::Output;
}
pub trait ChildrenMutVisitor<L, V: GraphFilter<L>> {
    type Output;
    fn visit_child_mut<G: Graph>(&mut self, key: KeyRef<'_>, child: ViewMut<'_, G, V>)
    -> &mut Self;
    fn finish(self) -> Self::Output;
}
pub trait ChildrenConsumer<L, V: GraphFilter<L>> {
    type Output;
    fn consume_child<G: Graph>(&mut self, key: Key, child: Bound<G, V>) -> &mut Self;
    fn finish(self) -> Self::Output;
}

pub struct View<'g, G: Graph, V> {
    pub graph: &'g G,
    pub viewer: V,
}
pub struct ViewMut<'g, G: Graph, V> {
    pub graph: &'g mut G,
    pub viewer: V,
}
pub struct Bound<G: Graph, V> {
    pub graph: G,
    pub viewer: V,
}

impl<'g, G: Graph, V> View<'g, G, V> {
    pub fn new(graph: &'g G, viewer: V) -> View<'g, G, V> {
        View { graph, viewer }
    }
}
impl<'g, G: Graph, V> ViewMut<'g, G, V> {
    pub fn new(graph: &'g mut G, viewer: V) -> Self {
        ViewMut { graph, viewer }
    }
}
impl<G: Graph, V> Bound<G, V> {
    pub fn new(graph: G, viewer: V) -> Bound<G, V> {
        Bound { graph, viewer }
    }
}

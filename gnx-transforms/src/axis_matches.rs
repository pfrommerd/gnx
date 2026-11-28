//! Collect transform matches in graph-builder order, without cloning matched leaves.
use gnx_graph::{
    ChildrenVisitor, Filter, Graph, GraphId, GraphVisitor, Key, KeyRef, Leaf, Node, View,
};
use std::{any::TypeId, collections::HashSet, marker::PhantomData};

pub(crate) fn axes<L: Leaf, F: for<'a> Filter<Leaf = L, Annotation<'a> = usize>>(
    graph: &impl Graph,
    filter: &F,
) -> Vec<(Vec<Key>, usize)> {
    collect::<Axes, _, _>(graph, filter)
}
pub(crate) fn locations<L: Leaf, F: Filter<Leaf = L>>(
    graph: &impl Graph,
    filter: &F,
) -> Vec<(Vec<Key>, ())> {
    collect::<Locations, _, _>(graph, filter)
}

trait Record<'g, L: Leaf, A> {
    type Item;
    fn matched<F: Filter<Leaf = L, Annotation<'g> = A>>(leaf: L::Ref<'g>, filter: F) -> Self::Item;
}
struct Axes;
impl<'g, L: Leaf> Record<'g, L, usize> for Axes {
    type Item = usize;
    fn matched<F: Filter<Leaf = L, Annotation<'g> = usize>>(leaf: L::Ref<'g>, filter: F) -> usize {
        filter.annotate(leaf)
    }
}
struct Locations;
impl<'g, L: Leaf, A> Record<'g, L, A> for Locations {
    type Item = ();
    fn matched<F: Filter<Leaf = L, Annotation<'g> = A>>(_: L::Ref<'g>, _: F) {}
}

fn collect<'g, R, L: Leaf, F: Filter<Leaf = L>>(
    graph: &'g impl Graph,
    filter: &F,
) -> Vec<(Vec<Key>, R::Item)>
where
    R: Record<'g, L, F::Annotation<'g>>,
{
    let mut state = Matches::<R, _> {
        items: Vec::new(),
        path: Vec::new(),
        seen: HashSet::new(),
        record: PhantomData,
    };
    graph.visit(filter, &mut state);
    state.items
}
struct Matches<R, T> {
    items: Vec<(Vec<Key>, T)>,
    path: Vec<Key>,
    seen: HashSet<(TypeId, GraphId)>,
    record: PhantomData<R>,
}
impl<'g, G: Graph, L: Leaf, A, R: Record<'g, L, A>> GraphVisitor<'g, G, L, A>
    for &mut Matches<R, R::Item>
{
    type Output = ();
    fn visit_leaf<F: Filter<Leaf = L, Annotation<'g> = A>>(self, value: L::Ref<'g>, filter: F) {
        self.items
            .push((self.path.clone(), R::matched(value, filter)));
    }
    fn visit_static<S: Leaf>(self, _: S::Ref<'g>) {}
    fn visit_node<N: Node, F: Filter<Leaf = L, Annotation<'g> = A>>(self, node: View<'g, N, F>) {
        node.visit_children(self);
    }
    fn visit_shared<S: Graph, F: Filter<Leaf = L, Annotation<'g> = A>>(
        self,
        id: GraphId,
        shared: View<'g, S, F>,
    ) {
        if self.seen.insert((TypeId::of::<G::Builder<L>>(), id)) {
            shared.visit(self);
        }
    }
}
impl<'g, N: Node, L: Leaf, A, R: Record<'g, L, A>> ChildrenVisitor<'g, N, L, A>
    for &mut Matches<R, R::Item>
{
    type Output = ();
    fn visit_child<C: Graph, F: Filter<Leaf = L, Annotation<'g> = A>>(
        &mut self,
        key: KeyRef<'g>,
        child: View<'g, C, F>,
    ) -> &mut Self {
        self.path.push(key.to_value());
        child.visit(&mut **self);
        self.path.pop();
        self
    }
    fn finish(self) {}
}

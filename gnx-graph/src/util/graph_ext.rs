use std::any::TypeId;
use std::collections::HashSet;

use crate::{ChildrenVisitor, Filter, Graph, GraphId, GraphVisitor, Key, KeyRef, Leaf, Node, View};

use super::dag::*;

#[rustfmt::skip]
pub trait GraphExt: Graph {
    fn to_dag<L: Leaf, F: Filter<L>, V, M: FnMut(L::Ref<'_>) -> V>(
        &self, filter: F, map: M
    ) -> Dag<V> {
        self.visit(filter, ToDag(map))
    }

    /// Collect selected leaves in builder reconstruction order.
    fn collect_leaves<L: Leaf, F: Filter<L>>(&self, filter: F) -> Vec<L> {
        let mut visitor = CollectLeaves { leaves: Vec::new(), seen: HashSet::new() };
        self.visit(filter, &mut visitor);
        visitor.leaves
    }

    /// Collect selected leaves with their key paths from the root, in the same
    /// order as [`GraphExt::collect_leaves`]. A shared subgraph's leaves are
    /// reported once, at the path of its first occurrence.
    fn collect_leaves_with_paths<L: Leaf, F: Filter<L>>(&self, filter: F) -> Vec<(Vec<Key>, L)> {
        let mut visitor = CollectPaths {
            leaves: Vec::new(),
            seen: HashSet::new(),
            path: Vec::new(),
        };
        self.visit(filter, &mut visitor);
        visitor.leaves
    }
}

impl<G: Graph> GraphExt for G {}

struct CollectPaths<L> {
    leaves: Vec<(Vec<Key>, L)>,
    seen: HashSet<(TypeId, GraphId)>,
    path: Vec<Key>,
}

impl<'g, G: Graph + ?Sized, L: Leaf> GraphVisitor<'g, G, L> for &mut CollectPaths<L> {
    type Output = ();

    fn visit_leaf(self, value: L::Ref<'g>) {
        self.leaves.push((self.path.clone(), L::clone_ref(value)));
    }

    fn visit_static<S: Leaf>(self, _: S::Ref<'g>) {}

    fn visit_node<N: Node, F: Filter<L>>(self, node: View<'g, N, F>) {
        node.visit_children(self);
    }

    fn visit_shared<S: Graph, F: Filter<L>>(self, id: GraphId, shared: View<'g, S, F>) {
        if self.seen.insert((TypeId::of::<G::Builder<L>>(), id)) {
            shared.visit(self);
        }
    }
}

impl<'g, N: Node, L: Leaf> ChildrenVisitor<'g, N, L> for &mut CollectPaths<L> {
    type Output = ();

    fn visit_child<C: Graph, F: Filter<L>>(
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

struct CollectLeaves<L> {
    leaves: Vec<L>,
    seen: HashSet<(TypeId, GraphId)>,
}

impl<'g, G: Graph + ?Sized, L: Leaf> GraphVisitor<'g, G, L> for &mut CollectLeaves<L> {
    type Output = ();

    fn visit_leaf(self, value: L::Ref<'g>) {
        self.leaves.push(L::clone_ref(value));
    }

    fn visit_static<S: Leaf>(self, _: S::Ref<'g>) {}

    fn visit_node<N: Node, F: Filter<L>>(self, node: View<'g, N, F>) {
        node.visit_children(self);
    }

    fn visit_shared<S: Graph, F: Filter<L>>(self, id: GraphId, shared: View<'g, S, F>) {
        // GraphContext tracks sharing by ID and builder type.
        if self.seen.insert((TypeId::of::<G::Builder<L>>(), id)) {
            shared.visit(self);
        }
    }
}

impl<'g, N: Node, L: Leaf> ChildrenVisitor<'g, N, L> for &mut CollectLeaves<L> {
    type Output = ();

    fn visit_child<C: Graph, F: Filter<L>>(
        &mut self,
        _: KeyRef<'g>,
        child: View<'g, C, F>,
    ) -> &mut Self {
        child.visit(&mut **self);
        self
    }

    fn finish(self) {}
}

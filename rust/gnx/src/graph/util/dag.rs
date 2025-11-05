#[rustfmt::skip]
use crate::graph::{
    ChildrenVisitor, Filter,
    GraphVisitor, Key, KeyRef,
    Graph, GraphId, Leaf, Node,
    View,
};

use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone)]
pub enum Dag<I> {
    Leaf(Option<I>),
    Node(HashMap<Key, Arc<Dag<I>>>),
}

// TODO: Implement Graph for Dag<I>

// The conversion visitor
pub struct ToDag<M>(pub M);

impl<'g, G: Graph + ?Sized, L: Leaf, I, M> GraphVisitor<'g, G, L> for ToDag<M>
where
    M: FnMut(L::Ref<'_>) -> I,
{
    type Output = Dag<I>;

    fn visit_leaf(mut self, value: L::Ref<'_>) -> Self::Output {
        Dag::Leaf(Some((self.0)(value)))
    }
    fn visit_static<S: Leaf>(self, _: S::Ref<'_>) -> Self::Output {
        Dag::Leaf(None)
    }
    fn visit_node<N: Node, F: Filter<L>>(self, node: View<'_, N, F>) -> Self::Output {
        let mut ctx = HashMap::new();
        node.graph.visit_children(
            node.filter,
            ToDagChildren {
                map: self.0,
                children: HashMap::new(),
                ctx: &mut ctx,
            },
        )
    }
    fn visit_shared<S: Graph, F: Filter<L>>(
        self,
        _id: GraphId,
        shared: View<'_, S, F>,
    ) -> Self::Output {
        shared.visit(self)
    }
}

struct ToDagChildren<'ctx, M, I> {
    map: M,
    children: HashMap<Key, Arc<Dag<I>>>,
    ctx: &'ctx mut HashMap<GraphId, Arc<Dag<I>>>,
}

impl<'g, 'ctx, N: Node, L: Leaf, M, I> ChildrenVisitor<'g, N, L> for ToDagChildren<'ctx, M, I>
where
    M: FnMut(L::Ref<'_>) -> I,
{
    type Output = Dag<I>;

    fn visit_child<C: Graph, F: Filter<L>>(
        &mut self,
        key: KeyRef<'_>,
        child: View<'_, C, F>,
    ) -> &mut Self {
        self.children.insert(
            key.to_value(),
            child.visit(ToDagWithContext {
                map: &mut self.map,
                ctx: self.ctx,
            }),
        );
        self
    }
    fn finish(self) -> Self::Output {
        Dag::Node(self.children)
    }
}

struct ToDagWithContext<'ctx, M, I> {
    map: M,
    ctx: &'ctx mut HashMap<GraphId, Arc<Dag<I>>>,
}

impl<'g, 'ctx, G: Graph, L: Leaf, M, I> GraphVisitor<'g, G, L> for ToDagWithContext<'ctx, M, I>
where
    M: FnMut(L::Ref<'_>) -> I,
{
    type Output = Arc<Dag<I>>;

    fn visit_leaf(mut self, value: L::Ref<'_>) -> Self::Output {
        Arc::new(Dag::Leaf(Some((self.map)(value))))
    }
    fn visit_static<S: Leaf>(self, _: S::Ref<'_>) -> Self::Output {
        Arc::new(Dag::Leaf(None))
    }
    fn visit_node<N: Node, F: Filter<L>>(mut self, node: View<'_, N, F>) -> Self::Output {
        let dag = node.graph.visit_children(
            node.filter,
            ToDagChildren {
                map: self.map,
                children: HashMap::new(),
                ctx: &mut self.ctx,
            },
        );
        Arc::new(dag)
    }
    fn visit_shared<S: Graph, F: Filter<L>>(
        mut self,
        id: GraphId,
        shared: View<'_, S, F>,
    ) -> Self::Output {
        if let Some(dag) = self.ctx.get(&id) {
            return dag.clone();
        }
        let dag = shared.visit(ToDagWithContext {
            map: &mut self.map,
            ctx: self.ctx,
        });
        self.ctx.insert(id, dag.clone());
        dag
    }
}

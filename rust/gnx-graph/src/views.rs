use super::*;

pub struct View<'g, G: Graph, F> {
    pub graph: &'g G,
    pub filter: F,
}
impl<'g, G: Graph, F: Clone> Clone for View<'g, G, F> {
    fn clone(&self) -> Self {
        Self {
            graph: self.graph,
            filter: self.filter.clone(),
        }
    }
}

pub struct Bound<G: Graph, F> {
    pub graph: G,
    pub filter: F,
}

impl<'g, G: Graph, F> View<'g, G, F> {
    pub fn new(graph: &'g G, filter: F) -> Self {
        View { graph, filter }
    }
    pub fn as_source<L>(&self) -> AsSource<'g, G, F>
    where
        L: Leaf,
        F: Filter<L>,
    {
        AsSource::new(self.graph, self.filter.clone())
    }

    pub fn visit<V, L>(&self, visitor: V) -> V::Output
    where
        L: Leaf,
        V: GraphVisitor<'g, G, L>,
        F: Filter<L>,
    {
        self.graph.visit(&self.filter, visitor)
    }
    pub fn visit_children<L, V>(&self, visitor: V) -> V::Output
    where
        G: Node,
        L: Leaf,
        V: ChildrenVisitor<'g, G, L>,
        F: Filter<L>,
    {
        self.graph.visit_children(&self.filter, visitor)
    }
}

impl<G: Graph, F> Bound<G, F> {
    pub fn new(graph: G, filter: F) -> Self {
        Bound { graph, filter }
    }
    pub fn visit_into<V, L>(self, visitor: V) -> V::Output
    where
        L: Leaf,
        V: GraphConsumer<G, L>,
        F: Filter<L>,
    {
        self.graph.visit_into(self.filter, visitor)
    }
}
// TODO: The view types should implement
// Graph and TypedGraph

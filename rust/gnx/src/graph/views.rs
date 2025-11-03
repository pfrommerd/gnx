use super::*;

pub struct View<'g, G: Graph, F> {
    pub graph: &'g G,
    pub filter: F,
}
pub struct ViewMut<'g, G: Graph, F> {
    pub graph: &'g mut G,
    pub filter: F,
}
pub struct Bound<G: Graph, F> {
    pub graph: G,
    pub filter: F,
}

impl<'g, G: Graph, F> View<'g, G, F> {
    pub fn new(graph: &'g G, filter: F) -> Self {
        View { graph, filter }
    }
}
impl<'g, G: Graph, F> ViewMut<'g, G, F> {
    pub fn new(graph: &'g mut G, filter: F) -> Self {
        ViewMut { graph, filter }
    }
}
impl<G: Graph, F> Bound<G, F> {
    pub fn new(graph: G, filter: F) -> Self {
        Bound { graph, filter }
    }
}
// TODO: The view types should implement
// Graph and TypedGraph

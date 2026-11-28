use super::*;
use crate::util::{Dag, GraphExt};

/// Borrowed graph with a filter selecting visible subtrees.
#[derive(Clone)]
pub struct View<'g, G: Graph, F: Filter> {
    pub graph: &'g G,
    pub filter: F,
}

/// Owned counterpart of [`View`]. The complete graph remains available via `graph`.
#[derive(Clone)]
pub struct Filtered<G: Graph, F: Filter> {
    pub graph: G,
    pub filter: F,
}

impl<'g, G: Graph, F: Filter> View<'g, G, F> {
    pub fn new(graph: &'g G, filter: F) -> Self {
        Self { graph, filter }
    }
    pub fn as_source(&self) -> AsSource<'g, G, F> {
        AsSource::new(self.graph, self.filter.clone())
    }
    pub fn visit<V: GraphVisitor<'g, G, F::Leaf, F::Annotation<'g>>>(
        &self,
        visitor: V,
    ) -> V::Output {
        self.graph.visit(&self.filter, visitor)
    }
    pub fn visit_children<V>(&self, visitor: V) -> V::Output
    where
        G: Node,
        V: ChildrenVisitor<'g, G, F::Leaf, F::Annotation<'g>>,
    {
        self.graph.visit_children(&self.filter, visitor)
    }
}

impl<G: Graph, F: Filter> Filtered<G, F> {
    pub fn new(graph: G, filter: F) -> Self {
        Self { graph, filter }
    }
    pub fn as_view(&self) -> View<'_, G, F> {
        View::new(&self.graph, self.filter.clone())
    }
    pub fn visit_into<V: GraphConsumer<G, F::Leaf>>(self, visitor: V) -> V::Output {
        self.graph.visit_into(self.filter, visitor)
    }
}

// Resolve the stored filter to subtree roots. Route the caller's filter from
// the original root (not from each match), preserving paths and annotations.
#[derive(Clone)]
struct Within<M, F>(M, F);
impl<L: Leaf, M: Filter<Leaf = ()>, F: Filter<Leaf = L>> Filter for Within<M, F> {
    type Leaf = L;
    type Owned = Within<M::Owned, F::Owned>;
    type ChildFilter<'a>
        = Within<M::ChildFilter<'a>, F::ChildFilter<'a>>
    where
        Self: 'a;
    type Annotation<'g> = F::Annotation<'g>;
    fn owned(&self) -> Self::Owned {
        Within(self.0.owned(), self.1.owned())
    }
    fn child<'a>(&'a self, k: KeyRef<'a>) -> Self::ChildFilter<'a> {
        Within(self.0.child(k), self.1.child(k))
    }
    fn annotate<'g>(&self, leaf: L::Ref<'g>) -> Self::Annotation<'g> {
        self.1.annotate(leaf)
    }
    fn matches_ref<'g, G: Graph>(&self, g: &'g G) -> Result<L::Ref<'g>, &'g G> {
        self.0.matches_ref(g)?;
        self.1.matches_ref(g)
    }
    fn matches_value<G: Graph>(&self, g: G) -> Result<L, G> {
        if self.0.matches_ref(&g).is_ok() {
            self.1.matches_value(g)
        } else {
            Err(g)
        }
    }
}

#[doc(hidden)]
#[derive(Clone)]
pub struct FilteredBuilder<B, F>(B, F);
impl<T: Leaf, B: Builder<T>, F: Filter + 'static> Builder<T> for FilteredBuilder<B, F> {
    type Graph = Filtered<B::Graph, F>;
    fn build<S: GraphSource<(), T>>(
        self,
        source: S,
        ctx: &mut GraphContext,
    ) -> Result<Self::Graph, S::Error> {
        Ok(Filtered::new(self.0.build(source, ctx)?, self.1))
    }
}

macro_rules! view_graph {
    ($name:ident $(, $lt:lifetime)?) => {
        impl<$($lt,)? G: Graph, F: Filter> Graph for $name<$($lt,)? G, F> {
            type Owned = Filtered<G::Owned, F::Owned>;
            type Builder<T: Leaf> = FilteredBuilder<G::Builder<T>, F::Owned>;

            fn builder<T: Leaf, Q: Filter<Leaf = T>, E: Error>(
                &self, query: Q, ctx: &mut GraphContext,
            ) -> Result<Self::Builder<T>, E> {
                let mask: Dag<All> = self.graph.to_dag(&self.filter, |_| All);
                let builder = self.graph.builder(Within(mask, query), ctx)?;
                Ok(FilteredBuilder(builder, self.filter.owned()))
            }

            fn replace<'g, T: Leaf, Q: Filter<Leaf = T>, S: GraphSource<T::Ref<'g>, T>>(
                &'g self, query: Q, source: S, ctx: &mut GraphContext,
            ) -> Result<Self::Owned, S::Error> {
                let mask = self.graph.to_dag(&self.filter, |_| All);
                let graph = self.graph.replace(Within(mask, query), source, ctx)?;
                Ok(Filtered::new(graph, self.filter.owned()))
            }

            fn visit<'g, T: Leaf, Q: Filter<Leaf = T>, V: GraphVisitor<'g, Self, T, Q::Annotation<'g>>>(
                &'g self, query: Q, visitor: V,
            ) -> V::Output {
                let mask = self.graph.to_dag(&self.filter, |_| All);
                self.graph.visit(Within(mask, query), GenericVisitor::from(visitor))
            }

            fn visit_into<'g, T: Leaf, Q: Filter<Leaf = T>, C: GraphConsumer<Self, T>>(
                self, query: Q, consumer: C,
            ) -> C::Output {
                let mask = self.graph.to_dag(&self.filter, |_| All);
                self.graph.visit_into(Within(mask, query), GenericConsumer::from(consumer))
            }
        }
    }
}
view_graph!(View, 'v);
view_graph!(Filtered);

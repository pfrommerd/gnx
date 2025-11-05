use super::*;

use std::rc::Rc;
use std::sync::Arc;

#[rustfmt::skip]
macro_rules! impl_rc {
    ($W:ident) => {
        // Arc<T: Graph<L>> implements Graph<L>
        impl<T: Graph> Graph for $W<T> {
            type Owned = $W<T::Owned>;
            type Builder<'g, L: Leaf, F: Filter<L>> = ViewBuilder<'g, Self, F>
            where
                Self: 'g;
            type OwnedBuilder<L: Leaf> = $W<T::OwnedBuilder<L>>;

            fn builder<'g, L: Leaf, F: Filter<L>>(&'g self, filter: F)
                -> Self::Builder<'g, L, F> { ViewBuilder::new(self, filter) }

            fn visit<'g, L: Leaf, F: Filter<L>, V: GraphVisitor<'g, Self, L>>(
                &'g self, filter: F, visitor: V
            ) -> V::Output {
                let id = GraphId::from($W::as_ptr(self) as u64);
                visitor.visit_shared(id, View::new(self.as_ref(), filter))
            }
            fn mut_visit<'g, L: Leaf, F: Filter<L>, V: GraphMutVisitor<'g, Self, L>>(
                &'g mut self, filter: F, visitor: V
            ) -> V::Output {
                let id = GraphId::from($W::as_ptr(self) as u64);
                let s: &'g Self = self;
                let v: &'g T = s.as_ref();
                visitor.visit_shared_mut(id, View::new(v, filter))
            }
            fn into_visit<L: Leaf, F: Filter<L>, C: GraphConsumer<Self, L>>(
                self, filter: F, consumer: C
            ) -> C::Output {
                let id = GraphId::from($W::as_ptr(&self) as u64);
                consumer.consume_shared(id, View::new(self.as_ref(), filter))
            }
        }
        impl<L: Leaf, S: TypedGraph<L>> TypedGraph<L> for $W<S> {}
        impl<'g, L: Leaf, F: Filter<L>, S: Graph> Builder<L> for ViewBuilder<'g, $W<S>, F> {
            type Graph = $W<S::Owned>;
            type Owned = $W<S::OwnedBuilder<L>>;
            fn build<GS: GraphSource<(), L>>(
                self,
                source: GS,
                ctx: &mut GraphContext,
            ) -> Result<Self::Graph, GS::Error> {
                let id = GraphId::from($W::as_ptr(self.graph) as u64);
                ctx.build(id, |ctx| {
                    self.graph.as_ref().builder(self.filter).build(source, ctx)
                       .map($W::new)
                })
            }
            fn to_owned_builder(&self, ctx: &mut GraphContext) -> Self::Owned {
                let id = GraphId::from($W::as_ptr(self.graph) as u64);
                ctx.build(id, |ctx| -> Result<Self::Owned, GraphError> {
                    let builder = self.graph.as_ref().builder(&self.filter);
                    Ok($W::new(builder.to_owned_builder(ctx)))
                }).unwrap()
            }
        }
        impl<L: Leaf, B: Builder<L>> Builder<L> for $W<B> {
            type Graph = $W<B::Graph>;
            type Owned = $W<B::Owned>;
            fn build<S: GraphSource<(), L>>(
                self,
                source: S,
                ctx: &mut GraphContext,
            ) -> Result<Self::Graph, S::Error> {
                let id = GraphId::from($W::as_ptr(&self) as u64);
                ctx.build(id, |ctx| {
                    self.as_ref().clone().build(source, ctx)
                       .map($W::new)
                })
            }
            fn to_owned_builder(&self, ctx: &mut GraphContext) -> Self::Owned {
                let id = GraphId::from($W::as_ptr(&self) as u64);
                ctx.build(id, |ctx| -> Result<Self::Owned, GraphError> {
                    Ok($W::new(self.as_ref().to_owned_builder(ctx)))
                }).unwrap()
            }
        }
    };
}

impl_rc!(Rc);
impl_rc!(Arc);

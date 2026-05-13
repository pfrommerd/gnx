use super::*;

use std::rc::Rc;
use std::sync::Arc;

#[rustfmt::skip]
macro_rules! impl_rc {
    ($W:ident) => {
        // Arc<T: Graph<L>> implements Graph<L>
        impl<T: Graph> Graph for $W<T> {
            type Owned = $W<T::Owned>;
            type Builder<L: Leaf> = $W<T::Builder<L>>;

            fn replace<'g, L: Leaf, F: Filter<L>, S: GraphSource<L::Ref<'g>, L>>(
                &'g self, filter: F, source: S, ctx: &mut GraphContext,
            ) -> Result<Self::Owned, S::Error> {
                let id = GraphId::from($W::as_ptr(self) as u64);
                ctx.create(id, |ctx| {
                    self.as_ref().replace(filter, source, ctx)
                       .map($W::new)
                })
            }
            fn builder<L: Leaf, F: Filter<L>, E: Error>(
                &self, filter: F, ctx: &mut GraphContext
            ) -> Result<Self::Builder<L>, E> {
                let id = GraphId::from($W::as_ptr(self) as u64);
                ctx.create(id, |ctx| {
                    self.as_ref().builder(filter, ctx)
                       .map($W::new)
                })
            }

            fn visit<'g, L: Leaf, F: Filter<L>, V: GraphVisitor<'g, Self, L>>(
                &'g self, filter: F, visitor: V
            ) -> V::Output {
                let id = GraphId::from($W::as_ptr(self) as u64);
                visitor.visit_shared(id, View::new(self.as_ref(), filter))
            }
            fn visit_into<L: Leaf, F: Filter<L>, C: GraphConsumer<Self, L>>(
                self, filter: F, consumer: C
            ) -> C::Output {
                let id = GraphId::from($W::as_ptr(&self) as u64);
                consumer.consume_shared(id, View::new(self.as_ref(), filter))
            }
        }
        impl<L: Leaf, S: TypedGraph<L>> TypedGraph<L> for $W<S> {}
        impl<L: Leaf, B: Builder<L>> Builder<L> for $W<B> {
            type Graph = $W<B::Graph>;
            fn build<S: GraphSource<(), L>>(
                self,
                source: S,
                ctx: &mut GraphContext,
            ) -> Result<Self::Graph, S::Error> {
                let id = GraphId::from($W::as_ptr(&self) as u64);
                ctx.create(id, |ctx| {
                    $W::as_ref(&self).clone().build(source, ctx)
                       .map($W::new)
                })
            }
        }
    };
}

impl_rc!(Rc);
impl_rc!(Arc);

impl<T: Graph> Graph for &T {
    type Owned = Arc<T::Owned>;
    type Builder<L: Leaf> = Arc<T::Builder<L>>;

    fn replace<'g, L: Leaf, F: Filter<L>, S: GraphSource<L::Ref<'g>, L>>(
        &'g self,
        filter: F,
        source: S,
        ctx: &mut GraphContext,
    ) -> Result<Self::Owned, S::Error> {
        let id = GraphId::from(*self as *const T as u64);
        ctx.create(id, |ctx| {
            Ok(Arc::new(T::replace(self, filter, source, ctx)?))
        })
    }

    fn builder<L: Leaf, F: Filter<L>, E: Error>(
        &self,
        filter: F,
        ctx: &mut GraphContext,
    ) -> Result<Self::Builder<L>, E> {
        let id = GraphId::from(*self as *const T as u64);
        ctx.create(id, |ctx| Ok(Arc::new(T::builder(self, filter, ctx)?)))
    }

    fn visit<'g, L: Leaf, F: Filter<L>, V: GraphVisitor<'g, Self, L>>(
        &'g self,
        filter: F,
        visitor: V,
    ) -> V::Output {
        let id = GraphId::from(*self as *const T as u64);
        visitor.visit_shared(id, View::new(self, filter))
    }

    fn visit_into<L: Leaf, F: Filter<L>, C: GraphConsumer<Self, L>>(
        self,
        filter: F,
        consumer: C,
    ) -> C::Output {
        let id = GraphId::from(self as *const T as u64);
        consumer.consume_shared(id, View::new(self, filter))
    }
}

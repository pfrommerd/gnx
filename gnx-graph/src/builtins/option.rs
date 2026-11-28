//! A present value occupies the enclosing field's path. An absent value is static.
use super::*;

impl<T: Graph> Graph for Option<T> {
    type Owned = Option<T::Owned>;
    type Builder<L: Leaf> = Option<T::Builder<L>>;

    fn builder<L: Leaf, F: Filter<Leaf = L>, E: Error>(
        &self,
        filter: F,
        ctx: &mut GraphContext,
    ) -> Result<Self::Builder<L>, E> {
        self.as_ref()
            .map(|value| value.builder(filter, ctx))
            .transpose()
    }
    fn replace<'g, L: Leaf, F: Filter<Leaf = L>, S: GraphSource<L::Ref<'g>, L>>(
        &'g self,
        filter: F,
        source: S,
        ctx: &mut GraphContext,
    ) -> Result<Self::Owned, S::Error> {
        match self {
            Some(value) => value.replace(filter, source, ctx).map(Some),
            None => {
                source.empty_leaf()?;
                Ok(None)
            }
        }
    }
    fn visit<'g, L: Leaf, F: Filter<Leaf = L>, V: GraphVisitor<'g, Self, L, F::Annotation<'g>>>(
        &'g self,
        filter: F,
        visitor: V,
    ) -> V::Output {
        match self {
            Some(value) => value.visit(filter, GenericVisitor::from(visitor)),
            None => visitor.visit_static::<()>(&()),
        }
    }
    fn visit_into<L: Leaf, F: Filter<Leaf = L>, C: GraphConsumer<Self, L>>(
        self,
        filter: F,
        consumer: C,
    ) -> C::Output {
        match self {
            Some(value) => value.visit_into(filter, GenericConsumer::from(consumer)),
            None => consumer.consume_static(()),
        }
    }
}
impl<L: Leaf, B: Builder<L>> Builder<L> for Option<B> {
    type Graph = Option<B::Graph>;
    fn build<S: GraphSource<(), L>>(
        self,
        source: S,
        ctx: &mut GraphContext,
    ) -> Result<Self::Graph, S::Error> {
        match self {
            Some(builder) => builder.build(source, ctx).map(Some),
            None => {
                source.empty_leaf()?;
                Ok(None)
            }
        }
    }
}

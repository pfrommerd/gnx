use super::*;

use std::sync::Arc;

impl<T: Graph> Graph for &T {
    type Owned = Arc<T::Owned>;
    type Builder<'g, L: Leaf, F: Filter<L>>
        = Arc<T::Builder<'g, L, F>>
    where
        Self: 'g;
    type OwnedBuilder<L: Leaf> = Arc<T::OwnedBuilder<L>>;

    fn builder<'g, L: Leaf, F: Filter<L>>(&'g self, filter: F) -> Self::Builder<'g, L, F> {
        Arc::new(T::builder(self, filter))
    }

    fn visit<'g, L: Leaf, F: Filter<L>, V: GraphVisitor<'g, Self, L>>(
        &'g self,
        filter: F,
        visitor: V,
    ) -> V::Output {
        let id = GraphId::from(*self as *const T as u64);
        visitor.visit_shared(id, View::new(self, filter))
    }

    fn mut_visit<'g, L: Leaf, F: Filter<L>, V: GraphMutVisitor<'g, Self, L>>(
        &'g mut self,
        filter: F,
        visitor: V,
    ) -> V::Output {
        let id = GraphId::from(*self as *const T as u64);
        visitor.visit_shared_mut(id, View::new(self, filter))
    }

    fn into_visit<L: Leaf, F: Filter<L>, C: GraphConsumer<Self, L>>(
        self,
        filter: F,
        consumer: C,
    ) -> C::Output {
        let id = GraphId::from(self as *const T as u64);
        consumer.consume_shared(id, View::new(self, filter))
    }
}

impl<T: Graph> Graph for &mut T {
    type Owned = Arc<T::Owned>;
    type Builder<'g, L: Leaf, F: Filter<L>>
        = Arc<T::Builder<'g, L, F>>
    where
        Self: 'g;
    type OwnedBuilder<L: Leaf> = Arc<T::OwnedBuilder<L>>;

    fn builder<'g, L: Leaf, F: Filter<L>>(&'g self, filter: F) -> Self::Builder<'g, L, F> {
        Arc::new(T::builder(self, filter))
    }

    fn visit<'g, L: Leaf, F: Filter<L>, V: GraphVisitor<'g, Self, L>>(
        &'g self,
        filter: F,
        visitor: V,
    ) -> V::Output {
        let visitor: GenericVisitor<_, _, _> = visitor.into();
        T::visit::<L, F, GenericVisitor<Self, L, V>>(self, filter, visitor)
    }

    fn mut_visit<'g, L: Leaf, F: Filter<L>, V: GraphMutVisitor<'g, Self, L>>(
        &'g mut self,
        filter: F,
        visitor: V,
    ) -> V::Output {
        let visitor: GenericMutVisitor<_, _, _> = visitor.into();
        T::mut_visit::<L, F, GenericMutVisitor<Self, L, V>>(self, filter, visitor)
    }

    fn into_visit<L: Leaf, F: Filter<L>, C: GraphConsumer<Self, L>>(
        self,
        filter: F,
        consumer: C,
    ) -> C::Output {
        consumer.consume_mut(ViewMut::new(self, filter))
    }
}

pub struct GenericVisitor<'g, G: Graph, L: Leaf, V: GraphVisitor<'g, G, L>>(
    pub V,
    std::marker::PhantomData<(&'g G, L)>,
);
pub struct GenericMutVisitor<'g, G: Graph, L: Leaf, V: GraphMutVisitor<'g, G, L>>(
    pub V,
    std::marker::PhantomData<(&'g G, L)>,
);

#[rustfmt::skip]
impl<'g, G, L, V> From<V> for GenericVisitor<'g, G, L, V>
        where G: Graph, L: Leaf, V: GraphVisitor<'g, G, L> {
    fn from(visitor: V) -> Self {
        Self(visitor, std::marker::PhantomData)
    }
}
#[rustfmt::skip]
impl<'g, G, L, V> From<V> for GenericMutVisitor<'g, G, L, V>
        where G: Graph, L: Leaf, V: GraphMutVisitor<'g, G, L> {
    fn from(visitor: V) -> Self {
        Self(visitor, std::marker::PhantomData)
    }
}

#[rustfmt::skip]
impl<'g, G, OG, L, V> GraphVisitor<'g, G, L> for GenericVisitor<'g, OG, L, V>
    where G: Graph, OG: Graph, L: Leaf, V: GraphVisitor<'g, OG, L>,
{
    type Output = V::Output;
    fn visit_leaf(self, value: L::Ref<'g>) -> Self::Output {
        self.0.visit_leaf(value)
    }
    fn visit_static<S: Leaf>(self, value: S::Ref<'g>)
        -> Self::Output { self.0.visit_static::<S>(value) }
    fn visit_node<N: Node, F: Filter<L>>(self, node: View<'g, N, F>)
        -> Self::Output { self.0.visit_node(node) }
    fn visit_shared<S: Graph, F: Filter<L>>(
        self, id: GraphId, shared: View<'g, S, F>,
    ) -> Self::Output { self.0.visit_shared(id, shared) }
}
#[rustfmt::skip]
impl<'g, G, OG, L, V> GraphMutVisitor<'g, G, L> for GenericMutVisitor<'g, OG, L, V>
    where G: Graph, OG: Graph, L: Leaf, V: GraphMutVisitor<'g, OG, L>,
{
    type Output = V::Output;
    fn visit_leaf_mut(self, value: L::RefMut<'g>)
        -> Self::Output { self.0.visit_leaf_mut(value) }
    fn visit_static_mut<S: Leaf>(self, value: S::RefMut<'g>)
        -> Self::Output { self.0.visit_static_mut::<S>(value) }
    fn visit_node_mut<N: Node, F: Filter<L>>(self, node: ViewMut<'g, N, F>)
        -> Self::Output { self.0.visit_node_mut(node) }
    fn visit_shared_mut<S: Graph, F: Filter<L>>(
        self, id: GraphId, shared: View<'g, S, F>,
    ) -> Self::Output { self.0.visit_shared_mut(id, shared) }
}

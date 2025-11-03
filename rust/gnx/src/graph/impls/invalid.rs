use crate::graph::*;
use castaway::LifetimeFree;

#[derive(Clone, Copy)]
pub struct Invalid();
unsafe impl LifetimeFree for Invalid {}

impl<L: Leaf> TypedGraph<L> for Invalid {}

#[rustfmt::skip]
#[allow(unused)]
impl Graph for Invalid {
    type GraphDef<I: Leaf, R: StaticRepr> = Invalid;
    type Owned = Invalid;

    fn graph_def<I: Leaf, L: Leaf, F, M>(
        &self, filter: F, map: impl Into<M>, ctx: &mut GraphContext,
    ) -> Result<Self::GraphDef<I, F::StaticRepr>, GraphError>
            where F: Filter<L>, M: FnMut(L::Ref<'_>) -> I {
        panic!()
    }
    fn into_graph_def<I: Leaf, L: Leaf, F, M>(
        self, filter: F, map: impl Into<M>, ctx: &mut GraphContext,
    ) -> Result<Self::GraphDef<I, F::StaticRepr>, GraphError>
            where F: Filter<L>, M: FnMut(LeafCow<'_, L>) -> I {
        panic!()
    }

    fn visit<L, F, V>(&self, filter: F, visitor: impl Into<V>) -> V::Output
            where L: Leaf, F: Filter<L>, V: GraphVisitor<L> {
        panic!()
    }
    fn mut_visit<L, F, M>(&mut self, filter: F, visitor: impl Into<M>) -> M::Output
            where L: Leaf, F: Filter<L>, M: GraphMutVisitor<L> {
        panic!()
    }
    fn into_visit<L, F, C>(self, filter: F, consumer: impl Into<C>) -> C::Output
            where L: Leaf, F: Filter<L>, C: GraphConsumer<L> {
        panic!()
    }
}

#[rustfmt::skip]
#[allow(unused)]
impl Leaf for Invalid {
    type Ref<'l> = &'l Invalid where Self: 'l;
    type RefMut<'l> = &'l mut Invalid where Self: 'l;

    fn as_ref<'l>(&'l self) -> Self::Ref<'l> { panic!() }
    fn as_mut<'l>(&'l mut self) -> Self::RefMut<'l> { panic!() }
    fn clone_ref(r: Self::Ref<'_>) -> Self { panic!() }
    fn clone_mut(r: Self::RefMut<'_>) -> Self { panic!() }

    fn try_as_ref<'v, V>(value: &'v V) -> Result<Self::Ref<'v>, &'v V> { Err(value) }
    fn try_as_mut<'v, V>(value: &'v mut V) -> Result<Self::RefMut<'v>, &'v mut V> { Err(value) }
    fn try_from_value<V>(value: V) -> Result<Self, V> { Err(value) }
}

#[rustfmt::skip]
#[allow(unused)]
impl<I: Leaf> GraphDef<I> for Invalid {
    type Graph = Invalid;

    fn build<V: Value, S: GraphSource<I, V>>(
            &self, source: S, ctx: &mut GraphContext,
    ) -> Result<Self::Graph, S::Error> {
        panic!()
    }
    fn into_build<V: Value, S: GraphSource<I, V>>(
            self, source: S, ctx: &mut GraphContext,
    ) -> Result<Self::Graph, S::Error> {
        panic!()
    }
}

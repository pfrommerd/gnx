use crate::*;
use crate::util::{impl_lifetime_free, LifetimeFree};

use gnx_expr::trace::{
    TraceCellRef, TraceRef, Tracer, TracerCell, Traceable,
};
use gnx_expr::array::{Array, ArrayRef, ArrayInfo};

impl_lifetime_free!(TraceRef, TraceCellRef, Array, ArrayRef);
impl_lifetime_free!(for<T: Traceable> Tracer<T>);
impl_lifetime_free!(for<T: Traceable> TracerCell<T>);

// Basic leaf types.
impl_leaf!(TraceRef);
impl_leaf!(TraceCellRef);

// A boilerplate Graph implementation for a Leaf type.
impl<T: Traceable + LifetimeFree + 'static> Graph for Tracer<T> {
    type Owned = Self;
    type Builder<L: Leaf> = LeafBuilder<Self>;

    fn replace<'g, L: Leaf, F: Filter<L>, S: GraphSource<L::Ref<'g>, L>>(
        &'g self, filter: F, source: S, ctx: &mut GraphContext
    ) -> Result<Self::Owned, S::Error> {
        // Try to match the outer tracer.
        if let Ok(r) = filter.matches_ref(self) {
            return Ok(source.leaf(r)?.try_into_value().map_err(|_| S::Error::invalid_leaf())?);
        }
        // Try to match the inner trace ref.
        if let Ok(r) = filter.matches_ref(self.trace_ref()) {
            return Ok(source.leaf(r)?.try_into_value().map_err(|_| S::Error::invalid_leaf())?);
        }
        Err(S::Error::invalid_leaf())
    }
    fn builder<'g, L: Leaf, F: Filter<L>, E: Error>(
        &'g self, filter: F, _ctx: &mut GraphContext
    ) -> Result<Self::Builder<L>, E> {
        todo!()
    }
    fn visit<'g, L: Leaf, F: Filter<L>, V: GraphVisitor<'g, Self, L>>(
        &'g self, filter: F, visitor: V
    ) -> V::Output { todo!() }
    fn visit_into<L: Leaf, F: Filter<L>, C: GraphConsumer<Self, L>>(
        self, filter: F, consumer: C
    ) -> C::Output { todo!() }
}

impl<T: Traceable + LifetimeFree + 'static> Leaf for Tracer<T> {
    type Ref<'l> = &'l Self;
    fn as_ref<'l>(&'l self) -> Self::Ref<'l> { self }
    fn clone_ref(v: Self::Ref<'_>) -> Self { v.clone() }
    fn try_from_value<V>(g: V) -> Result<Self, V> {
        todo!()
    }
    fn try_from_ref<'v, V>(graph: &'v V) -> Result<Self::Ref<'v>, &'v V> {
        todo!()
    }
    fn try_into_value<V: 'static>(self) -> Result<V, Self> {
        todo!()
    }
}
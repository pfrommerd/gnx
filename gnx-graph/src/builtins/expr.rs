use crate::*;
use crate::util::{impl_lifetime_free, LifetimeFree};

use gnx_expr::trace::{
    TraceCellRef, TraceRef, Tracer, TracerCell, Traceable, Generic,
};
use gnx_expr::array::{Array, ArrayRef, ArrayInfo};
use gnx_expr::device::{Device, DeviceInfo};

impl_lifetime_free!(TraceRef, TraceCellRef, Array, ArrayRef, Device, ArrayInfo, DeviceInfo);
impl_lifetime_free!(for<T: Traceable> Tracer<T>);
impl_lifetime_free!(for<T: Traceable> TracerCell<T>);

// Basic leaf types.
impl_leaf!(TraceRef);
impl_leaf!(TraceCellRef);
impl_leaf!(ArrayInfo);
impl_leaf!(DeviceInfo);

// Note Tracer<> and TracerCell<> types are intentionally not implemented as leaves.
// TraceRef and TraceCellRef are only leaves so that we can filter all Traceable types.

impl Graph for Array {
    type Owned = Self;
    type Builder<L: Leaf> = LeafBuilder<Self>;

    fn replace<'g, L: Leaf, F: Filter<L>, S: GraphSource<L::Ref<'g>, L>>(
        &'g self, filter: F, source: S, _ctx: &mut GraphContext,
    ) -> Result<Self::Owned, S::Error> {
        let trace_ref = self.tracer().trace_ref();
        match filter_matches_ref_any!(filter, self, trace_ref) {
            Ok(r) => Ok(source.leaf(r)?.try_into_value().map_err(|_| S::Error::invalid_leaf())?),
            Err(_) => Ok(self.clone()),
        }
    }
    fn builder<'g, L: Leaf, F: Filter<L>, E: Error>(
        &'g self, filter: F, _ctx: &mut GraphContext,
    ) -> Result<Self::Builder<L>, E> {
        let trace_ref = self.tracer().trace_ref();
        match filter_matches_ref_any!(filter, self, trace_ref) {
            Ok(_) => Ok(LeafBuilder::Leaf),
            Err(_) => Ok(LeafBuilder::Static(self.clone())),
        }
    }
    fn visit<'g, L: Leaf, F: Filter<L>, V: GraphVisitor<'g, Self, L>>(
        &'g self, filter: F, visitor: V,
    ) -> V::Output {
        let trace_ref = self.tracer().trace_ref();
        match filter_matches_ref_any!(filter, self, trace_ref) {
            Ok(r) => visitor.visit_leaf(r),
            Err(_) => visitor.visit_static::<Self>(self)
        }
    }
    fn visit_into<'g, L: Leaf, F: Filter<L>, C: GraphConsumer<Self, L>>(
        self, filter: F, consumer: C,
    ) -> C::Output {
        let trace_ref = self.tracer().trace_ref();
        // Try to match each value in order. If the ref matches, convert the value
        // and call the conversion on the value type.
        if let Ok(_) = filter.matches_ref(&self) {
            let value = filter.matches_value(self).or(Err(())).expect("Filter matched ref but not value");
            return consumer.consume_leaf(value)
        }
        if let Ok(_) = filter.matches_ref(trace_ref) {
            let trace_ref = self.into_tracer().into_trace_ref();
            let value = filter.matches_value(trace_ref).or(Err(())).expect("Filter matched ref but not value");
            return consumer.consume_leaf(value)
        }
        // If we get here, the filter didn't match any of the refs, so we're static.
        // Consume as the wrapped Array type.
        consumer.consume_static::<Self>(self)
    }
}

impl Leaf for Array {
    type Ref<'l> = &'l Array;
    fn as_ref<'l>(&'l self) -> Self::Ref<'l> { self }
    fn clone_ref(v: Self::Ref<'_>) -> Self { v.clone() }
    fn try_from_value<V>(g: V) -> Result<Self, V> {
        try_specialize!(g, Self).or_else(|g| {
            if let Ok(r) = try_specialize!(&g, &TraceRef) {
                let r: &Tracer<Generic> = r.into();
                if r.try_cast::<Array>().is_ok() {
                    // We know that g is a Array, so we can do an unchecked cast to Array.
                    let r: Tracer<Generic> = try_specialize!(g, TraceRef).ok().unwrap().into();
                    return Ok(Array::from(r.unchecked_cast_into()))
                }
            }
            Err(g)
        })
    }
    fn try_from_ref<'v, V>(graph: &'v V) -> Result<Self::Ref<'v>, &'v V> {
        try_specialize!(graph, &Self).or_else(|graph| {
            match try_specialize!(graph, &TraceRef) {
                Ok(r) => {
                    let r: &Tracer<Generic> = r.into();
                    match r.try_cast::<Array>() {
                        Ok(r) => Ok(r.into()),
                        Err(_) => Err(graph),
                    }
                }
                Err(graph) => Err(graph),
            }
        })
    }
    fn try_into_value<V: 'static>(self) -> Result<V, Self> {
        try_specialize!(self, V)
    }
}
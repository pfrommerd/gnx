use crate::*;
use crate::util::impl_lifetime_free;

use gnx_expr::trace::{
    TraceCellRef, TraceRef, Tracer, TracerCell, Traceable, Generic,
};
use gnx_backend::{Device, DeviceInfo, Array, ArrayRef, ArrayInfo};

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
            Ok(r) => Ok(
                Self::try_from_value(source.leaf(r)?).map_err(|_| S::Error::invalid_leaf())?,
            ),
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
        cast!(g, Self).or_else(|g| {
            if let Ok(r) = cast!(&g, &TraceRef) {
                let r: &Tracer<Generic> = r.into();
                if r.try_cast_ref::<Array>().is_ok() {
                    // We know that g is a Array, so we can do an unchecked cast to Array.
                    let r: Tracer<Generic> = cast!(g, TraceRef).ok().unwrap().into();
                    // SAFETY: We know that it is safe to cast to Array since
                    // we just checked that the info matches.
                    return Ok(Array::from(unsafe { r.cast_unchecked() }))
                }
            }
            Err(g)
        })
    }
    fn try_from_ref<'v, V>(graph: &'v V) -> Result<Self::Ref<'v>, &'v V> {
        cast!(graph, &Self).or_else(|graph| {
            match cast!(graph, &TraceRef) {
                Ok(r) => {
                    let r: &Tracer<Generic> = r.into();
                    match r.try_cast_ref::<Array>() {
                        Ok(r) => Ok(r.into()),
                        Err(_) => Err(graph),
                    }
                }
                Err(graph) => Err(graph),
            }
        })
    }
    fn try_into_value<V: 'static>(self) -> Result<V, Self> {
        cast!(self, V)
    }
}

impl Graph for Device {
    type Owned = Self;
    type Builder<L: Leaf> = LeafBuilder<Self>;

    fn replace<'g, L: Leaf, F: Filter<L>, S: GraphSource<L::Ref<'g>, L>>(
        &'g self, filter: F, source: S, _ctx: &mut GraphContext,
    ) -> Result<Self::Owned, S::Error> {
        let trace_ref = self.tracer().trace_ref();
        match filter_matches_ref_any!(filter, self, trace_ref) {
            Ok(r) => Ok(
                Self::try_from_value(source.leaf(r)?).map_err(|_| S::Error::invalid_leaf())?,
            ),
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
        if let Ok(_) = filter.matches_ref(&self) {
            let value = filter.matches_value(self).or(Err(())).expect("Filter matched ref but not value");
            return consumer.consume_leaf(value)
        }
        if let Ok(_) = filter.matches_ref(trace_ref) {
            let trace_ref = self.into_tracer().into_trace_ref();
            let value = filter.matches_value(trace_ref).or(Err(())).expect("Filter matched ref but not value");
            return consumer.consume_leaf(value)
        }
        consumer.consume_static::<Self>(self)
    }
}

impl Leaf for Device {
    type Ref<'l> = &'l Device;
    fn as_ref<'l>(&'l self) -> Self::Ref<'l> { self }
    fn clone_ref(v: Self::Ref<'_>) -> Self { v.clone() }
    fn try_from_value<V>(g: V) -> Result<Self, V> {
        cast!(g, Self).or_else(|g| {
            if let Ok(r) = cast!(&g, &TraceRef) {
                let r: &Tracer<Generic> = r.into();
                if r.try_cast_ref::<Device>().is_ok() {
                    let r: Tracer<Generic> = cast!(g, TraceRef).ok().unwrap().into();
                    // SAFETY: We know that it is safe to cast to Device since
                    // we just checked that the info matches.
                    return Ok(Device::from(unsafe { r.cast_unchecked() }))
                }
            }
            Err(g)
        })
    }
    fn try_from_ref<'v, V>(graph: &'v V) -> Result<Self::Ref<'v>, &'v V> {
        cast!(graph, &Self).or_else(|graph| {
            match cast!(graph, &TraceRef) {
                Ok(r) => {
                    let r: &Tracer<Generic> = r.into();
                    match r.try_cast_ref::<Device>() {
                        Ok(r) => Ok(r.into()),
                        Err(_) => Err(graph),
                    }
                }
                Err(graph) => Err(graph),
            }
        })
    }
    fn try_into_value<V: 'static>(self) -> Result<V, Self> {
        cast!(self, V)
    }
}

impl Graph for ArrayRef {
    type Owned = Self;
    type Builder<L: Leaf> = LeafBuilder<Self>;

    fn replace<'g, L: Leaf, F: Filter<L>, S: GraphSource<L::Ref<'g>, L>>(
        &'g self, filter: F, source: S, _ctx: &mut GraphContext,
    ) -> Result<Self::Owned, S::Error> {
        let trace_cell_ref = self.tracer().cell_ref();
        match filter_matches_ref_any!(filter, self, trace_cell_ref) {
            Ok(r) => Ok(
                Self::try_from_value(source.leaf(r)?).map_err(|_| S::Error::invalid_leaf())?,
            ),
            Err(_) => Ok(self.clone()),
        }
    }
    fn builder<'g, L: Leaf, F: Filter<L>, E: Error>(
        &'g self, filter: F, _ctx: &mut GraphContext,
    ) -> Result<Self::Builder<L>, E> {
        let trace_cell_ref = self.tracer().cell_ref();
        match filter_matches_ref_any!(filter, self, trace_cell_ref) {
            Ok(_) => Ok(LeafBuilder::Leaf),
            Err(_) => Ok(LeafBuilder::Static(self.clone())),
        }
    }
    fn visit<'g, L: Leaf, F: Filter<L>, V: GraphVisitor<'g, Self, L>>(
        &'g self, filter: F, visitor: V,
    ) -> V::Output {
        let trace_cell_ref = self.tracer().cell_ref();
        match filter_matches_ref_any!(filter, self, trace_cell_ref) {
            Ok(r) => visitor.visit_leaf(r),
            Err(_) => visitor.visit_static::<Self>(self)
        }
    }
    fn visit_into<'g, L: Leaf, F: Filter<L>, C: GraphConsumer<Self, L>>(
        self, filter: F, consumer: C,
    ) -> C::Output {
        let trace_cell_ref = self.tracer().cell_ref();
        if let Ok(_) = filter.matches_ref(&self) {
            let value = filter.matches_value(self).or(Err(())).expect("Filter matched ref but not value");
            return consumer.consume_leaf(value)
        }
        if let Ok(_) = filter.matches_ref(trace_cell_ref) {
            let trace_cell_ref: TraceCellRef = self.into_tracer().into();
            let value = filter.matches_value(trace_cell_ref).or(Err(())).expect("Filter matched ref but not value");
            return consumer.consume_leaf(value)
        }
        consumer.consume_static::<Self>(self)
    }
}

impl Leaf for ArrayRef {
    type Ref<'l> = &'l ArrayRef;
    fn as_ref<'l>(&'l self) -> Self::Ref<'l> { self }
    fn clone_ref(v: Self::Ref<'_>) -> Self { v.clone() }
    fn try_from_value<V>(g: V) -> Result<Self, V> {
        cast!(g, Self).or_else(|g| {
            if let Ok(r) = cast!(&g, &TraceCellRef) {
                let r: &TracerCell<Generic> = r.into();
                if r.try_cast_ref::<Array>().is_ok() {
                    let r: TracerCell<Generic> = cast!(g, TraceCellRef).ok().unwrap().into();
                    return Ok(ArrayRef::from(r.try_cast::<Array>().ok().unwrap()))
                }
            }
            Err(g)
        })
    }
    fn try_from_ref<'v, V>(graph: &'v V) -> Result<Self::Ref<'v>, &'v V> {
        cast!(graph, &Self).or_else(|graph| {
            match cast!(graph, &TraceCellRef) {
                Ok(r) => {
                    let r: &TracerCell<Generic> = r.into();
                    match r.try_cast_ref::<Array>() {
                        Ok(r) => Ok(r.into()),
                        Err(_) => Err(graph),
                    }
                }
                Err(graph) => Err(graph),
            }
        })
    }
    fn try_into_value<V: 'static>(self) -> Result<V, Self> {
        cast!(self, V)
    }
}
use crate::util::impl_lifetime_free;
use crate::*;

use gnx_base::{Array, ArrayInfo, ArrayRef};
use gnx_expr::trace::{CellTrace, CellTracer, Generic, ObjectTracer, Tracer};

use super::trace::{cell_ref, into_cell_ref, into_value_ref, value_ref};
use std::any::TypeId;

impl_lifetime_free!(Array, ArrayRef, ArrayInfo, gnx_base::DType);
impl_leaf!(ArrayInfo, gnx_base::DType);

impl Graph for Array {
    type Owned = Self;
    type Builder<L: Leaf> = LeafBuilder<Self>;

    fn replace<'g, L: Leaf, F: Filter<L>, S: GraphSource<L::Ref<'g>, L>>(
        &'g self,
        filter: F,
        source: S,
        _ctx: &mut GraphContext,
    ) -> Result<Self::Owned, S::Error> {
        let trace_ref = self.tracer().trace_ref();
        match filter_matches_ref_any!(filter, self, trace_ref) {
            Ok(r) => {
                Ok(Self::try_from_value(source.leaf(r)?).map_err(|_| S::Error::invalid_leaf())?)
            }
            Err(_) => Ok(self.clone()),
        }
    }
    fn builder<'g, L: Leaf, F: Filter<L>, E: Error>(
        &'g self,
        filter: F,
        _ctx: &mut GraphContext,
    ) -> Result<Self::Builder<L>, E> {
        let trace_ref = self.tracer().trace_ref();
        match filter_matches_ref_any!(filter, self, trace_ref) {
            Ok(_) => Ok(LeafBuilder::Leaf),
            Err(_) => Ok(LeafBuilder::Static(self.clone())),
        }
    }
    fn visit<'g, L: Leaf, F: Filter<L>, V: GraphVisitor<'g, Self, L, F::Annotation<'g>>>(
        &'g self,
        filter: F,
        visitor: V,
    ) -> V::Output {
        let trace_ref = self.tracer().trace_ref();
        match filter_matches_ref_any!(filter, self, trace_ref) {
            Ok(r) => visitor.visit_leaf(r, filter),
            Err(_) => visitor.visit_static::<Self>(self),
        }
    }
    fn visit_into<'g, L: Leaf, F: Filter<L>, C: GraphConsumer<Self, L>>(
        self,
        filter: F,
        consumer: C,
    ) -> C::Output {
        let trace_ref = self.tracer().trace_ref();
        // Try to match each value in order. If the ref matches, convert the value
        // and call the conversion on the value type.
        if let Ok(_) = filter.matches_ref(&self) {
            let value = filter
                .matches_value(self)
                .or(Err(()))
                .expect("Filter matched ref but not value");
            return consumer.consume_leaf(value);
        }
        if let Ok(_) = filter.matches_ref(trace_ref) {
            let trace_ref = self.into_tracer().into_trace_ref();
            let value = filter
                .matches_value(trace_ref)
                .or(Err(()))
                .expect("Filter matched ref but not value");
            return consumer.consume_leaf(value);
        }
        // If we get here, the filter didn't match any of the refs, so we're static.
        // Consume as the wrapped Array type.
        consumer.consume_static::<Self>(self)
    }
}

impl Leaf for Array {
    type Ref<'l> = &'l Array;
    fn as_ref<'l>(&'l self) -> Self::Ref<'l> {
        self
    }
    fn clone_ref(v: Self::Ref<'_>) -> Self {
        v.clone()
    }
    fn try_from_value<V>(g: V) -> Result<Self, V> {
        cast!(g, Self).or_else(|g| {
            if let Some(r) = value_ref(&g) {
                let r: &Tracer<Generic> = r.into();
                if r.try_cast_ref::<Array>().is_ok() {
                    // We know that g is a Array, so we can do an unchecked cast to Array.
                    let r: Tracer<Generic> = into_value_ref(g).ok().unwrap().into();
                    // SAFETY: We know that it is safe to cast to Array since
                    // we just checked that the info matches.
                    return Ok(Array::from(unsafe { r.cast_unchecked() }));
                }
            }
            Err(g)
        })
    }
    fn try_from_ref<'v, V>(graph: &'v V) -> Result<Self::Ref<'v>, &'v V> {
        cast!(graph, &Self).or_else(|graph| match value_ref(graph) {
            Some(r) => {
                let r: &Tracer<Generic> = r.into();
                match r.try_cast_ref::<Array>() {
                    Ok(r) => Ok(r.into()),
                    Err(_) => Err(graph),
                }
            }
            None => Err(graph),
        })
    }
    fn try_into_value<V: 'static>(self) -> Result<V, Self> {
        cast!(self, V).or_else(|value| {
            if TypeId::of::<V>() == TypeId::of::<ObjectTracer>() {
                return Ok(
                    cast!(ObjectTracer::Ref(value.into_tracer().into_trace_ref()), V)
                        .ok()
                        .unwrap(),
                );
            }
            if let Ok(_) = cast!(value.tracer(), &V) {
                return Ok(cast!(value.into_tracer(), V).ok().unwrap());
            }
            if let Ok(_) = cast!(value.tracer().trace_ref(), &V) {
                return Ok(cast!(value.into_tracer().into_trace_ref(), V).ok().unwrap());
            }
            Err(value)
        })
    }
}

impl Graph for ArrayRef {
    type Owned = Self;
    type Builder<L: Leaf> = LeafBuilder<Self>;

    fn replace<'g, L: Leaf, F: Filter<L>, S: GraphSource<L::Ref<'g>, L>>(
        &'g self,
        filter: F,
        source: S,
        _ctx: &mut GraphContext,
    ) -> Result<Self::Owned, S::Error> {
        let trace_cell_ref = self.tracer().cell_ref();
        match filter_matches_ref_any!(filter, self, trace_cell_ref) {
            Ok(r) => {
                Ok(Self::try_from_value(source.leaf(r)?).map_err(|_| S::Error::invalid_leaf())?)
            }
            Err(_) => Ok(self.clone()),
        }
    }
    fn builder<'g, L: Leaf, F: Filter<L>, E: Error>(
        &'g self,
        filter: F,
        _ctx: &mut GraphContext,
    ) -> Result<Self::Builder<L>, E> {
        let trace_cell_ref = self.tracer().cell_ref();
        match filter_matches_ref_any!(filter, self, trace_cell_ref) {
            Ok(_) => Ok(LeafBuilder::Leaf),
            Err(_) => Ok(LeafBuilder::Static(self.clone())),
        }
    }
    fn visit<'g, L: Leaf, F: Filter<L>, V: GraphVisitor<'g, Self, L, F::Annotation<'g>>>(
        &'g self,
        filter: F,
        visitor: V,
    ) -> V::Output {
        let trace_cell_ref = self.tracer().cell_ref();
        match filter_matches_ref_any!(filter, self, trace_cell_ref) {
            Ok(r) => visitor.visit_leaf(r, filter),
            Err(_) => visitor.visit_static::<Self>(self),
        }
    }
    fn visit_into<'g, L: Leaf, F: Filter<L>, C: GraphConsumer<Self, L>>(
        self,
        filter: F,
        consumer: C,
    ) -> C::Output {
        let trace_cell_ref = self.tracer().cell_ref();
        if let Ok(_) = filter.matches_ref(&self) {
            let value = filter
                .matches_value(self)
                .or(Err(()))
                .expect("Filter matched ref but not value");
            return consumer.consume_leaf(value);
        }
        if let Ok(_) = filter.matches_ref(trace_cell_ref) {
            let trace_cell_ref: CellTrace = self.into_tracer().into();
            let value = filter
                .matches_value(trace_cell_ref)
                .or(Err(()))
                .expect("Filter matched ref but not value");
            return consumer.consume_leaf(value);
        }
        consumer.consume_static::<Self>(self)
    }
}

impl Leaf for ArrayRef {
    type Ref<'l> = &'l ArrayRef;
    fn as_ref<'l>(&'l self) -> Self::Ref<'l> {
        self
    }
    fn clone_ref(v: Self::Ref<'_>) -> Self {
        v.clone()
    }
    fn try_from_value<V>(g: V) -> Result<Self, V> {
        cast!(g, Self).or_else(|g| {
            if let Some(r) = cell_ref(&g) {
                let r: &CellTracer<Generic> = r.into();
                if r.try_cast_ref::<Array>().is_ok() {
                    let r: CellTracer<Generic> = into_cell_ref(g).ok().unwrap().into();
                    return Ok(ArrayRef::from(r.try_cast::<Array>().ok().unwrap()));
                }
            }
            Err(g)
        })
    }
    fn try_from_ref<'v, V>(graph: &'v V) -> Result<Self::Ref<'v>, &'v V> {
        cast!(graph, &Self).or_else(|graph| match cell_ref(graph) {
            Some(r) => {
                let r: &CellTracer<Generic> = r.into();
                match r.try_cast_ref::<Array>() {
                    Ok(r) => Ok(r.into()),
                    Err(_) => Err(graph),
                }
            }
            None => Err(graph),
        })
    }
    fn try_into_value<V: 'static>(self) -> Result<V, Self> {
        cast!(self, V).or_else(|value| {
            if TypeId::of::<V>() == TypeId::of::<ObjectTracer>() {
                let cell: CellTrace = value.into_tracer().into();
                return Ok(cast!(ObjectTracer::Cell(cell), V).ok().unwrap());
            }
            if TypeId::of::<V>() == TypeId::of::<CellTrace>() {
                let cell: CellTrace = value.into_tracer().into();
                return Ok(cast!(cell, V).ok().unwrap());
            }
            Err(value)
        })
    }
}

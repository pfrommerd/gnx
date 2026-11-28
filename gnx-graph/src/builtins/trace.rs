//! Transparent graph leaves for immutable trace references and mutable cell handles.

use crate::util::impl_lifetime_free;
use crate::*;
use gnx_expr::{CellTrace, CellTracer, ObjectTracer, ObjectTracerRef, Trace, Traceable, Tracer};

impl_lifetime_free!(Trace, CellTrace, ObjectTracer);
impl_lifetime_free!(for<T: Traceable> Tracer<T>);
impl_lifetime_free!(for<T: Traceable> CellTracer<T>);
impl_leaf!(Trace);
impl_leaf!(CellTrace);

// Borrow the contained handle, never the cell's contents. Typed graph adapters
// also use these helpers when rebuilding from a heterogeneous trace-object source.
pub(super) fn value_ref<V>(graph: &V) -> Option<&Trace> {
    if let Ok(value) = cast!(graph, &Trace) {
        return Some(value);
    }
    match cast!(graph, &ObjectTracer).ok()? {
        ObjectTracer::Ref(value) => Some(value),
        ObjectTracer::Cell(_) => None,
    }
}
pub(super) fn cell_ref<V>(graph: &V) -> Option<&CellTrace> {
    if let Ok(cell) = cast!(graph, &CellTrace) {
        return Some(cell);
    }
    match cast!(graph, &ObjectTracer).ok()? {
        ObjectTracer::Cell(cell) => Some(cell),
        ObjectTracer::Ref(_) => None,
    }
}
pub(super) fn into_value_ref<V>(graph: V) -> Result<Trace, V> {
    cast!(graph, Trace).or_else(|graph| {
        if matches!(cast!(&graph, &ObjectTracer), Ok(ObjectTracer::Ref(_))) {
            let ObjectTracer::Ref(value) = cast!(graph, ObjectTracer).ok().unwrap() else {
                unreachable!()
            };
            Ok(value)
        } else {
            Err(graph)
        }
    })
}
pub(super) fn into_cell_ref<V>(graph: V) -> Result<CellTrace, V> {
    cast!(graph, CellTrace).or_else(|graph| {
        if matches!(cast!(&graph, &ObjectTracer), Ok(ObjectTracer::Cell(_))) {
            let ObjectTracer::Cell(cell) = cast!(graph, ObjectTracer).ok().unwrap() else {
                unreachable!()
            };
            Ok(cell)
        } else {
            Err(graph)
        }
    })
}

impl Leaf for ObjectTracer {
    type Ref<'a> = ObjectTracerRef<'a>;

    fn as_ref<'a>(&'a self) -> ObjectTracerRef<'a> {
        self.as_ref()
    }
    fn clone_ref(value: ObjectTracerRef<'_>) -> Self {
        value.to_owned()
    }
    fn try_from_value<V>(value: V) -> Result<Self, V> {
        cast!(value, Self)
            .or_else(|value| cast!(value, Trace).map(Self::Ref))
            .or_else(|value| cast!(value, CellTrace).map(Self::Cell))
    }
    fn try_from_ref<'a, V>(graph: &'a V) -> Result<ObjectTracerRef<'a>, &'a V> {
        if let Ok(object) = cast!(graph, &Self) {
            return Ok(object.as_ref());
        }
        if let Ok(value) = cast!(graph, &Trace) {
            return Ok(ObjectTracerRef::Ref(value));
        }
        if let Ok(cell) = cast!(graph, &CellTrace) {
            return Ok(ObjectTracerRef::Cell(cell));
        }
        Err(graph)
    }
    fn try_into_value<V: 'static>(self) -> Result<V, Self> {
        cast!(self, V).or_else(|object| match object {
            Self::Ref(value) => cast!(value, V).map_err(Self::Ref),
            Self::Cell(cell) => cast!(cell, V).map_err(Self::Cell),
        })
    }
}

fn matches<'a, L: Leaf, F: Filter<Leaf = L>>(
    object: &'a ObjectTracer,
    filter: &F,
) -> Option<L::Ref<'a>> {
    // Prefer matching the whole object, then expose its handle as an alternate
    // view. Never visit both and never descend into the trace DAG or a cell.
    filter.matches_ref(object).ok().or_else(|| match object {
        ObjectTracer::Ref(value) => filter.matches_ref(value).ok(),
        ObjectTracer::Cell(cell) => filter.matches_ref(cell).ok(),
    })
}

impl Graph for ObjectTracer {
    type Owned = Self;
    type Builder<L: Leaf> = LeafBuilder<Self>;

    fn replace<'g, L: Leaf, F: Filter<Leaf = L>, S: GraphSource<L::Ref<'g>, L>>(
        &'g self,
        filter: F,
        source: S,
        _ctx: &mut GraphContext,
    ) -> Result<Self, S::Error> {
        match matches::<L, F>(self, &filter) {
            Some(reference) => source
                .leaf(reference)?
                .try_into_value::<Self>()
                .or_else(Self::try_from_value)
                .map_err(|_| S::Error::invalid_leaf()),
            None => Ok(self.clone()),
        }
    }
    fn builder<L: Leaf, F: Filter<Leaf = L>, E: Error>(
        &self,
        filter: F,
        _ctx: &mut GraphContext,
    ) -> Result<Self::Builder<L>, E> {
        Ok(if matches::<L, F>(self, &filter).is_some() {
            LeafBuilder::Leaf
        } else {
            LeafBuilder::Static(self.clone())
        })
    }
    fn visit<'g, L: Leaf, F: Filter<Leaf = L>, V: GraphVisitor<'g, Self, L, F::Annotation<'g>>>(
        &'g self,
        filter: F,
        visitor: V,
    ) -> V::Output {
        match matches::<L, F>(self, &filter) {
            Some(reference) => visitor.visit_leaf(reference, filter),
            None => visitor.visit_static::<Self>(self.as_ref()),
        }
    }
    fn visit_into<'g, L: Leaf, F: Filter<Leaf = L>, C: GraphConsumer<Self, L>>(
        self,
        filter: F,
        consumer: C,
    ) -> C::Output {
        if filter.matches_ref(&self).is_ok() {
            let value = filter
                .matches_value(self)
                .ok()
                .expect("filter matched ref but not value");
            return consumer.consume_leaf(value);
        }
        match self {
            Self::Ref(value) => {
                if filter.matches_ref(&value).is_ok() {
                    let value = filter
                        .matches_value(value)
                        .ok()
                        .expect("filter matched ref but not value");
                    consumer.consume_leaf(value)
                } else {
                    consumer.consume_static::<Self>(Self::Ref(value))
                }
            }
            Self::Cell(cell) => {
                if filter.matches_ref(&cell).is_ok() {
                    let value = filter
                        .matches_value(cell)
                        .ok()
                        .expect("filter matched ref but not value");
                    consumer.consume_leaf(value)
                } else {
                    consumer.consume_static::<Self>(Self::Cell(cell))
                }
            }
        }
    }
}

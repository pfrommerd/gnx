use anyhash::{Hash, HasherWrite};

use gnx_expr::{CellKey, CellTrace, ObjectTracer, Trace, TracerKey};
use gnx_math::{Array, ArrayInfo, ArrayRef, Device, DeviceInfo};

use super::hash::CoreHash;
use crate::*;

// Unfiltered trace leaves compare by handle, not contents. When selected, only
// the leaf hole is compared; tracing metadata belongs to the caller's key.
macro_rules! impl_trace_traits {
    ($ty:ty, $identity:path) => {
        impl GraphPartialEq for $ty {
            fn graph_partial_eq<L: Leaf, F: Filter<L>>(
                &self,
                other: &Self,
                filter: F,
                _ctx: &mut GraphEqContext,
            ) -> bool {
                match (
                    self.visit(&filter, IsSelected),
                    other.visit(&filter, IsSelected),
                ) {
                    (true, true) => true,
                    (false, false) => $identity(self) == $identity(other),
                    _ => false,
                }
            }
        }

        impl GraphEq for $ty {
            fn builders_eq<L: Leaf>(
                left: &Self::Builder<L>,
                right: &Self::Builder<L>,
                ctx: &mut GraphEqContext,
            ) -> bool {
                left.graph_eq(right, ctx)
            }
        }

        impl GraphHash for $ty {
            fn graph_hash<H: HasherWrite, L: Leaf, F: Filter<L>>(
                &self,
                hasher: &mut H,
                filter: F,
                _ctx: &mut GraphHashContext,
            ) {
                if self.visit(filter, IsSelected) {
                    HashMarker::Excluded.hash(hasher);
                    std::any::type_name::<L>().hash(hasher);
                } else {
                    HashMarker::Included.hash(hasher);
                    std::any::type_name::<Self>().hash(hasher);
                    CoreHash($identity(self)).hash(hasher);
                }
            }

            fn hash_builder<H: HasherWrite, L: Leaf>(
                builder: &Self::Builder<L>,
                hasher: &mut H,
                ctx: &mut GraphHashContext,
            ) {
                match builder {
                    LeafBuilder::Leaf => {
                        HashMarker::Excluded.hash(hasher);
                        std::any::type_name::<L>().hash(hasher);
                    }
                    LeafBuilder::Static(value) => {
                        value.graph_hash(hasher, Nothing::<()>::filter(), ctx)
                    }
                }
            }
        }
    };
}

struct IsSelected;
impl<'g, G: Graph, L: Leaf> GraphVisitor<'g, G, L> for IsSelected {
    type Output = bool;
    fn visit_leaf(self, _: L::Ref<'g>) -> bool {
        true
    }
    fn visit_static<S: Leaf>(self, _: S::Ref<'g>) -> bool {
        false
    }
    fn visit_node<N: Node, F: Filter<L>>(self, _: View<'g, N, F>) -> bool {
        unreachable!("expected a leaf")
    }
    fn visit_shared<S: Graph, F: Filter<L>>(self, _: GraphId, _: View<'g, S, F>) -> bool {
        unreachable!("expected a leaf")
    }
}

#[derive(std::hash::Hash, PartialEq, Eq)]
enum ObjectIdentity {
    Value(TracerKey),
    Cell(CellKey),
}
fn object_identity(value: &ObjectTracer) -> ObjectIdentity {
    match value {
        ObjectTracer::Ref(value) => ObjectIdentity::Value(TracerKey::from(value)),
        ObjectTracer::Cell(cell) => ObjectIdentity::Cell(CellKey::from(cell)),
    }
}
fn array_identity(value: &Array) -> TracerKey {
    TracerKey::from(value.tracer().trace_ref())
}
fn array_cell_identity(value: &ArrayRef) -> CellKey {
    CellKey::from(value.tracer().cell_ref())
}
fn device_identity(value: &Device) -> TracerKey {
    TracerKey::from(value.tracer().trace_ref())
}

impl_trace_traits!(Trace, TracerKey::from);
impl_trace_traits!(CellTrace, CellKey::from);
impl_trace_traits!(ObjectTracer, object_identity);
impl_trace_traits!(Array, array_identity);
impl_trace_traits!(ArrayRef, array_cell_identity);
impl_trace_traits!(Device, device_identity);
impl_trace_traits!(ArrayInfo, ArrayInfo::clone);
impl_trace_traits!(DeviceInfo, DeviceInfo::clone);

use crate::{bytes::ImBytes, string::ImString};
use gnx::graph::*;
use gnx::util::{try_specialize, LifetimeFree};
use pyo3::prelude::*;
use std::sync::Arc;

#[derive(Clone)]
pub struct ObjectHandle(pub Arc<Py<PyAny>>);
unsafe impl LifetimeFree for ObjectHandle {}

#[derive(Clone)]
pub enum PyLeaf {
    None,
    Int(i64),
    Float(f64),
    Bool(bool),
    Bytes(ImBytes),
    String(ImString),
    // A generic python object as a leaf
    Other(ObjectHandle),
}
#[rustfmt::skip]
#[derive(Clone, Copy)]
pub enum PyLeafRef<'l> {
    None, Int(&'l i64), Float(&'l f64), Bool(&'l bool),
    Bytes(&'l [u8]), String(&'l str),
    Other(&'l ObjectHandle),
}
impl PyLeafRef<'_> {
    pub fn cloned(self) -> PyLeaf {
        use PyLeafRef::*;
        match self {
            None => PyLeaf::None,
            Int(v) => PyLeaf::Int(v.clone()),
            Float(v) => PyLeaf::Float(*v),
            Bool(v) => PyLeaf::Bool(*v),
            Bytes(v) => PyLeaf::Bytes(Vec::from(v).into()),
            String(v) => PyLeaf::String(std::string::String::from(v).into()),
            Other(v) => PyLeaf::Other(v.clone()),
        }
    }
}

#[rustfmt::skip]
impl Graph for PyLeaf {
    type Owned = Self;
    type Builder<L: Leaf> = LeafBuilder<Self>;

    fn replace<'g, L: Leaf, F: Filter<L>, S: GraphSource<L::Ref<'g>, L>>(
        &'g self, filter: F, source: S, _ctx: &mut GraphContext,
    ) -> Result<Self::Owned, S::Error> {
        match filter.matches_ref(self) {
            Ok(r) => Ok(source.leaf(r)?
                .try_into_value()
                .map_err(|_| S::Error::invalid_leaf())?),
            Err(_) => Ok(self.clone())
        }
    }
    fn builder<'g, L: Leaf, F: Filter<L>, E: Error>(
        &'g self, filter: F, _: &mut GraphContext
    ) -> Result<Self::Builder<L>, E> {
        match filter.matches_ref(self) {
            Ok(_) => Ok(LeafBuilder::Leaf),
            Err(_) => Ok(LeafBuilder::Static(self.clone())),
        }
    }
    fn visit<'g, L: Leaf, F: Filter<L>, V: GraphVisitor<'g, Self, L>>(
        &'g self, filter: F, visitor: V
    ) -> V::Output {
        match filter.matches_ref(self) {
            Ok(r) => visitor.visit_leaf(r),
            Err(s) => visitor.visit_static::<Self>(s.as_ref())
        }
    }
    fn visit_into<L: Leaf, F: Filter<L>, C: GraphConsumer<Self, L>>(
        self, filter: F, consumer: C
    ) -> C::Output {
        match filter.matches_value(self) {
            Ok(v) => consumer.consume_leaf(v),
            Err(s) => consumer.consume_static::<Self>(s)
        }
    }
}
impl TypedGraph<PyLeaf> for PyLeaf {}

#[rustfmt::skip]
impl Leaf for PyLeaf {
    type Ref<'l> = PyLeafRef<'l> where Self: 'l;
    fn as_ref<'l>(&'l self) -> Self::Ref<'l> {
        use PyLeaf::*;
        match self {
            None => PyLeafRef::None, Int(v) => PyLeafRef::Int(v),
            Float(v) => PyLeafRef::Float(v), Bool(v) => PyLeafRef::Bool(v),
            Bytes(v) => PyLeafRef::Bytes(v.as_ref()),
            String(v) => PyLeafRef::String(v.as_str()),
            Other(v) => PyLeafRef::Other(v),
        }
    }
    fn clone_ref(v: Self::Ref<'_>) -> Self { v.cloned() }
    fn try_from_ref<'v, V>(graph: &'v V) -> Result<Self::Ref<'v>, &'v V> {
        Err(graph).or_else(|graph| try_specialize!(graph, &'v i64).map(PyLeafRef::Int))
            .or_else(|graph| try_specialize!(graph, &'v f64).map(PyLeafRef::Float))
            .or_else(|graph| try_specialize!(graph, &'v ImBytes).map(|x| PyLeafRef::Bytes(x.as_slice())))
            .or_else(|graph| try_specialize!(graph, &'v Vec<u8>).map(|x| PyLeafRef::Bytes(x.as_slice())))
            .or_else(|graph| try_specialize!(graph, &'v ImString).map(|x| PyLeafRef::String(x.as_str())))
            .or_else(|graph| try_specialize!(graph, &'v String).map(|x| PyLeafRef::String(x.as_str())))
            .or_else(|graph| try_specialize!(graph, &'v ObjectHandle).map(PyLeafRef::Other))
    }
    fn try_from_value<V>(graph: V) -> Result<Self, V> {
        Err(graph).or_else(|graph| try_specialize!(graph, i64).map(PyLeaf::Int))
            .or_else(|graph| try_specialize!(graph, f64).map(PyLeaf::Float))
            .or_else(|graph| try_specialize!(graph, ImBytes).map(PyLeaf::Bytes))
            .or_else(|graph| try_specialize!(graph, Vec<u8>).map(|x| PyLeaf::Bytes(x.into())))
            .or_else(|graph| try_specialize!(graph, ImString).map(PyLeaf::String))
            .or_else(|graph| try_specialize!(graph, String).map(|x| PyLeaf::String(x.into())))
            .or_else(|graph| try_specialize!(graph, ObjectHandle).map(PyLeaf::Other))
    }
    fn try_into_value<V: 'static>(self) -> Result<V, Self> {
        // Try casting into any of the possible types, if that fails,
        // try the leaf value itself
        match self {
            PyLeaf::None => try_specialize!(PyLeaf::None, V).map_err(|_| PyLeaf::None),
            PyLeaf::Int(v) => try_specialize!(v, V).map_err(PyLeaf::Int),
            PyLeaf::Float(v) => try_specialize!(v, V).map_err(PyLeaf::Float),
            PyLeaf::Bool(v) => try_specialize!(v, V).map_err(PyLeaf::Bool),
            PyLeaf::Bytes(v) => try_specialize!(v, V).map_err(PyLeaf::Bytes),
            PyLeaf::String(v) => try_specialize!(v, V).map_err(PyLeaf::String),
            PyLeaf::Other(v) => try_specialize!(v, V).map_err(PyLeaf::Other),
        }.or_else(|leaf| try_specialize!(leaf, V))
    }
}

use crate::{bytes::ImBytes, string::ImString};
use gnx::graph::*;
use gnx::util::cast;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes};
use pyo3::Bound as PyBound;
use std::convert::Infallible;

#[derive(Clone)]
pub enum PyLeaf {
    None,
    Int(i64),
    Float(f64),
    Bool(bool),
    Bytes(ImBytes),
    String(ImString),
}

impl PyLeaf {
    pub fn into_pyobject_bound<'py>(self, py: Python<'py>) -> PyResult<PyBound<'py, PyAny>> {
        Ok(match self {
            PyLeaf::None => py.None().into_bound(py),
            PyLeaf::Int(v) => v.into_pyobject(py)?.into_any(),
            PyLeaf::Float(v) => v.into_pyobject(py)?.into_any(),
            PyLeaf::Bool(v) => PyBool::new(py, v).to_owned().into_any(),
            PyLeaf::Bytes(v) => PyBytes::new(py, v.as_slice()).into_any(),
            PyLeaf::String(v) => v.as_str().into_pyobject(py)?.into_any(),
        })
    }
}

impl<'py> IntoPyObject<'py> for PyLeaf {
    type Target = PyAny;
    type Output = PyBound<'py, PyAny>;
    type Error = Infallible;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(self
            .into_pyobject_bound(py)
            .expect("PyLeaf conversion is infallible"))
    }
}

#[rustfmt::skip]
#[derive(Clone, Copy)]
pub enum PyLeafRef<'l> {
    None, Int(&'l i64), Float(&'l f64), Bool(&'l bool),
    Bytes(&'l [u8]), String(&'l str),
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
        }
    }
    fn clone_ref(v: Self::Ref<'_>) -> Self { v.cloned() }
    fn try_from_ref<'v, V>(graph: &'v V) -> Result<Self::Ref<'v>, &'v V> {
        Err(graph).or_else(|graph| cast!(graph, &'v i64).map(PyLeafRef::Int))
            .or_else(|graph| cast!(graph, &'v f64).map(PyLeafRef::Float))
            .or_else(|graph| cast!(graph, &'v ImBytes).map(|x| PyLeafRef::Bytes(x.as_slice())))
            .or_else(|graph| cast!(graph, &'v Vec<u8>).map(|x| PyLeafRef::Bytes(x.as_slice())))
            .or_else(|graph| cast!(graph, &'v ImString).map(|x| PyLeafRef::String(x.as_str())))
            .or_else(|graph| cast!(graph, &'v String).map(|x| PyLeafRef::String(x.as_str())))
    }
    fn try_from_value<V>(graph: V) -> Result<Self, V> {
        Err(graph).or_else(|graph| cast!(graph, i64).map(PyLeaf::Int))
            .or_else(|graph| cast!(graph, f64).map(PyLeaf::Float))
            .or_else(|graph| cast!(graph, ImBytes).map(PyLeaf::Bytes))
            .or_else(|graph| cast!(graph, Vec<u8>).map(|x| PyLeaf::Bytes(x.into())))
            .or_else(|graph| cast!(graph, ImString).map(PyLeaf::String))
            .or_else(|graph| cast!(graph, String).map(|x| PyLeaf::String(x.into())))
    }
    fn try_into_value<V: 'static>(self) -> Result<V, Self> {
        // Try casting into any of the possible types, if that fails,
        // try the leaf value itself
        match self {
            PyLeaf::None => cast!(PyLeaf::None, V).map_err(|_| PyLeaf::None),
            PyLeaf::Int(v) => cast!(v, V).map_err(PyLeaf::Int),
            PyLeaf::Float(v) => cast!(v, V).map_err(PyLeaf::Float),
            PyLeaf::Bool(v) => cast!(v, V).map_err(PyLeaf::Bool),
            PyLeaf::Bytes(v) => cast!(v, V).map_err(PyLeaf::Bytes),
            PyLeaf::String(v) => cast!(v, V).map_err(PyLeaf::String),
        }.or_else(|leaf| cast!(leaf, V))
    }
}

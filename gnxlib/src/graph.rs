use gnx::expr::ObjectTracer;
use gnx::graph::anyhash::{Hash, HasherWrite};
use gnx::graph::*;
use gnx::util::LifetimeFree;
use pyo3::exceptions::{PyReferenceError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{
    PyBool, PyBytes, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple, PyWeakrefReference,
};
use pyo3::Bound as PyBound;
use std::collections::HashMap;
use std::sync::Arc;

use crate::leaf::PyLeaf;
use crate::tracer::{extract_tracer, wrap_tracer};

#[pyclass(name = "Graph", module = "gnxlib.graph", skip_from_py_object)]
#[derive(Clone)]
pub struct GraphContainer(Arc<PyGraph>);

#[pymethods]
impl GraphContainer {
    /// Parse exact builtins, live GNX tracers, and types with __graph__ hooks.
    #[staticmethod]
    fn from_value(value: PyGraph) -> Self {
        Self(Arc::new(value))
    }

    fn to_value(&self) -> PyGraph {
        (*self.0).clone()
    }
}

/// Accept a Python tree or an already parsed Graph as an evaluation root.
pub(crate) fn parse_roots(value: &PyBound<'_, PyAny>) -> PyResult<Arc<PyGraph>> {
    if let Ok(graph) = value.cast::<GraphContainer>() {
        Ok(graph.borrow().0.clone())
    } else {
        Ok(Arc::new(value.extract()?))
    }
}

#[derive(Clone)]
pub struct CustomType(Arc<Py<PyWeakrefReference>>);

impl CustomType {
    fn new(ty: &PyBound<'_, PyAny>) -> PyResult<Self> {
        // Cache builders must not keep Python classes and their closures alive.
        Ok(Self(Arc::new(PyWeakrefReference::new(ty)?.unbind())))
    }

    fn identity(&self) -> usize {
        // Callback-free weakrefs are reused by CPython. Holding the weakref in
        // the builder also prevents identity reuse after the class is collected.
        self.0.as_ptr() as usize
    }

    fn resolve<'py>(&self, py: Python<'py>) -> PyResult<PyBound<'py, PyAny>> {
        self.0
            .bind(py)
            .upgrade()
            .ok_or_else(|| PyReferenceError::new_err("custom graph type is no longer alive"))
    }
}

#[derive(Clone)]
pub struct DictKey(Key);

impl DictKey {
    fn as_ref(&self) -> KeyRef<'_> {
        self.0.as_ref()
    }

    fn from_py(key: PyBound<'_, PyAny>) -> PyResult<Self> {
        if key.is_exact_instance_of::<PyString>() {
            Ok(Self(Key::from(key.extract::<String>()?)))
        } else if key.is_exact_instance_of::<PyInt>() {
            Ok(Self(Key::from(key.extract::<isize>()?)))
        } else {
            Err(PyTypeError::new_err(
                "graph dict keys and __graph__ keys must be builtin str or int",
            ))
        }
    }

    fn key_object<'py>(&self, py: Python<'py>) -> PyResult<PyBound<'py, PyAny>> {
        match self.as_ref() {
            KeyRef::Named(key) => Ok(key.into_pyobject(py)?.into_any()),
            KeyRef::Index(key) => Ok(key.into_pyobject(py)?.into_any()),
            KeyRef::Transparent => unreachable!("Python graph keys are named or indexed"),
        }
    }
}

#[derive(Clone)]
pub enum PyGraph {
    Leaf(PyLeaf),
    Tracer(ObjectTracer),
    Tuple(Vec<PyGraph>),
    List(Vec<PyGraph>),
    Dict(Vec<(DictKey, PyGraph)>),
    Custom(CustomType, Vec<(DictKey, PyGraph)>),
}
#[derive(Clone)]
pub enum PyBuilder {
    Leaf,
    Static(PyLeaf),
    Tracer(LeafBuilder<ObjectTracer>),
    Tuple(Vec<PyBuilder>),
    List(Vec<PyBuilder>),
    Dict(Vec<(DictKey, PyBuilder)>),
    Custom(CustomType, Vec<(DictKey, PyBuilder)>),
}
unsafe impl LifetimeFree for PyGraph {}

#[rustfmt::skip]
impl Graph for PyGraph {
    type Owned = Self;
    type Builder<L: Leaf> = PyBuilder;

    fn builder<L: Leaf, F: Filter<Leaf = L>, E: Error>(
            &self, filter: F, mut ctx: &mut GraphContext
    ) -> Result<Self::Builder<L>, E> {
        use PyGraph::*;
        Ok(match filter.matches_ref(self) {
            Ok(_) => PyBuilder::Leaf,
            Err(graph) => match graph {
                // Try and match the leaf itself on the filter
                Leaf(leaf) => match filter.matches_ref(leaf) {
                    Ok(_) => PyBuilder::Leaf,
                    Err(_) => PyBuilder::Static(leaf.clone()),
                },
                Tracer(tracer) => PyBuilder::Tracer(tracer.builder(filter, ctx)?),
                Tuple(children) => PyBuilder::Tuple(children.iter().enumerate().map(|(i, child)| {
                    child.builder(filter.child(KeyRef::from(i)), &mut ctx)
                }).collect::<Result<Vec<PyBuilder>, E>>()?),
                List(children) => PyBuilder::List(children.iter().enumerate().map(|(i, child)| {
                    child.builder(filter.child(KeyRef::from(i)), &mut ctx)
                }).collect::<Result<Vec<PyBuilder>, E>>()?),
                Dict(children) => PyBuilder::Dict(children.iter().map(|(key, child)| {
                    Ok((key.clone(), child.builder(filter.child(key.as_ref()), &mut ctx)?))
                }).collect::<Result<Vec<(DictKey, PyBuilder)>, E>>()?),
                Custom(typ, children) => PyBuilder::Custom(typ.clone(), children.iter().map(|(key, child)| {
                    Ok((key.clone(), child.builder(filter.child(key.as_ref()), &mut ctx)?))
                }).collect::<Result<Vec<(DictKey, PyBuilder)>, E>>()?),
            }
        })
    }

    fn replace<'g, L: Leaf, F: Filter<Leaf = L>, S: GraphSource<L::Ref<'g>, L>>(
        &'g self, filter: F, source: S, ctx: &mut GraphContext
    ) -> Result<Self::Owned, S::Error> {
        use PyGraph::*;
        Ok(match filter.matches_ref(self) {
            Ok(r) => L::try_into_value(source.leaf(r)?).map_err(|_| S::Error::invalid_leaf())?,
            Err(graph) => match graph {
                Leaf(leaf) => Leaf(match filter.matches_ref(leaf) {
                    Ok(_) => leaf.replace(filter, source, ctx)?,
                    Err(_) => leaf.clone(),
                }),
                Tracer(tracer) => Tracer(tracer.replace(filter, source, ctx)?),
                Tuple(children) => Tuple({
                    let mut ns = source.node()?;
                    children.iter().enumerate().map(|(i, child)| child.replace(
                            filter.child(KeyRef::from(i)), ns.expect_child(KeyRef::from(i))?, ctx
                    )).collect::<Result<Vec<PyGraph>, S::Error>>()?
                }),
                List(children) => List({
                    let mut ns = source.node()?;
                    children.iter().enumerate().map(|(i, child)| child.replace(
                        filter.child(KeyRef::from(i)), ns.expect_child(KeyRef::from(i))?, ctx
                    )).collect::<Result<Vec<PyGraph>, S::Error>>()?
                }),
                Dict(children) => Dict({
                    let mut ns = source.node()?;
                    children.iter().map(|(key, child)| {
                        let child = child.replace(filter.child(key.as_ref()), ns.expect_child(key.as_ref())?, ctx)?;
                        Ok((key.clone(), child))
                    }).collect::<Result<Vec<(DictKey, PyGraph)>, S::Error>>()?
                }),
                Custom(typ, children) => Custom(typ.clone(), {
                    let mut ns = source.node()?;
                    children.iter().map(|(key, child)| {
                        let child = child.replace(filter.child(key.as_ref()), ns.expect_child(key.as_ref())?, ctx)?;
                        Ok((key.clone(), child))
                    }).collect::<Result<Vec<(DictKey, PyGraph)>, S::Error>>()?
                }),
            }
        })
    }

    fn visit<'g, L: Leaf, F: Filter<Leaf = L>, V: GraphVisitor<'g, Self, L, F::Annotation<'g>>>(
        &'g self,
        filter: F,
        visitor: V,
    ) -> V::Output {
        match filter.matches_ref(self) {
            Ok(r) => visitor.visit_leaf(r, filter),
            Err(graph) => match graph {
                PyGraph::Leaf(leaf) => match filter.matches_ref(leaf) {
                    Ok(r) => visitor.visit_leaf(r, filter),
                    Err(s) => visitor.visit_static::<PyLeaf>(s.as_ref()),
                },
                PyGraph::Tracer(tracer) => tracer.visit(filter, GenericVisitor::<PyGraph, L, V>::from(visitor)),
                _ => visitor.visit_node(View::new(graph, filter.owned())),
            },
        }
    }
    fn visit_into<L: Leaf, F: Filter<Leaf = L>, C: GraphConsumer<Self, L>>(
        self,
        filter: F,
        consumer: C,
    ) -> C::Output {
        match filter.matches_value(self) {
            Ok(v) => consumer.consume_leaf(v),
            Err(graph) => match graph {
                PyGraph::Leaf(leaf) => match filter.matches_value(leaf) {
                    Ok(v) => consumer.consume_leaf(v),
                    Err(s) => consumer.consume_static::<PyLeaf>(s),
                },
                PyGraph::Tracer(tracer) => tracer.visit_into(filter, GenericConsumer::<PyGraph, L, C>::from(consumer)),
                _ => consumer.consume_node(gnx::graph::Filtered::new(graph, filter)),
            },
        }
    }
}

impl GraphHash for PyGraph {
    fn graph_hash<H: HasherWrite, L: Leaf, F: Filter<Leaf = L>>(
        &self,
        hasher: &mut H,
        filter: F,
        ctx: &mut GraphHashContext,
    ) {
        let builder = self
            .builder::<L, _, SourceError>(filter, &mut GraphContext::new())
            .expect("Python graph builder");
        Self::hash_builder::<H, L>(&builder, hasher, ctx);
    }

    fn hash_builder<H: HasherWrite, L: Leaf>(
        builder: &PyBuilder,
        state: &mut H,
        ctx: &mut GraphHashContext,
    ) {
        use PyBuilder::*;
        match builder {
            Leaf => 0u8.hash(state),
            Static(value) => {
                1u8.hash(state);
                hash_static(value, state);
            }
            Tracer(builder) => {
                2u8.hash(state);
                ObjectTracer::hash_builder::<H, L>(builder, state, ctx);
            }
            Tuple(children) | List(children) => {
                state.write_u8(if matches!(builder, Tuple(_)) { 3 } else { 4 });
                children.len().hash(state);
                for child in children {
                    Self::hash_builder::<H, L>(child, state, ctx);
                }
            }
            Dict(children) | Custom(_, children) => {
                if let Custom(typ, _) = builder {
                    6u8.hash(state);
                    typ.identity().hash(state);
                } else {
                    5u8.hash(state);
                }
                children.len().hash(state);
                for (key, child) in children {
                    hash_key(key, state);
                    Self::hash_builder::<H, L>(child, state, ctx);
                }
            }
        }
    }
}

impl GraphPartialEq for PyGraph {
    fn graph_partial_eq<L: Leaf, F: Filter<Leaf = L>>(
        &self,
        other: &Self,
        filter: F,
        ctx: &mut GraphEqContext,
    ) -> bool {
        let left = self
            .builder::<L, _, SourceError>(&filter, &mut ctx.left)
            .expect("Python graph builder");
        let right = other
            .builder::<L, _, SourceError>(filter, &mut ctx.right)
            .expect("Python graph builder");
        Self::builders_eq::<L>(&left, &right, ctx)
    }
}

impl GraphEq for PyGraph {
    fn builders_eq<L: Leaf>(left: &PyBuilder, right: &PyBuilder, ctx: &mut GraphEqContext) -> bool {
        use PyBuilder::*;
        match (left, right) {
            (Leaf, Leaf) => true,
            (Static(a), Static(b)) => static_eq(a, b),
            (Tracer(a), Tracer(b)) => ObjectTracer::builders_eq::<L>(a, b, ctx),
            (Tuple(a), Tuple(b)) | (List(a), List(b)) => {
                same_sequence(a, b, |a, b| Self::builders_eq::<L>(a, b, ctx))
            }
            (Dict(a), Dict(b)) => same_keys::<L>(a, b, ctx),
            (Custom(ty_a, a), Custom(ty_b, b)) => {
                ty_a.identity() == ty_b.identity() && same_keys::<L>(a, b, ctx)
            }
            _ => false,
        }
    }
}

fn same_sequence<T>(a: &[T], b: &[T], mut compare: impl FnMut(&T, &T) -> bool) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| compare(a, b))
}

fn same_keys<L: Leaf>(
    a: &[(DictKey, PyBuilder)],
    b: &[(DictKey, PyBuilder)],
    ctx: &mut GraphEqContext,
) -> bool {
    same_sequence(a, b, |(key_a, a), (key_b, b)| {
        key_a.0 == key_b.0 && PyGraph::builders_eq::<L>(a, b, ctx)
    })
}

fn hash_key(key: &DictKey, state: &mut impl HasherWrite) {
    match key.as_ref() {
        KeyRef::Named(value) => {
            0u8.hash(state);
            value.hash(state);
        }
        KeyRef::Index(value) => {
            1u8.hash(state);
            value.hash(state);
        }
        KeyRef::Transparent => 2u8.hash(state),
    }
}

fn hash_static(value: &PyLeaf, state: &mut impl HasherWrite) {
    use PyLeaf::*;
    match value {
        None => 0u8.hash(state),
        Bool(value) => (1u8, *value).hash(state),
        Int(value) => (2u8, *value).hash(state),
        Float(value) => (3u8, value.to_bits()).hash(state),
        Bytes(value) => (4u8, value.as_slice()).hash(state),
        String(value) => (5u8, value.as_str()).hash(state),
    }
}

fn static_eq(a: &PyLeaf, b: &PyLeaf) -> bool {
    use PyLeaf::*;
    match (a, b) {
        (None, None) => true,
        (Bool(a), Bool(b)) => a == b,
        (Int(a), Int(b)) => a == b,
        (Float(a), Float(b)) => a.to_bits() == b.to_bits(),
        (Bytes(a), Bytes(b)) => a.as_slice() == b.as_slice(),
        (String(a), String(b)) => a.as_str() == b.as_str(),
        _ => false,
    }
}

impl Node for PyGraph {
    fn visit_children<'g, L, F, V>(&'g self, filter: F, mut visitor: V) -> V::Output
    where
        L: Leaf,
        F: Filter<Leaf = L>,
        V: ChildrenVisitor<'g, Self, L, F::Annotation<'g>>,
    {
        match self {
            PyGraph::Tuple(children) | PyGraph::List(children) => {
                children.iter().enumerate().for_each(|(i, child)| {
                    let key = KeyRef::from(i);
                    visitor.visit_child(key, View::new(child, filter.child(key)));
                });
            }
            PyGraph::Dict(children) | PyGraph::Custom(_, children) => {
                children.iter().for_each(|(key, child)| {
                    visitor.visit_child(key.as_ref(), View::new(child, filter.child(key.as_ref())));
                });
            }
            PyGraph::Leaf(_) | PyGraph::Tracer(_) => {}
        }
        visitor.finish()
    }

    fn visit_into_children<L, F, C>(self, filter: F, mut consumer: C) -> C::Output
    where
        L: Leaf,
        F: Filter<Leaf = L>,
        C: ChildrenConsumer<Self, L>,
    {
        match self {
            PyGraph::Tuple(children) | PyGraph::List(children) => {
                children.into_iter().enumerate().for_each(|(i, child)| {
                    consumer.consume_child(
                        Key::from(i),
                        gnx::graph::Filtered::new(child, filter.child(KeyRef::from(i))),
                    );
                });
            }
            PyGraph::Dict(children) | PyGraph::Custom(_, children) => {
                children.into_iter().for_each(|(key, child)| {
                    consumer.consume_child(
                        key.as_ref().to_value(),
                        gnx::graph::Filtered::new(child, filter.child(key.as_ref())),
                    );
                });
            }
            PyGraph::Leaf(_) | PyGraph::Tracer(_) => {}
        }
        consumer.finish()
    }
}

impl<L: Leaf> Builder<L> for PyBuilder {
    type Graph = PyGraph;

    fn build<S: GraphSource<(), L>>(
        self,
        source: S,
        ctx: &mut GraphContext,
    ) -> Result<Self::Graph, S::Error> {
        use PyBuilder::*;
        Ok(match self {
            Leaf => match L::try_into_value::<PyGraph>(source.leaf(())?) {
                Ok(value) => value,
                Err(leaf) => PyGraph::Leaf(
                    PyLeaf::try_from_value(leaf).map_err(|_| S::Error::invalid_leaf())?,
                ),
            },
            Static(value) => {
                source.empty_leaf()?;
                PyGraph::Leaf(value)
            }
            Tracer(builder) => PyGraph::Tracer(builder.build(source, ctx)?),
            Tuple(children) => PyGraph::Tuple(build_indexed(children, source, ctx)?),
            List(children) => PyGraph::List(build_indexed(children, source, ctx)?),
            Dict(children) => PyGraph::Dict(build_keyed(children, source, ctx)?),
            Custom(typ, children) => PyGraph::Custom(typ, build_keyed(children, source, ctx)?),
        })
    }
}

fn build_indexed<L: Leaf, S: GraphSource<(), L>>(
    children: Vec<PyBuilder>,
    source: S,
    ctx: &mut GraphContext,
) -> Result<Vec<PyGraph>, S::Error> {
    let mut ns = source.node()?;
    children
        .into_iter()
        .enumerate()
        .map(|(i, child)| child.build(ns.expect_child(KeyRef::from(i))?, ctx))
        .collect()
}

fn build_keyed<L: Leaf, S: GraphSource<(), L>>(
    children: Vec<(DictKey, PyBuilder)>,
    source: S,
    ctx: &mut GraphContext,
) -> Result<Vec<(DictKey, PyGraph)>, S::Error> {
    let mut ns = source.node()?;
    children
        .into_iter()
        .map(|(key, child)| {
            let child = child.build(ns.expect_child(key.as_ref())?, ctx)?;
            Ok((key, child))
        })
        .collect()
}

#[derive(Default)]
pub(crate) struct GraphParser<'py> {
    // Hooks may yield temporary containers. Keep them alive so that Python
    // can't reuse their addresses during this parse.
    containers: HashMap<usize, PyBound<'py, PyAny>>,
}

impl<'py> GraphParser<'py> {
    pub(crate) fn parse_args(
        args: &PyBound<'py, PyTuple>,
        kwargs: Option<&PyBound<'py, PyDict>>,
    ) -> PyResult<PyGraph> {
        let mut parser = Self::default();
        let args = parser.parse(args.as_any(), 0)?;
        let kwargs = match kwargs {
            Some(kwargs) => parser.parse(kwargs.as_any(), 0)?,
            None => PyGraph::Dict(Vec::new()),
        };
        Ok(PyGraph::Tuple(vec![args, kwargs]))
    }

    fn parse(&mut self, obj: &PyBound<'py, PyAny>, depth: usize) -> PyResult<PyGraph> {
        if depth > 128 {
            return Err(PyTypeError::new_err("graph nesting depth exceeds 128"));
        }
        if let Some(tracer) = extract_tracer(obj) {
            return Ok(PyGraph::Tracer(tracer));
        }
        // Use exact builtin types, not coercions or Python-defined methods.
        let leaf = if obj.is_none() {
            PyLeaf::None
        } else if obj.is_exact_instance_of::<PyBool>() {
            PyLeaf::Bool(obj.extract()?)
        } else if obj.is_exact_instance_of::<PyInt>() {
            PyLeaf::Int(obj.extract()?)
        } else if obj.is_exact_instance_of::<PyFloat>() {
            PyLeaf::Float(obj.extract()?)
        } else if let Ok(bytes) = obj.cast_exact::<PyBytes>() {
            PyLeaf::Bytes(bytes.as_bytes().to_vec().into())
        } else if obj.is_exact_instance_of::<PyString>() {
            PyLeaf::String(obj.extract::<String>()?.into())
        } else if let Ok(tuple) = obj.cast_exact::<PyTuple>() {
            self.visit_container(obj)?;
            return tuple
                .iter()
                .map(|child| self.parse(&child, depth + 1))
                .collect::<PyResult<_>>()
                .map(PyGraph::Tuple);
        } else if let Ok(list) = obj.cast_exact::<PyList>() {
            self.visit_container(obj)?;
            // Snapshot mutable containers before a child hook can mutate them.
            let children: Vec<_> = list.iter().collect();
            return children
                .into_iter()
                .map(|child| self.parse(&child, depth + 1))
                .collect::<PyResult<_>>()
                .map(PyGraph::List);
        } else if let Ok(dict) = obj.cast_exact::<PyDict>() {
            self.visit_container(obj)?;
            let children: Vec<_> = dict.iter().collect();
            return children
                .into_iter()
                .map(|(key, child)| Ok((DictKey::from_py(key)?, self.parse(&child, depth + 1)?)))
                .collect::<PyResult<_>>()
                .map(PyGraph::Dict);
        } else {
            return self.parse_custom(obj, depth);
        };
        Ok(PyGraph::Leaf(leaf))
    }

    fn parse_custom(&mut self, obj: &PyBound<'py, PyAny>, depth: usize) -> PyResult<PyGraph> {
        let ty = obj.get_type();
        let Some(hooks) = ty.getattr_opt("__graph__")? else {
            return Err(PyTypeError::new_err(format!(
                "unsupported graph value of type {}",
                ty.name()?
            )));
        };
        self.visit_container(obj)?;
        let [flatten, _] = graph_hooks(&hooks)?;
        let mut children = Vec::new();
        for item in flatten.call1((obj,))?.try_iter()? {
            let pair = item?.cast_into_exact::<PyTuple>()?;
            if pair.len() != 2 {
                return Err(PyTypeError::new_err(
                    "__graph__ flatten must yield (key, child) tuples",
                ));
            }
            let key = DictKey::from_py(pair.get_item(0)?)?;
            let child = self.parse(&pair.get_item(1)?, depth + 1)?;
            children.push((key, child));
        }
        Ok(PyGraph::Custom(CustomType::new(ty.as_any())?, children))
    }

    fn visit_container(&mut self, obj: &PyBound<'py, PyAny>) -> PyResult<()> {
        if self
            .containers
            .insert(obj.as_ptr() as usize, obj.clone())
            .is_some()
        {
            return Err(PyTypeError::new_err(
                "graph contains cyclic or shared Python containers",
            ));
        }
        Ok(())
    }
}

fn graph_hooks<'py>(hooks: &PyBound<'py, PyAny>) -> PyResult<[PyBound<'py, PyAny>; 2]> {
    if let Ok(hooks) = hooks.cast_exact::<PyTuple>() {
        if hooks.len() == 2 {
            let flatten = hooks.get_item(0)?;
            let unflatten = hooks.get_item(1)?;
            if flatten.is_callable() && unflatten.is_callable() {
                return Ok([flatten, unflatten]);
            }
        }
    }
    Err(PyTypeError::new_err(
        "type.__graph__ must be a (flatten, unflatten) tuple of callables",
    ))
}

impl PyGraph {
    pub fn into_pyobject_bound<'py>(self, py: Python<'py>) -> PyResult<PyBound<'py, PyAny>> {
        match self {
            PyGraph::Leaf(leaf) => leaf.into_pyobject_bound(py),
            PyGraph::Tracer(tracer) => wrap_tracer(py, tracer),
            PyGraph::Tuple(children) => Ok(PyTuple::new(py, children)?.into_any()),
            PyGraph::List(children) => Ok(PyList::new(py, children)?.into_any()),
            PyGraph::Dict(children) => {
                let dict = PyDict::new(py);
                for (key, child) in children {
                    dict.set_item(key.key_object(py)?, child)?;
                }
                Ok(dict.into_any())
            }
            PyGraph::Custom(typ, children) => {
                let py_type = typ.resolve(py)?;
                let [_, unflatten] = graph_hooks(&py_type.getattr("__graph__")?)?;
                let items = children
                    .into_iter()
                    .map(|(key, child)| Ok((key.key_object(py)?, child.into_pyobject_bound(py)?)))
                    .collect::<PyResult<Vec<_>>>()?;
                Ok(unflatten.call1((items,))?)
            }
        }
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for PyGraph {
    type Error = PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        GraphParser::default().parse(&obj, 0)
    }
}

impl<'py> IntoPyObject<'py> for PyGraph {
    type Target = PyAny;
    type Output = PyBound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        self.into_pyobject_bound(py)
    }
}

use gnx::graph::*;
use gnx::util::LifetimeFree;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule, PyString, PyTuple, PyType};
use pyo3::Bound as PyBound;
use std::borrow::Cow;
use std::sync::Arc;

use crate::leaf::PyLeaf;
use crate::gnxlib::GraphContainer;

#[derive(Clone)]
pub struct TypeInfo {
    pub module: String,
    pub name: String,
}

impl TypeInfo {
    fn from_type(py_type: &PyBound<'_, PyType>) -> PyResult<Self> {
        Ok(Self {
            module: py_type.getattr("__module__")?.extract()?,
            name: py_type.getattr("__qualname__")?.extract()?,
        })
    }

    fn import<'py>(&self, py: Python<'py>) -> PyResult<PyBound<'py, PyAny>> {
        let module = PyModule::import(py, &self.module)?;
        let mut value = module.getattr(self.name.split('.').next().unwrap_or(&self.name))?;
        for part in self.name.split('.').skip(1) {
            value = value.getattr(part)?;
        }
        Ok(value)
    }
}

#[derive(Clone)]
#[allow(unused)]
pub struct DictKey(Arc<Py<PyAny>>, Key);

impl DictKey {
    pub fn as_ref<'r>(&'r self) -> KeyRef<'r> {
        match &self.1 {
            Key::Attr(name) => KeyRef::Attr(name.as_ref()),
            Key::DictKey(key) => KeyRef::DictKey(key.as_ref()),
            Key::DictIndex(index) => KeyRef::DictIndex(*index),
            Key::Index(index) => KeyRef::Index(*index),
        }
    }

    fn from_py(key: PyBound<'_, PyAny>) -> PyResult<Self> {
        let graph_key = if let Ok(key_str) = key.cast::<PyString>() {
            Key::DictKey(Cow::Owned(key_str.to_str()?.to_owned()))
        } else if let Ok(index) = key.extract::<i64>() {
            Key::DictIndex(index)
        } else {
            return Err(PyTypeError::new_err(
                "graph keys must be str or int-compatible values",
            ));
        };
        Ok(Self(Arc::new(key.unbind()), graph_key))
    }

    fn key_object<'py>(&self, py: Python<'py>) -> PyBound<'py, PyAny> {
        self.0.bind(py).clone()
    }
}

#[derive(Clone)]
pub enum PyGraph {
    Leaf(PyLeaf),
    Tuple(Vec<PyGraph>),
    List(Vec<PyGraph>),
    Dict(Vec<(DictKey, PyGraph)>),
    Dataclass(TypeInfo, Vec<(String, PyGraph)>),
    Custom(TypeInfo, Vec<(DictKey, PyGraph)>),
    Shared(Arc<PyGraph>),
}
#[derive(Clone)]
pub enum PyBuilder {
    Leaf,
    Static(PyLeaf),
    Tuple(Vec<PyBuilder>),
    List(Vec<PyBuilder>),
    Dict(Vec<(DictKey, PyBuilder)>),
    Dataclass(TypeInfo, Vec<(String, PyBuilder)>),
    Custom(TypeInfo, Vec<(DictKey, PyBuilder)>),
    Shared(Arc<PyBuilder>),
}
unsafe impl LifetimeFree for PyGraph {}

#[rustfmt::skip]
impl Graph for PyGraph {
    type Owned = Self;
    type Builder<L: Leaf> = PyBuilder;

    fn builder<L: Leaf, F: Filter<L>, E: Error>(
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
                Tuple(children) => PyBuilder::Tuple(children.iter().enumerate().map(|(i, child)| {
                    child.builder(filter.child(KeyRef::Index(i)), &mut ctx)
                }).collect::<Result<Vec<PyBuilder>, E>>()?),
                List(children) => PyBuilder::List(children.iter().enumerate().map(|(i, child)| {
                    child.builder(filter.child(KeyRef::Index(i)), &mut ctx)
                }).collect::<Result<Vec<PyBuilder>, E>>()?),
                Dict(children) => PyBuilder::Dict(children.iter().map(|(key, child)| {
                    Ok((key.clone(), child.builder(filter.child(key.as_ref()), &mut ctx)?))
                }).collect::<Result<Vec<(DictKey, PyBuilder)>, E>>()?),
                Dataclass(typ, children) => PyBuilder::Dataclass(typ.clone(), children.iter().map(|(key, child)| {
                    Ok((key.clone(), child.builder(filter.child(KeyRef::Attr(key)), &mut ctx)?))
                }).collect::<Result<Vec<(String, PyBuilder)>, E>>()?),
                Custom(typ, children) => PyBuilder::Custom(typ.clone(), children.iter().map(|(key, child)| {
                    Ok((key.clone(), child.builder(filter.child(key.as_ref()), &mut ctx)?))
                }).collect::<Result<Vec<(DictKey, PyBuilder)>, E>>()?),
                Shared(g) => PyBuilder::Shared(g.builder(filter, ctx)?),
            }
        })
    }

    fn replace<'g, L: Leaf, F: Filter<L>, S: GraphSource<L::Ref<'g>, L>>(
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
                Tuple(children) => Tuple({
                    let mut ns = source.node()?;
                    children.iter().enumerate().map(|(i, child)| child.replace(
                            filter.child(KeyRef::Index(i)), ns.expect_child(KeyRef::Index(i))?, ctx
                    )).collect::<Result<Vec<PyGraph>, S::Error>>()?
                }),
                List(children) => List({
                    let mut ns = source.node()?;
                    children.iter().enumerate().map(|(i, child)| child.replace(
                        filter.child(KeyRef::Index(i)), ns.expect_child(KeyRef::Index(i))?, ctx
                    )).collect::<Result<Vec<PyGraph>, S::Error>>()?
                }),
                Dict(children) => Dict({
                    let mut ns = source.node()?;
                    children.iter().map(|(key, child)| {
                        let child = child.replace(filter.child(key.as_ref()), ns.expect_child(key.as_ref())?, ctx)?;
                        Ok((key.clone(), child))
                    }).collect::<Result<Vec<(DictKey, PyGraph)>, S::Error>>()?
                }),
                Dataclass(typ, children) => Dataclass(typ.clone(), {
                    let mut ns = source.node()?;
                    children.iter().map(|(key, child)| {
                        let key_ref = KeyRef::Attr(key);
                        let child = child.replace(filter.child(key_ref), ns.expect_child(key_ref)?, ctx)?;
                        Ok((key.clone(), child))
                    }).collect::<Result<Vec<(String, PyGraph)>, S::Error>>()?
                }),
                Custom(typ, children) => Custom(typ.clone(), {
                    let mut ns = source.node()?;
                    children.iter().map(|(key, child)| {
                        let child = child.replace(filter.child(key.as_ref()), ns.expect_child(key.as_ref())?, ctx)?;
                        Ok((key.clone(), child))
                    }).collect::<Result<Vec<(DictKey, PyGraph)>, S::Error>>()?
                }),
                Shared(g) => Shared(g.replace(filter, source, ctx)?),
            }
        })
    }

    fn visit<'g, L: Leaf, F: Filter<L>, V: GraphVisitor<'g, Self, L>>(
        &'g self,
        filter: F,
        visitor: V,
    ) -> V::Output {
        match filter.matches_ref(self) {
            Ok(r) => visitor.visit_leaf(r),
            Err(graph) => match graph {
                PyGraph::Leaf(leaf) => match filter.matches_ref(leaf) {
                    Ok(r) => visitor.visit_leaf(r),
                    Err(s) => visitor.visit_static::<PyLeaf>(s.as_ref()),
                },
                PyGraph::Shared(g) => visitor.visit_shared(
                    GraphId::from(Arc::as_ptr(g) as usize as u64),
                    View::new(g.as_ref(), filter),
                ),
                _ => visitor.visit_node(View::new(graph, filter)),
            },
        }
    }
    fn visit_into<L: Leaf, F: Filter<L>, C: GraphConsumer<Self, L>>(
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
                PyGraph::Shared(g) => consumer.consume_shared(
                    GraphId::from(Arc::as_ptr(&g) as usize as u64),
                    View::new(g.as_ref(), filter),
                ),
                _ => consumer.consume_node(gnx::graph::Bound::new(graph, filter)),
            },
        }
    }
}

impl TypedGraph<PyLeaf> for PyGraph {}

impl Node for PyGraph {
    fn visit_children<'g, L, F, V>(&'g self, filter: F, mut visitor: V) -> V::Output
    where
        L: Leaf,
        F: Filter<L>,
        V: ChildrenVisitor<'g, Self, L>,
    {
        match self {
            PyGraph::Tuple(children) | PyGraph::List(children) => {
                children.iter().enumerate().for_each(|(i, child)| {
                    let key = KeyRef::Index(i);
                    visitor.visit_child(key, View::new(child, filter.child(key)));
                });
            }
            PyGraph::Dict(children) | PyGraph::Custom(_, children) => {
                children.iter().for_each(|(key, child)| {
                    visitor.visit_child(key.as_ref(), View::new(child, filter.child(key.as_ref())));
                });
            }
            PyGraph::Dataclass(_, children) => {
                children.iter().for_each(|(key, child)| {
                    let key = KeyRef::Attr(key);
                    visitor.visit_child(key, View::new(child, filter.child(key)));
                });
            }
            PyGraph::Leaf(_) | PyGraph::Shared(_) => {}
        }
        visitor.finish()
    }

    fn visit_into_children<L, F, C>(self, filter: F, mut consumer: C) -> C::Output
    where
        L: Leaf,
        F: Filter<L>,
        C: ChildrenConsumer<Self, L>,
    {
        match self {
            PyGraph::Tuple(children) | PyGraph::List(children) => {
                children.into_iter().enumerate().for_each(|(i, child)| {
                    consumer.consume_child(Key::Index(i), gnx::graph::Bound::new(child, filter.child(KeyRef::Index(i))));
                });
            }
            PyGraph::Dict(children) | PyGraph::Custom(_, children) => {
                children.into_iter().for_each(|(key, child)| {
                    consumer.consume_child(key.as_ref().to_value(), gnx::graph::Bound::new(child, filter.child(key.as_ref())));
                });
            }
            PyGraph::Dataclass(_, children) => {
                children.into_iter().for_each(|(key, child)| {
                    consumer.consume_child(Key::Attr(key.clone().into()), gnx::graph::Bound::new(child, filter.child(KeyRef::Attr(&key))));
                });
            }
            PyGraph::Leaf(_) | PyGraph::Shared(_) => {}
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
            Tuple(children) => PyGraph::Tuple(build_indexed(children, source, ctx)?),
            List(children) => PyGraph::List(build_indexed(children, source, ctx)?),
            Dict(children) => PyGraph::Dict(build_keyed(children, source, ctx)?),
            Dataclass(typ, children) => PyGraph::Dataclass(typ, build_attrs(children, source, ctx)?),
            Custom(typ, children) => PyGraph::Custom(typ, build_keyed(children, source, ctx)?),
            Shared(builder) => PyGraph::Shared(builder.build(source, ctx)?),
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
        .map(|(i, child)| child.build(ns.expect_child(KeyRef::Index(i))?, ctx))
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

fn build_attrs<L: Leaf, S: GraphSource<(), L>>(
    children: Vec<(String, PyBuilder)>,
    source: S,
    ctx: &mut GraphContext,
) -> Result<Vec<(String, PyGraph)>, S::Error> {
    let mut ns = source.node()?;
    children
        .into_iter()
        .map(|(key, child)| {
            let child = child.build(ns.expect_child(KeyRef::Attr(&key))?, ctx)?;
            Ok((key, child))
        })
        .collect()
}

impl PyGraph {
    fn from_bound(obj: &PyBound<'_, PyAny>) -> PyResult<Self> {
        if obj.is_none() {
            return Ok(PyGraph::Leaf(PyLeaf::None));
        }
        if let Ok(container) = obj.cast::<GraphContainer>() {
            return Ok(PyGraph::Shared(container.borrow().0.clone()));
        }
        if let Ok(value) = obj.extract::<bool>() {
            return Ok(PyGraph::Leaf(PyLeaf::Bool(value)));
        }
        if let Ok(value) = obj.extract::<i64>() {
            return Ok(PyGraph::Leaf(PyLeaf::Int(value)));
        }
        if let Ok(value) = obj.extract::<f64>() {
            return Ok(PyGraph::Leaf(PyLeaf::Float(value)));
        }
        if let Ok(value) = obj.extract::<crate::bytes::ImBytes>() {
            return Ok(PyGraph::Leaf(PyLeaf::Bytes(value)));
        }
        if let Ok(value) = obj.extract::<crate::string::ImString>() {
            return Ok(PyGraph::Leaf(PyLeaf::String(value)));
        }
        if let Ok(tuple) = obj.cast::<PyTuple>() {
            return tuple
                .iter()
                .map(|child| PyGraph::from_bound(&child))
                .collect::<PyResult<Vec<_>>>()
                .map(PyGraph::Tuple);
        }
        if let Ok(list) = obj.cast::<PyList>() {
            return list
                .iter()
                .map(|child| PyGraph::from_bound(&child))
                .collect::<PyResult<Vec<_>>>()
                .map(PyGraph::List);
        }
        if let Ok(dict) = obj.cast::<PyDict>() {
            return dict
                .iter()
                .map(|(key, value)| Ok((DictKey::from_py(key)?, PyGraph::from_bound(&value)?)))
                .collect::<PyResult<Vec<_>>>()
                .map(PyGraph::Dict);
        }

        let py_type = obj.get_type();
        if let Ok(graph_attr) = py_type.getattr("__graph__") {
            let graph = graph_attr.cast_into::<PyTuple>()?;
            if graph.len() != 2 {
                return Err(PyTypeError::new_err("type.__graph__ must be a (flatten, unflatten) tuple"));
            }
            let flatten = graph.get_item(0)?;
            let flattened = flatten.call1((obj,))?;
            let children = flattened
                .try_iter()?
                .map(|item| {
                    let item = item?;
                    let pair = item.cast_into::<PyTuple>()?;
                    if pair.len() != 2 {
                        return Err(PyTypeError::new_err(
                            "flatten must return a sequence of (key, child) tuples",
                        ));
                    }
                    Ok((
                        DictKey::from_py(pair.get_item(0)?)?,
                        PyGraph::from_bound(&pair.get_item(1)?)?,
                    ))
                })
                .collect::<PyResult<Vec<_>>>()?;
            return Ok(PyGraph::Custom(TypeInfo::from_type(&py_type)?, children));
        }

        let dataclasses = PyModule::import(obj.py(), "dataclasses")?;
        if !obj.is_instance_of::<PyType>()
            && dataclasses
                .getattr("is_dataclass")?
                .call1((obj,))?
                .extract::<bool>()?
        {
            let fields = dataclasses.getattr("fields")?.call1((obj,))?;
            let children = fields
                .try_iter()?
                .map(|field| {
                    let field = field?;
                    let name: String = field.getattr("name")?.extract()?;
                    Ok((name.clone(), PyGraph::from_bound(&obj.getattr(name)?)?))
                })
                .collect::<PyResult<Vec<_>>>()?;
            return Ok(PyGraph::Dataclass(TypeInfo::from_type(&py_type)?, children));
        }

        Ok(PyGraph::Leaf(PyLeaf::Other(crate::leaf::ObjectHandle(Arc::new(
            obj.as_unbound().clone_ref(obj.py()),
        )))))
    }

    pub fn into_pyobject_bound<'py>(self, py: Python<'py>) -> PyResult<PyBound<'py, PyAny>> {
        match self {
            PyGraph::Leaf(leaf) => leaf.into_pyobject_bound(py),
            PyGraph::Tuple(children) => Ok(PyTuple::new(py, children)?.into_any()),
            PyGraph::List(children) => Ok(PyList::new(py, children)?.into_any()),
            PyGraph::Dict(children) => {
                let dict = PyDict::new(py);
                for (key, child) in children {
                    dict.set_item(key.key_object(py), child)?;
                }
                Ok(dict.into_any())
            }
            PyGraph::Dataclass(typ, children) => {
                let py_type = typ.import(py)?;
                let dataclasses = PyModule::import(py, "dataclasses")?;
                let fields = dataclasses.getattr("fields")?.call1((&py_type,))?;
                let kwargs = PyDict::new(py);
                let deferred = PyList::empty(py);

                for field in fields.try_iter()? {
                    let field = field?;
                    let name: String = field.getattr("name")?.extract()?;
                    if let Some((_, child)) = children.iter().find(|(key, _)| key == &name) {
                        if field.getattr("init")?.extract::<bool>()? {
                            kwargs.set_item(&name, child.clone())?;
                        } else {
                            deferred.append((name, child.clone()))?;
                        }
                    }
                }

                let instance = py_type.call((), Some(&kwargs))?;
                let object_type = PyModule::import(py, "builtins")?.getattr("object")?;
                let setattr = object_type.getattr("__setattr__")?;
                for item in deferred.iter() {
                    let pair = item.cast_into::<PyTuple>()?;
                    setattr.call1((&instance, pair.get_item(0)?, pair.get_item(1)?))?;
                }
                Ok(instance)
            }
            PyGraph::Custom(typ, children) => {
                let py_type = typ.import(py)?;
                let graph = py_type.getattr("__graph__")?.cast_into::<PyTuple>()?;
                if graph.len() != 2 {
                    return Err(PyTypeError::new_err("type.__graph__ must be a (flatten, unflatten) tuple"));
                }
                let unflatten = graph.get_item(1)?;
                let items = children
                    .into_iter()
                    .map(|(key, child)| Ok((key.key_object(py), child.into_pyobject_bound(py)?)))
                    .collect::<PyResult<Vec<_>>>()?;
                Ok(unflatten.call1((items,))?)
            }
            PyGraph::Shared(graph) => graph.as_ref().clone().into_pyobject_bound(py),
        }
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for PyGraph {
    type Error = PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        PyGraph::from_bound(&obj)
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
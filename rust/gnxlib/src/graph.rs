use gnx::graph::*;
use gnx::util::LifetimeFree;
use pyo3::prelude::*;
use std::sync::Arc;

use crate::leaf::PyLeaf;
use crate::string::ImString;

#[derive(Clone)]
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
}

#[derive(Clone)]
pub enum PyGraph {
    Leaf(PyLeaf),
    Tuple(Vec<PyGraph>),
    List(Vec<PyGraph>),
    Dict(Vec<(DictKey, PyGraph)>),
    Shared(Arc<PyGraph>),
}
#[derive(Clone)]
pub enum PyBuilder {
    Leaf,
    Static(PyLeaf),
    Tuple(Vec<PyBuilder>),
    List(Vec<PyBuilder>),
    Dict(Vec<(DictKey, PyBuilder)>),
    Shared(Arc<PyBuilder>),
}
unsafe impl LifetimeFree for PyGraph {}

#[rustfmt::skip]
impl Graph for PyGraph {
    type Owned = Self;
    type Builder<L: Leaf> = PyBuilder;

    fn builder<L: Leaf, F: Filter<L>>(
            &self, filter: F, mut ctx: &mut GraphContext
    ) -> Result<Self::Builder<L>, GraphError> {
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
                }).collect::<Result<Vec<PyBuilder>, GraphError>>()?),
                _ => panic!()
            }
        })
    }

    fn replace<'g, L: Leaf, F: Filter<L>, S: GraphSource<L::Ref<'g>, L>>(
        &'g self, filter: F, source: S, ctx: &mut GraphContext
    ) -> Result<Self::Owned, S::Error> {
        use PyGraph::*;
        Ok(match filter.matches_ref(self) {
            Ok(r) => L::try_into_value(source.leaf(r)?).map_err(|_| GraphError::InvalidLeaf)?,
            Err(graph) => match graph {
                Leaf(leaf) => Leaf(match filter.matches_ref(leaf) {
                    Ok(_) => leaf.replace(filter, source, ctx)?,
                    Err(_) => leaf.clone(),
                }),
                Tuple(children) => Tuple({
                    let mut ns = source.node()?;
                    children.iter().enumerate().map(|(i, child)| child.replace(
                            filter.child(KeyRef::Index(i)), ns.child(KeyRef::Index(i))?, ctx
                    )).collect::<Result<Vec<PyGraph>, S::Error>>()?
                }),
                List(children) => List({
                    let mut ns = source.node()?;
                    children.iter().enumerate().map(|(i, child)| child.replace(
                        filter.child(KeyRef::Index(i)), ns.child(KeyRef::Index(i))?, ctx
                    )).collect::<Result<Vec<PyGraph>, S::Error>>()?
                }),
                Dict(children) => Dict({
                    let mut ns = source.node()?;
                    children.iter().map(|(key, child)| {
                        let child = child.replace(filter.child(key.as_ref()), ns.child(key.as_ref())?, ctx)?;
                        Ok((key.clone(), child))
                    }).collect::<Result<Vec<(DictKey, PyGraph)>, S::Error>>()?
                }),
                Shared(_g) => todo!()
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
            Err(_) => todo!()
        }
    }
    fn visit_into<L: Leaf, F: Filter<L>, C: GraphConsumer<Self, L>>(
        self,
        filter: F,
        consumer: C,
    ) -> C::Output {
        match filter.matches_value(self) {
            Ok(v) => consumer.consume_leaf(v),
            Err(_) => todo!()
        }
    }
}

impl<L: Leaf> Builder<L> for PyBuilder {
    type Graph = PyGraph;

    fn build<S: GraphSource<(), L>>(
        self,
        _source: S,
        _ctx: &mut GraphContext,
    ) -> Result<Self::Graph, S::Error> {
        todo!()
    }
}

// use gnx::graph::{GraphId, Key};
// use gnx::{Array, ArrayRef, ImBytes, ImString};
// use num_bigint::BigInt;
// use pyo3::exceptions::PyTypeError;
// use std::collections::{HashMap, HashSet};
// use std::sync::Arc;

// use pyo3::prelude::*;

// // We can convert a python object to a PyGraph and back.
// // PyGraph, in turn, supports the Graph trait
// #[derive(Debug, Clone)]
// pub enum PyLeaf {
//     None,
//     Int(BigInt),
//     Float(f64),
//     Bool(bool),
//     String(ImString),
//     Bytes(ImBytes),
//     Array(Array),
//     ArrayRef(ArrayRef),
// }

// #[derive(Debug, Clone)]
// #[rustfmt::skip]
// pub enum PyType {
//     None, Int,
//     Float, Bool,
//     String, Bytes,
//     Array, ArrayRef,
//     Tuple, List, Dict,
//     NamedTuple(Arc<Py<PyAny>>),
//     Dataclass(Arc<Py<PyAny>>),
// }

// #[derive(Debug, Clone)]
// #[rustfmt::skip]
// pub struct PyNodeDef {
//     id: Option<GraphId>,
//     ty: PyType,
//     // None for leaf nodes
//     children: Option<Vec<(Key<'static>, PyGraphDef)>>,
// }

// #[derive(Clone)]
// pub struct PyGraphDef(Arc<PyNodeDef>);

// #[derive(Debug, Clone)]
// pub struct PyGraph {
//     def: PyGraphDef,
//     leaves: Vec<PyLeaf>,
// }

// impl std::fmt::Display for PyLeaf {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         use PyLeaf::*;
//         match self {
//             None => write!(f, "None"),
//             Int(v) => write!(f, "{}", v),
//             Float(v) => write!(f, "{}", v),
//             Bool(v) => write!(f, "{}", v),
//             String(v) => write!(f, "{:?}", v),
//             Bytes(v) => write!(f, "{:?}", v),
//             Array(v) => write!(f, "{:?}", v),
//             ArrayRef(v) => write!(f, "{:?}", v),
//         }
//     }
// }

// impl std::fmt::Display for PyGraph {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         write!(f, "{} {:?}", self.def, self.leaves)
//     }
// }

// impl PyType {
//     pub fn of_leaf(leaf: &PyLeaf) -> Self {
//         use PyLeaf::*;
//         match leaf {
//             None => PyType::None,
//             Int(_) => PyType::Int,
//             Float(_) => PyType::Float,
//             Bool(_) => PyType::Bool,
//             String(_) => PyType::String,
//             Bytes(_) => PyType::Bytes,
//             Array(_) => PyType::Array,
//             ArrayRef(_) => PyType::ArrayRef,
//         }
//     }
// }

// impl std::fmt::Debug for PyGraphDef {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         write!(f, "{:?}", self.0.as_ref())
//     }
// }

// impl std::fmt::Display for PyGraphDef {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         let def = self.0.as_ref();
//         match def.id {
//             Some(id) => write!(f, "{}: {:?}", id, def.ty)?,
//             None => write!(f, "{:?}", def.ty)?,
//         }
//         if let Some(children) = &def.children {
//             if !children.is_empty() {
//                 write!(f, " {{")?;
//                 for (key, child) in children {
//                     write!(f, "\n  {} -> {}", key, child)?;
//                 }
//                 write!(f, "\n}}")?;
//             }
//         }
//         Ok(())
//     }
// }

// impl PyGraphDef {
//     pub fn leaf(id: Option<GraphId>, ty: PyType) -> Self {
//         PyGraphDef(Arc::new(PyNodeDef {
//             id,
//             ty,
//             children: None,
//         }))
//     }
//     pub fn id(&self) -> Option<GraphId> {
//         self.0.id
//     }

//     pub fn ty(&self) -> &PyType {
//         &self.0.ty
//     }

//     // Formatting with a specialized leaf handler

//     // Canonicalization of GraphIds to make them dense from 0..N

//     fn collect_canonical_ids(
//         &self,
//         seen: &mut HashSet<GraphId>,
//         map: &mut HashMap<GraphId, GraphId>,
//     ) {
//         let def = self.0.as_ref();
//         if let Some(id) = def.id {
//             if seen.contains(&id) && !map.contains_key(&id) {
//                 let new_id = (seen.len() as u64).into();
//                 map.insert(id, new_id);
//             }
//             seen.insert(id);
//         }
//         if let Some(children) = &def.children {
//             for (_key, child) in children {
//                 child.collect_canonical_ids(seen, map);
//             }
//         }
//     }

//     fn remap_ids(&self, remap: &HashMap<GraphId, GraphId>, discard_other: bool) -> Self {
//         let def = self.0.as_ref();
//         let new_id = if discard_other {
//             def.id.map(|id| remap.get(&id).map(|x| *x)).flatten()
//         } else {
//             def.id.map(|id| remap.get(&id).map(|x| *x).unwrap_or(id))
//         };
//         let new_children = def.children.as_ref().map(|children| {
//             children
//                 .iter()
//                 .map(|(key, child)| (key.clone(), child.remap_ids(remap, discard_other)))
//                 .collect()
//         });
//         PyGraphDef(Arc::new(PyNodeDef {
//             id: new_id,
//             ty: def.ty.clone(),
//             children: new_children,
//         }))
//     }

//     pub fn canonicalize(&self) -> Self {
//         let mut seen: HashSet<GraphId> = HashSet::new();
//         let mut remap: HashMap<GraphId, GraphId> = HashMap::new();
//         self.collect_canonical_ids(&mut seen, &mut remap);
//         self.remap_ids(&remap, true)
//     }

//     fn extract<'a, 'py>(
//         obj: Borrowed<'a, 'py, PyAny>,
//         ids: &mut HashMap<GraphId, PyGraphDef>,
//         leaves: &mut Vec<PyLeaf>,
//     ) -> Result<PyGraphDef, PyErr> {
//         let id = ((obj.as_ptr() as usize) as u64).into();
//         if let Some(existing) = ids.get(&id) {
//             return Ok(existing.clone());
//         }
//         let id = Some(id);
//         #[rustfmt::skip]
//         let leaf = || -> Option<PyLeaf> {
//             Some(if obj.is_none() { PyLeaf::None }
//             else if let Ok(v) = obj.extract::<bool>() { PyLeaf::Bool(v) }
//             else if let Ok(v) = obj.extract::<BigInt>() { PyLeaf::Int(v) }
//             else if let Ok(v) = obj.extract::<f64>() { PyLeaf::Float(v) }
//             else if let Ok(v) = obj.extract::<ImString>() { PyLeaf::String(v) }
//             else if let Ok(v) = obj.extract::<ImBytes>() { PyLeaf::Bytes(v) }
//             else if let Ok(v) = obj.extract::<Array>() { PyLeaf::Array(v) }
//             else if let Ok(v) = obj.extract::<ArrayRef>() { PyLeaf::ArrayRef(v) }
//             else { return None })
//         }();
//         if let Some(leaf) = leaf {
//             let ty = PyType::of_leaf(&leaf);
//             leaves.push(leaf);
//             return Ok(PyGraphDef::leaf(id, ty));
//         }
//         Err(PyTypeError::new_err("Unsupported type"))
//     }
// }

// impl<'a, 'py> FromPyObject<'a, 'py> for PyGraph {
//     type Error = PyErr;

//     #[rustfmt::skip]
//     fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
//         let mut ids: HashMap<GraphId, PyGraphDef> = HashMap::new();
//         let mut leaves: Vec<PyLeaf> = Vec::new();
//         let def = PyGraphDef::extract(obj, &mut ids, &mut leaves)?.canonicalize();
//         Ok(PyGraph { def, leaves })
//     }
// }

// impl<'py> IntoPyObject<'py> for &PyGraph {
//     type Target = PyAny;
//     type Output = Bound<'py, Self::Target>;
//     type Error = PyErr;
//     fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
//         Err(PyTypeError::new_err("Not implemented"))
//     }
// }

// impl<'py> IntoPyObject<'py> for PyGraph {
//     type Target = PyAny;
//     type Output = Bound<'py, Self::Target>;
//     type Error = PyErr;
//     fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
//         (&self).into_pyobject(py)
//     }
// }

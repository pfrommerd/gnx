use gnx::graph::*;
use gnx::util::LifetimeFree;
use pyo3::prelude::*;
use std::sync::Arc;

use crate::leaf::PyLeaf;

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
                _ => panic!()
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
#[rustfmt::skip]
use crate::graph::*;

use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone)]
pub enum Dag<I> {
    Leaf(Option<I>),
    Node(DagNode<I>),
}
#[derive(Clone)]
pub enum DagNode<I> {
    Map(HashMap<Key, DagChild<I>>),
    // For efficiency, we store
    // nodes with only sequential Key::Index
    // values as a Vec
    List(Vec<DagChild<I>>),
}
#[derive(Clone)]
pub enum DagChild<I> {
    Owned(Dag<I>),
    Shared(Arc<Dag<I>>),
}
impl<I> DagChild<I> {
    fn replace<'g, L: Leaf, F: Filter<L>, S: GraphSource<L::Ref<'g>, L>>(
        &'g self,
        filter: F,
        source: S,
        ctx: &mut GraphContext,
    ) -> Result<DagChild<I>, S::Error>
    where
        I: Leaf,
    {
        Ok(match self {
            DagChild::Owned(v) => DagChild::Owned(v.replace(filter, source, ctx)?),
            DagChild::Shared(v) => {
                let id = GraphId::from(Arc::as_ptr(v) as u64);
                DagChild::Shared(ctx.create(id, |ctx| -> Result<Arc<Dag<I>>, S::Error> {
                    Ok(Arc::new(v.as_ref().replace(filter, source, ctx)?))
                })?)
            }
        })
    }
    fn builder<L: Leaf, F: Filter<L>>(
        &self,
        filter: F,
        ctx: &mut GraphContext,
    ) -> Result<ChildBuilder<I>, GraphError>
    where
        I: Leaf,
    {
        Ok(match self {
            DagChild::Owned(v) => ChildBuilder::Owned(v.builder(filter, ctx)?),
            DagChild::Shared(v) => ChildBuilder::Shared({
                let id = GraphId::from(Arc::as_ptr(v) as u64);
                ctx.create(id, |ctx| Ok(Arc::new(v.as_ref().builder(filter, ctx)?)))?
            }),
        })
    }
}

#[rustfmt::skip]
impl<I: Leaf> Graph for Dag<I> {
    type Owned = Self;
    type Builder<L: Leaf> = DagBuilder<I>;

    fn replace<'g, L: Leaf, F: Filter<L>, S: GraphSource<L::Ref<'g>, L>>(
        &'g self,
        filter: F,
        source: S,
        ctx: &mut GraphContext,
    ) -> Result<Self::Owned, S::Error> {
        Ok(match filter.matches_ref(self) {
            Ok(r) => source
                .leaf(r)?
                .try_into_value()
                .map_err(|_| GraphError::InvalidLeaf)?,
            Err(graph) => match graph {
                Dag::Leaf(None) => {
                    source.empty_leaf()?;
                    Dag::Leaf(None)
                }
                Dag::Leaf(Some(x)) => Dag::Leaf(Some(x.replace(filter, source, ctx)?)),
                Dag::Node(DagNode::Map(children)) => {
                    let mut ns = source.node()?;
                    let children: Result<HashMap<Key, DagChild<I>>, S::Error> =
                        children.iter().map(|(k, v)| {
                            let child = v.replace(
                                filter.child(k.as_ref()), ns.child(k.as_ref())?, ctx
                            )?;
                            Ok((k.clone(), child))
                        }).collect();
                    Dag::Node(DagNode::Map(children?))
                }
                Dag::Node(DagNode::List(children)) => {
                    let mut ns = source.node()?;
                    let children: Result<Vec<DagChild<I>>, S::Error> =
                        children.iter().enumerate().map(|(i, v)| v.replace(
                            filter.child(KeyRef::Index(i)),
                            ns.child(KeyRef::Index(i))?,
                            ctx
                        )).collect();
                    Dag::Node(DagNode::List(children?))
                }
            },
        })
    }
    fn builder<L: Leaf, F: Filter<L>>(
        &self, filter: F, mut ctx: &mut GraphContext
    ) -> Result<Self::Builder<L>, GraphError> {
        Ok(match filter.matches_ref(self) {
            Ok(_) => DagBuilder::Leaf,
            Err(graph) => match graph {
                Dag::Leaf(None) => DagBuilder::Leaf,
                Dag::Leaf(Some(x)) => match filter.matches_ref(x) {
                    Ok(_) => DagBuilder::Leaf,
                    Err(_) => DagBuilder::Static(x.clone()),
                },
                Dag::Node(DagNode::Map(children)) => {
                    let children: Result<HashMap<Key, ChildBuilder<I>>, GraphError> =
                        children.iter().map(|(k, v)| {
                            let child = v.builder(filter.child(k.as_ref()), &mut ctx)?;
                            Ok((k.clone(), child))
                        }).collect();
                    DagBuilder::MapNode(children?)
                }
                Dag::Node(DagNode::List(children)) => {
                    let children: Result<Vec<ChildBuilder<I>>, GraphError> =
                        children.iter().enumerate().map(|(i, v)| v.builder(
                            filter.child(KeyRef::Index(i)),
                            &mut ctx
                        )).collect();
                    DagBuilder::ListNode(children?)
                }
            },
        })
    }
    fn visit<'g, L: Leaf, F: Filter<L>, V: GraphVisitor<'g, Self, L>>(
        &'g self, filter: F, visitor: V
    ) -> V::Output {
        match filter.matches_ref(self) {
            Ok(r) => visitor.visit_leaf(r),
            Err(graph) => match graph {
                Dag::Leaf(Some(x)) => match filter.matches_ref(x) {
                    Ok(r) => visitor.visit_leaf(r),
                    Err(_) => visitor.visit_static::<()>(&()),
                }
                Dag::Leaf(None) => visitor.visit_static::<()>(&()),
                Dag::Node(_) => visitor.visit_node(View::new(self, filter)),
            },
        }
    }
    fn visit_into<'g, L: Leaf, F: Filter<L>, C: GraphConsumer<Self, L>>(
        self, filter: F, consumer: C
    ) -> C::Output {
        match filter.matches_value(self) {
            Ok(r) => consumer.consume_leaf(r),
            Err(graph) => match graph {
                Dag::Leaf(Some(x)) => match filter.matches_value(x) {
                    Ok(r) => consumer.consume_leaf(r),
                    Err(x) => consumer.consume_static::<I>(x),
                }
                Dag::Leaf(None) => consumer.consume_static::<()>(()),
                Dag::Node(_) => consumer.consume_node(Bound::new(graph, filter)),
            },
        }
    }
}
impl<I: Leaf> Node for Dag<I> {
    fn visit_children<'g, L: Leaf, F: Filter<L>, V: ChildrenVisitor<'g, Self, L>>(
        &'g self,
        filter: F,
        visitor: V,
    ) -> V::Output {
        match self {
            Dag::Node(DagNode::Map(children)) => {
                let mut v = visitor;
                for (k, child) in children {
                    let child = match child {
                        DagChild::Owned(v) => v,
                        DagChild::Shared(v) => v.as_ref(),
                    };
                    v.visit_child(k.as_ref(), View::new(child, filter.child(k.as_ref())));
                }
                v.finish()
            }
            Dag::Node(DagNode::List(children)) => {
                let mut v = visitor;
                for (i, child) in children.iter().enumerate() {
                    let child = match child {
                        DagChild::Owned(v) => v,
                        DagChild::Shared(v) => v.as_ref(),
                    };
                    let k = KeyRef::Index(i);
                    v.visit_child(k, View::new(child, filter.child(k)));
                }
                v.finish()
            }
            _ => visitor.finish(),
        }
    }
    fn visit_into_children<L, F, C>(self, filter: F, consumer: C) -> C::Output
    where
        L: Leaf,
        F: Filter<L>,
        C: ChildrenConsumer<Self, L>,
    {
        match self {
            Dag::Node(DagNode::Map(children)) => {
                let mut c = consumer;
                for (k, child) in children {
                    let sk = k.clone();
                    let f = filter.child(sk.as_ref());
                    match child {
                        DagChild::Owned(v) => c.consume_child(k, Bound::new(v, f)),
                        DagChild::Shared(v) => c.consume_child(k, Bound::new(v.as_ref(), f)),
                    };
                }
                c.finish()
            }
            Dag::Node(DagNode::List(children)) => {
                let mut c = consumer;
                for (i, child) in children.into_iter().enumerate() {
                    let (k, kr) = (Key::Index(i), KeyRef::Index(i));
                    match child {
                        DagChild::Owned(v) => c.consume_child(k, Bound::new(v, filter.child(kr))),
                        DagChild::Shared(v) => {
                            c.consume_child(k, Bound::new(v.as_ref(), filter.child(kr)))
                        }
                    };
                }
                c.finish()
            }
            _ => consumer.finish(),
        }
    }
}
impl<I: Leaf> TypedGraph<I> for Dag<I> {}

#[derive(Clone)]
pub enum DagBuilder<I> {
    Leaf,
    Static(I),
    MapNode(HashMap<Key, ChildBuilder<I>>),
    ListNode(Vec<ChildBuilder<I>>),
}
#[derive(Clone)]
pub enum ChildBuilder<I> {
    Owned(DagBuilder<I>),
    Shared(Arc<DagBuilder<I>>),
}
impl<I: Leaf> ChildBuilder<I> {
    fn build<L: Leaf, S: GraphSource<(), L>>(
        self,
        source: S,
        ctx: &mut crate::graph::GraphContext,
    ) -> Result<DagChild<I>, S::Error> {
        Ok(match self {
            ChildBuilder::Owned(v) => DagChild::Owned(v.build(source, ctx)?),
            ChildBuilder::Shared(v) => {
                let id = GraphId::from(Arc::as_ptr(&v) as u64);
                DagChild::Shared(ctx.create(id, |ctx| -> Result<Arc<Dag<I>>, S::Error> {
                    Ok(Arc::new(v.as_ref().clone().build(source, ctx)?))
                })?)
            }
        })
    }
}

impl<L: Leaf, I: Leaf> Builder<L> for DagBuilder<I> {
    type Graph = Dag<I>;

    fn build<S: GraphSource<(), L>>(
        self,
        source: S,
        mut ctx: &mut crate::graph::GraphContext,
    ) -> Result<Self::Graph, S::Error> {
        use DagBuilder::*;
        match self {
            Leaf => Ok(Dag::Leaf(Some(
                source
                    .leaf(())?
                    .try_into_value()
                    .map_err(|_| GraphError::InvalidLeaf)?,
            ))),
            Static(v) => {
                source.empty_leaf()?;
                Ok(Dag::Leaf(Some(v)))
            }
            ListNode(children) => Ok(Dag::Node(DagNode::List({
                let mut ns = source.node()?;
                children
                    .into_iter()
                    .enumerate()
                    .map(|(i, child)| child.build(ns.child(KeyRef::Index(i))?, &mut ctx))
                    .collect::<Result<Vec<DagChild<I>>, S::Error>>()?
            }))),
            MapNode(children) => Ok(Dag::Node(DagNode::Map({
                let mut ns = source.node()?;
                children
                    .into_iter()
                    .map(|(k, child)| {
                        let child = child.build(ns.child(k.as_ref())?, &mut ctx)?;
                        Ok((k, child))
                    })
                    .collect::<Result<HashMap<Key, DagChild<I>>, S::Error>>()?
            }))),
        }
    }
}

// The conversion visitor
pub struct ToDag<M>(pub M);

impl<'g, G: Graph + ?Sized, L: Leaf, I, M> GraphVisitor<'g, G, L> for ToDag<M>
where
    M: FnMut(L::Ref<'_>) -> I,
{
    type Output = Dag<I>;

    fn visit_leaf(mut self, value: L::Ref<'_>) -> Self::Output {
        Dag::Leaf(Some((self.0)(value)))
    }
    fn visit_static<S: Leaf>(self, _: S::Ref<'_>) -> Self::Output {
        Dag::Leaf(None)
    }
    fn visit_node<N: Node, F: Filter<L>>(self, node: View<'_, N, F>) -> Self::Output {
        let mut ctx = HashMap::new();
        node.graph.visit_children(
            node.filter,
            ToDagChildren {
                map: self.0,
                map_children: HashMap::new(),
                seq_children: Vec::new(),
                ctx: &mut ctx,
            },
        )
    }
    fn visit_shared<S: Graph, F: Filter<L>>(
        self,
        _id: GraphId,
        shared: View<'_, S, F>,
    ) -> Self::Output {
        shared.visit(self)
    }
}

struct ToDagChildren<'ctx, M, I> {
    map: M,
    map_children: HashMap<Key, DagChild<I>>,
    seq_children: Vec<DagChild<I>>,
    ctx: &'ctx mut HashMap<GraphId, Arc<Dag<I>>>,
}

impl<'g, 'ctx, N: Node, L: Leaf, M, I> ChildrenVisitor<'g, N, L> for ToDagChildren<'ctx, M, I>
where
    M: FnMut(L::Ref<'_>) -> I,
{
    type Output = Dag<I>;

    fn visit_child<C: Graph, F: Filter<L>>(
        &mut self,
        key: KeyRef<'_>,
        child: View<'_, C, F>,
    ) -> &mut Self {
        let key = key.to_value();
        let child = child.visit(ToDagChild {
            map: &mut self.map,
            ctx: self.ctx,
        });
        if key == Key::Index(self.seq_children.len()) {
            self.seq_children.push(child);
        } else {
            self.map_children.insert(key, child);
        }
        self
    }
    fn finish(self) -> Self::Output {
        if self.map_children.is_empty() {
            Dag::Node(DagNode::List(self.seq_children))
        } else {
            let mut children = self.map_children;
            children.extend(
                self.seq_children
                    .into_iter()
                    .enumerate()
                    .map(|(i, x)| (Key::Index(i), x)),
            );
            Dag::Node(DagNode::Map(children))
        }
    }
}

struct ToDagChild<'ctx, M, I> {
    map: M,
    ctx: &'ctx mut HashMap<GraphId, Arc<Dag<I>>>,
}

impl<'g, 'ctx, G: Graph, L: Leaf, M, I> GraphVisitor<'g, G, L> for ToDagChild<'ctx, M, I>
where
    M: FnMut(L::Ref<'_>) -> I,
{
    type Output = DagChild<I>;

    fn visit_leaf(mut self, value: L::Ref<'_>) -> Self::Output {
        DagChild::Owned(Dag::Leaf(Some((self.map)(value))))
    }
    fn visit_static<S: Leaf>(self, _: S::Ref<'_>) -> Self::Output {
        DagChild::Owned(Dag::Leaf(None))
    }
    fn visit_node<N: Node, F: Filter<L>>(mut self, node: View<'_, N, F>) -> Self::Output {
        DagChild::Owned(node.graph.visit_children(
            node.filter,
            ToDagChildren {
                map: self.map,
                map_children: HashMap::new(),
                seq_children: Vec::new(),
                ctx: &mut self.ctx,
            },
        ))
    }
    fn visit_shared<S: Graph, F: Filter<L>>(
        mut self,
        id: GraphId,
        shared: View<'_, S, F>,
    ) -> Self::Output {
        if let Some(dag) = self.ctx.get(&id) {
            return DagChild::Shared(dag.clone());
        }
        let dag = shared.visit(ToDagChild {
            map: &mut self.map,
            ctx: self.ctx,
        });
        match dag {
            DagChild::Owned(v) => {
                let dag = Arc::new(v);
                self.ctx.insert(id, dag.clone());
                DagChild::Shared(dag)
            }
            DagChild::Shared(v) => {
                self.ctx.insert(id, v.clone());
                DagChild::Shared(v)
            }
        }
    }
}

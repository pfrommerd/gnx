use crate::core::*;

use super::OrderMap;

use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone)]
pub enum Dag<I> {
    Leaf(Option<I>),
    Node(DagNode<I>),
}

#[derive(Clone)]
pub enum DagNode<I> {
    Map(OrderMap<Key, DagChild<I>>),
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

/// A single leaf: `Dag::from(0)` is `dag!(0)`.
impl<I> From<I> for Dag<I> {
    fn from(value: I) -> Self {
        Dag::Leaf(Some(value))
    }
}

impl<I> Dag<I> {
    /// Map selected leaves, visiting each shared node once. Map children are
    /// visited in insertion order.
    pub fn map_leaves<O>(&self, mut map: impl FnMut(&I) -> O) -> Dag<O> {
        fn map_dag<I, O>(
            dag: &Dag<I>,
            map: &mut impl FnMut(&I) -> O,
            shared: &mut HashMap<*const Dag<I>, Arc<Dag<O>>>,
        ) -> Dag<O> {
            let mut child = |child: &DagChild<I>| match child {
                DagChild::Owned(dag) => DagChild::Owned(map_dag(dag, map, shared)),
                DagChild::Shared(dag) => {
                    let id = Arc::as_ptr(dag);
                    if let Some(mapped) = shared.get(&id) {
                        return DagChild::Shared(mapped.clone());
                    }
                    let mapped = Arc::new(map_dag(dag, map, shared));
                    shared.insert(id, mapped.clone());
                    DagChild::Shared(mapped)
                }
            };
            match dag {
                Dag::Leaf(leaf) => Dag::Leaf(leaf.as_ref().map(map)),
                Dag::Node(DagNode::List(children)) => {
                    Dag::Node(DagNode::List(children.iter().map(child).collect()))
                }
                Dag::Node(DagNode::Map(children)) => Dag::Node(DagNode::Map(
                    children
                        .iter()
                        .map(|(key, value)| (key.clone(), child(value)))
                        .collect(),
                )),
            }
        }
        map_dag(self, &mut map, &mut HashMap::new())
    }
}

impl<I> DagChild<I> {
    fn replace<'g, L: Leaf, F: Filter<Leaf = L>, S: GraphSource<L::Ref<'g>, L>>(
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
    fn builder<L: Leaf, F: Filter<Leaf = L>, E: Error>(
        &self,
        filter: F,
        ctx: &mut GraphContext,
    ) -> Result<ChildBuilder<I>, E>
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

    fn replace<'g, L: Leaf, F: Filter<Leaf = L>, S: GraphSource<L::Ref<'g>, L>>(
        &'g self,
        filter: F,
        source: S,
        ctx: &mut GraphContext,
    ) -> Result<Self::Owned, S::Error> {
        Ok(match filter.matches_ref(self) {
            Ok(r) => source
                .leaf(r)?
                .try_into_value()
                .map_err(|_| S::Error::invalid_leaf())?,
            Err(graph) => match graph {
                Dag::Leaf(None) => {
                    source.empty_leaf()?;
                    Dag::Leaf(None)
                }
                Dag::Leaf(Some(x)) => Dag::Leaf(Some(x.replace(filter, source, ctx)?)),
                Dag::Node(DagNode::Map(children)) => {
                    let mut ns = source.node()?;
                    let children: Result<OrderMap<Key, DagChild<I>>, S::Error> =
                        children.iter().map(|(k, v)| {
                            let child = v.replace(
                                filter.child(k.as_ref()), ns.expect_child(k.as_ref())?, ctx
                            )?;
                            Ok((k.clone(), child))
                        }).collect();
                    Dag::Node(DagNode::Map(children?))
                }
                Dag::Node(DagNode::List(children)) => {
                    let mut ns = source.node()?;
                    let children: Result<Vec<DagChild<I>>, S::Error> =
                        children.iter().enumerate().map(|(i, v)| v.replace(
                            filter.child(KeyRef::Index(i as isize)),
                            ns.expect_child(KeyRef::Index(i as isize))?,
                            ctx
                        )).collect();
                    Dag::Node(DagNode::List(children?))
                }
            },
        })
    }
    fn builder<L: Leaf, F: Filter<Leaf = L>, E: Error>(
        &self, filter: F, mut ctx: &mut GraphContext
    ) -> Result<Self::Builder<L>, E> {
        Ok(match filter.matches_ref(self) {
            Ok(_) => DagBuilder::Leaf,
            Err(graph) => match graph {
                Dag::Leaf(None) => DagBuilder::Empty,
                Dag::Leaf(Some(x)) => match filter.matches_ref(x) {
                    Ok(_) => DagBuilder::Leaf,
                    Err(_) => DagBuilder::Static(x.clone()),
                },
                Dag::Node(DagNode::Map(children)) => {
                    let children: Result<OrderMap<Key, ChildBuilder<I>>, E> =
                        children.iter().map(|(k, v)| {
                            let child = v.builder(filter.child(k.as_ref()), &mut ctx)?;
                            Ok((k.clone(), child))
                        }).collect();
                    DagBuilder::MapNode(children?)
                }
                Dag::Node(DagNode::List(children)) => {
                    let children: Result<Vec<ChildBuilder<I>>, E> =
                        children.iter().enumerate().map(|(i, v)| v.builder(
                            filter.child(KeyRef::Index(i as isize)),
                            &mut ctx
                        )).collect();
                    DagBuilder::ListNode(children?)
                }
            },
        })
    }
    fn visit<'g, L: Leaf, F: Filter<Leaf = L>, V: GraphVisitor<'g, Self, L, F::Annotation<'g>>>(
        &'g self, filter: F, visitor: V
    ) -> V::Output {
        match filter.matches_ref(self) {
            Ok(r) => visitor.visit_leaf(r, filter),
            Err(graph) => match graph {
                Dag::Leaf(Some(x)) => match filter.matches_ref(x) {
                    Ok(r) => visitor.visit_leaf(r, filter),
                    Err(_) => visitor.visit_static::<()>(&()),
                }
                Dag::Leaf(None) => visitor.visit_static::<()>(&()),
                Dag::Node(_) => visitor.visit_node(View::new(self, filter)),
            },
        }
    }
    fn visit_into<'g, L: Leaf, F: Filter<Leaf = L>, C: GraphConsumer<Self, L>>(
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
                Dag::Node(_) => consumer.consume_node(Filtered::new(graph, filter)),
            },
        }
    }
}
impl<I: Leaf> Node for Dag<I> {
    fn visit_children<
        'g,
        L: Leaf,
        F: Filter<Leaf = L>,
        V: ChildrenVisitor<'g, Self, L, F::Annotation<'g>>,
    >(
        &'g self,
        filter: F,
        visitor: V,
    ) -> V::Output {
        match self {
            Dag::Node(DagNode::Map(children)) => {
                let mut v = visitor;
                for (k, child) in children {
                    visit_child(&mut v, k.as_ref(), child, filter.child(k.as_ref()));
                }
                v.finish()
            }
            Dag::Node(DagNode::List(children)) => {
                let mut v = visitor;
                for (i, child) in children.iter().enumerate() {
                    let k = KeyRef::Index(i as isize);
                    visit_child(&mut v, k, child, filter.child(k));
                }
                v.finish()
            }
            _ => visitor.finish(),
        }
    }
    fn visit_into_children<L, F, C>(self, filter: F, consumer: C) -> C::Output
    where
        L: Leaf,
        F: Filter<Leaf = L>,
        C: ChildrenConsumer<Self, L>,
    {
        match self {
            Dag::Node(DagNode::Map(children)) => {
                let mut c = consumer;
                for (k, child) in children {
                    let sk = k.clone();
                    let f = filter.child(sk.as_ref());
                    match child {
                        DagChild::Owned(v) => c.consume_child(k, Filtered::new(v, f)),
                        DagChild::Shared(v) => c.consume_child(k, Filtered::new(v.as_ref(), f)),
                    };
                }
                c.finish()
            }
            Dag::Node(DagNode::List(children)) => {
                let mut c = consumer;
                for (i, child) in children.into_iter().enumerate() {
                    let (k, kr) = (Key::from(i), KeyRef::from(i));
                    match child {
                        DagChild::Owned(v) => {
                            c.consume_child(k, Filtered::new(v, filter.child(kr)))
                        }
                        DagChild::Shared(v) => {
                            c.consume_child(k, Filtered::new(v.as_ref(), filter.child(kr)))
                        }
                    };
                }
                c.finish()
            }
            _ => consumer.finish(),
        }
    }
}

fn visit_child<
    'g,
    I: Leaf,
    L: Leaf,
    F: Filter<Leaf = L>,
    V: ChildrenVisitor<'g, Dag<I>, L, F::Annotation<'g>>,
>(
    visitor: &mut V,
    key: KeyRef<'g>,
    child: &'g DagChild<I>,
    filter: F,
) {
    match child {
        DagChild::Owned(node) => visitor.visit_child(key, View::new(node, filter)),
        DagChild::Shared(node) => visitor.visit_child(key, View::new(node, filter)),
    };
}

#[derive(Clone)]
pub enum DagBuilder<I> {
    Leaf,
    Empty,
    Static(I),
    MapNode(OrderMap<Key, ChildBuilder<I>>),
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
        ctx: &mut GraphContext,
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
        mut ctx: &mut GraphContext,
    ) -> Result<Self::Graph, S::Error> {
        use DagBuilder::*;
        match self {
            Leaf => Ok(Dag::Leaf(Some(
                source
                    .leaf(())?
                    .try_into_value()
                    .map_err(|_| S::Error::invalid_leaf())?,
            ))),
            Empty => {
                source.empty_leaf()?;
                Ok(Dag::Leaf(None))
            }
            Static(v) => {
                source.empty_leaf()?;
                Ok(Dag::Leaf(Some(v)))
            }
            ListNode(children) => Ok(Dag::Node(DagNode::List({
                let mut ns = source.node()?;
                children
                    .into_iter()
                    .enumerate()
                    .map(|(i, child)| child.build(ns.expect_child(KeyRef::from(i))?, &mut ctx))
                    .collect::<Result<Vec<DagChild<I>>, S::Error>>()?
            }))),
            MapNode(children) => Ok(Dag::Node(DagNode::Map({
                let mut ns = source.node()?;
                children
                    .into_iter()
                    .map(|(k, child)| {
                        let child = child.build(ns.expect_child(k.as_ref())?, &mut ctx)?;
                        Ok((k, child))
                    })
                    .collect::<Result<OrderMap<Key, DagChild<I>>, S::Error>>()?
            }))),
        }
    }
}

// The conversion visitor
pub struct ToDag<M>(pub M);

impl<'g, A, G: Graph + ?Sized, L: Leaf, I, M> GraphVisitor<'g, G, L, A> for ToDag<M>
where
    M: FnMut(L::Ref<'_>) -> I,
{
    type Output = Dag<I>;

    fn visit_leaf<F: Filter<Leaf = L, Annotation<'g> = A>>(
        mut self,
        value: L::Ref<'_>,
        _: F,
    ) -> Self::Output {
        Dag::Leaf(Some((self.0)(value)))
    }
    fn visit_static<S: Leaf>(self, _: S::Ref<'_>) -> Self::Output {
        Dag::Leaf(None)
    }
    fn visit_node<N: Node, F: Filter<Leaf = L, Annotation<'g> = A>>(
        mut self,
        node: View<'g, N, F>,
    ) -> Self::Output {
        let mut ctx = HashMap::new();
        node.graph.visit_children(
            node.filter,
            ToDagChildren {
                map: &mut self.0,
                children: DagNode::List(Vec::new()),
                ctx: &mut ctx,
            },
        )
    }
    fn visit_shared<S: Graph, F: Filter<Leaf = L, Annotation<'g> = A>>(
        self,
        _id: GraphId,
        shared: View<'g, S, F>,
    ) -> Self::Output {
        shared.visit(self)
    }
}

struct ToDagChildren<'ctx, M, I> {
    map: &'ctx mut M,
    children: DagNode<I>,
    ctx: &'ctx mut HashMap<GraphId, Arc<Dag<I>>>,
}

impl<'g, 'ctx, A, N: Node, L: Leaf, M, I> ChildrenVisitor<'g, N, L, A> for ToDagChildren<'ctx, M, I>
where
    M: FnMut(L::Ref<'_>) -> I,
{
    type Output = Dag<I>;

    fn visit_child<C: Graph, F: Filter<Leaf = L, Annotation<'g> = A>>(
        &mut self,
        key: KeyRef<'_>,
        child: View<'g, C, F>,
    ) -> &mut Self {
        let key = key.to_value();
        let child = child.visit(ToDagChild {
            map: self.map,
            ctx: self.ctx,
        });
        if let DagNode::List(children) = &mut self.children {
            if key == Key::from(children.len()) {
                children.push(child);
                return self;
            }
            self.children = DagNode::Map(
                std::mem::take(children)
                    .into_iter()
                    .enumerate()
                    .map(|(index, child)| (Key::from(index), child))
                    .collect(),
            );
        }
        let DagNode::Map(children) = &mut self.children else {
            unreachable!()
        };
        children.insert(key, child);
        self
    }
    fn finish(self) -> Self::Output {
        Dag::Node(self.children)
    }
}

struct ToDagChild<'ctx, M, I> {
    map: &'ctx mut M,
    ctx: &'ctx mut HashMap<GraphId, Arc<Dag<I>>>,
}

impl<'g, 'ctx, A, G: Graph, L: Leaf, M, I> GraphVisitor<'g, G, L, A> for ToDagChild<'ctx, M, I>
where
    M: FnMut(L::Ref<'_>) -> I,
{
    type Output = DagChild<I>;

    fn visit_leaf<F: Filter<Leaf = L, Annotation<'g> = A>>(
        self,
        value: L::Ref<'_>,
        _: F,
    ) -> Self::Output {
        DagChild::Owned(Dag::Leaf(Some((self.map)(value))))
    }
    fn visit_static<S: Leaf>(self, _: S::Ref<'_>) -> Self::Output {
        DagChild::Owned(Dag::Leaf(None))
    }
    fn visit_node<N: Node, F: Filter<Leaf = L, Annotation<'g> = A>>(
        self,
        node: View<'g, N, F>,
    ) -> Self::Output {
        DagChild::Owned(node.graph.visit_children(
            node.filter,
            ToDagChildren {
                map: self.map,
                children: DagNode::List(Vec::new()),
                ctx: self.ctx,
            },
        ))
    }
    fn visit_shared<S: Graph, F: Filter<Leaf = L, Annotation<'g> = A>>(
        self,
        id: GraphId,
        shared: View<'g, S, F>,
    ) -> Self::Output {
        if let Some(dag) = self.ctx.get(&id) {
            return DagChild::Shared(dag.clone());
        }
        let dag = shared.visit(ToDagChild {
            map: self.map,
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

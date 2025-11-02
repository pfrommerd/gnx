// Blanket Graph implementations for common rust types

use super::impl_leaf;

mod containers;

// Builtin types
impl_leaf!(u8);
// impl_leaf!(u16);
// impl_leaf!(u32);
// impl_leaf!(u64);

// impl_leaf!(i8);
// impl_leaf!(i16);
// impl_leaf!(i32);
// impl_leaf!(i64);

// impl_leaf!(f32);
// impl_leaf!(f64);

use super::*;
use std::borrow::Cow;

pub enum LeafDef<I, T, R>
where
    I: Clone + 'static,
    T: Graph + Clone + 'static,
    R: NonLeafRepr,
{
    Leaf(I),
    NonLeaf(R::NonLeaf<T>),
}

impl<I, T, R> Clone for LeafDef<I, T, R>
where
    I: Clone + 'static,
    T: Graph + Clone + 'static,
    R: NonLeafRepr,
{
    fn clone(&self) -> Self {
        match self {
            LeafDef::Leaf(i) => LeafDef::Leaf(i.clone()),
            LeafDef::NonLeaf(nl) => LeafDef::NonLeaf(nl.clone()),
        }
    }
}

impl<I, T, R> GraphDef<I> for LeafDef<I, T, R>
where
    I: Clone + 'static,
    T: Graph + Clone + 'static,
    R: NonLeafRepr,
{
    type Graph = T;

    fn build<L, B, S>(
        &self,
        builder: B,
        source: S,
        _ctx: &mut GraphContext,
    ) -> Result<Self::Graph, S::Error>
    where
        B: LeafBuilder<I, L>,
        S: GraphSource<I, L>,
    {
        match self {
            LeafDef::Leaf(leaf) => Ok(builder.try_build(source.leaf(Cow::Borrowed(leaf))?)?),
            LeafDef::NonLeaf(value) => Ok(R::try_from_repr(Cow::Borrowed(value))?),
        }
    }

    fn into_build<L, B, S>(
        self,
        builder: B,
        source: S,
        _ctx: &mut GraphContext,
    ) -> Result<Self::Graph, S::Error>
    where
        B: LeafBuilder<I, L>,
        S: GraphSource<I, L>,
    {
        match self {
            LeafDef::Leaf(leaf) => Ok(builder.try_build(source.leaf(Cow::Owned(leaf))?)?),
            LeafDef::NonLeaf(value) => Ok(R::try_from_repr(Cow::Owned(value))?),
        }
    }
}

#[derive(Clone)]
pub enum NodeDef<I, N>
where
    I: Clone + 'static,
    N: GraphDef<I>,
{
    Leaf(I),
    Node(N),
}

impl<I: Clone + 'static, N: GraphDef<I>> GraphDef<I> for NodeDef<I, N> {
    type Graph = N::Graph;

    fn build<L, B, S>(
        &self,
        builder: B,
        source: S,
        ctx: &mut GraphContext,
    ) -> Result<Self::Graph, S::Error>
    where
        B: LeafBuilder<I, L>,
        S: GraphSource<I, L>,
    {
        match self {
            NodeDef::Leaf(leaf) => Ok(builder.try_build(source.leaf(Cow::Borrowed(leaf))?)?),
            NodeDef::Node(node_def) => node_def.build(builder, source, ctx),
        }
    }

    fn into_build<L, B, S>(
        self,
        builder: B,
        source: S,
        ctx: &mut GraphContext,
    ) -> Result<Self::Graph, S::Error>
    where
        B: LeafBuilder<I, L>,
        S: GraphSource<I, L>,
    {
        match self {
            NodeDef::Leaf(leaf) => Ok(builder.try_build(source.leaf(Cow::Owned(leaf))?)?),
            NodeDef::Node(node_def) => node_def.into_build(builder, source, ctx),
        }
    }
}

// LeafDef and BoundDef themselves are graphs!
// This way we can use a graphdef to select axes to vmap over.
// let graph = ...; // some graph
// let target_axes = graph.graph_def(Of<Array>, |a| 0)
//
// zip(&mut graph, target_shapes).visit_mut(
//    LeafMap::where(Of<(Array, Shape)>, |(x, target_shape)| x.reshape(target_shape)
// ))

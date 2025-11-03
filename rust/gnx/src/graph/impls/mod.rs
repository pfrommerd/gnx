// Blanket Graph implementations for common rust types

use super::*;
use std::borrow::Cow;

// mod containers;
// mod rc;

mod invalid;
pub use invalid::Invalid;

// Builtin types
// impl_leaf!(u8);
// impl_leaf!(u16);
// impl_leaf!(u32);
// impl_leaf!(u64);

// impl_leaf!(i8);
// impl_leaf!(i16);
// impl_leaf!(i32);
// impl_leaf!(i64);

// impl_leaf!(f32);
// impl_leaf!(f64);

// A leafdef is a leaf of type I
// which potentially contains type R::NonLeaf<L>
// as a static value
#[derive(Clone)]
pub enum LeafDef<I: Leaf, R: StaticRepr, L: Leaf> {
    Leaf(I),
    Static(R::Repr<L>),
}

#[derive(Clone)]
pub enum NodeDef<I, N> {
    Leaf(I),
    Node(N),
}

#[rustfmt::skip]
impl<I: Leaf, R: StaticRepr, L: Leaf> GraphDef<I> for LeafDef<I, R, L> {
    type Graph = L;

    fn build<V: Value, S: GraphSource<I, V>>(
        &self, source: S, _ctx: &mut GraphContext,
    ) -> Result<Self::Graph, S::Error> {
        match self {
            LeafDef::Leaf(leaf) => {
                let value = source.leaf(LeafCow::Borrowed(leaf.as_ref()))?;
                Ok(value.convert_into().map_err(|_| GraphError::InvalidType)?)
            }
            LeafDef::Static(value) => Ok(R::try_from_repr(Cow::Borrowed(value))?),
        }
    }

    fn into_build<V: Value, S: GraphSource<I, V>>(
        self, source: S, _ctx: &mut GraphContext,
    ) -> Result<Self::Graph, S::Error> {
        match self {
            LeafDef::Leaf(info) => {
                let value = source.leaf(LeafCow::Owned(info))?;
                Ok(value.convert_into().map_err(|_| GraphError::InvalidType)?)
            },
            LeafDef::Static(value) => Ok(R::try_from_repr(Cow::Owned(value))?),
        }
    }
}

#[rustfmt::skip]
impl<I: Leaf, N: Node + GraphDef<I>> GraphDef<I> for NodeDef<I, N> {
    type Graph = N::Graph;

    fn build<V: Value, S: GraphSource<I, V>>(
        &self, source: S, ctx: &mut GraphContext,
    ) -> Result<Self::Graph, S::Error> {
        match self {
            NodeDef::Leaf(leaf) => {
                let value = source.leaf(LeafCow::Borrowed(leaf.as_ref()))?;
                Ok(value.convert_into::<N::Graph>().map_err(
                    |_| GraphError::InvalidType
                )?)
            }
            NodeDef::Node(node_def) => node_def.build(source, ctx)
        }
    }
    fn into_build<V: Value, S: GraphSource<I, V>>(
        self, source: S, ctx: &mut GraphContext,
    ) -> Result<Self::Graph, S::Error> {
        match self {
            NodeDef::Leaf(leaf) => {
                let value = source.leaf(LeafCow::Owned(leaf))?;
                Ok(value.convert_into::<N::Graph>().map_err(
                    |_| GraphError::InvalidType
                )?)
            }
            NodeDef::Node(node_def) => node_def.into_build(source, ctx),
        }
    }
}

// LeafDef and NodeDef themselves are graphs!

#[rustfmt::skip]
#[allow(unused)]
impl<I: Leaf, R: StaticRepr, S: Leaf> Graph for LeafDef<I, R, S> {
    // TODO: Cannot currently take a graphdef of a graphdef!
    // Would need LeafDef<I, R, S> itself to implement Leaf
    // and guarantee lifetime freedom
    type GraphDef<J: Leaf, RR: StaticRepr> = Invalid;
    type Owned = Invalid;

    fn graph_def<J: Leaf, L: Leaf, F, M>(
        &self, filter: F, map: impl Into<M>, ctx: &mut GraphContext,
    ) -> Result<Self::GraphDef<J, F::StaticRepr>, GraphError>
            where F: Filter<L>, M: FnMut(L::Ref<'_>) -> J {
        Err(GraphError::GraphDefUnsupported)
    }
    fn into_graph_def<J: Leaf, L: Leaf, F, M>(
        self, filter: F, map: impl Into<M>, ctx: &mut GraphContext,
    ) -> Result<Self::GraphDef<J, F::StaticRepr>, GraphError>
            where F: Filter<L>, M: FnMut(LeafCow<'_, L>) -> J {
        Err(GraphError::GraphDefUnsupported)
    }

    fn visit<L: Leaf, F: Filter<L>, V: GraphVisitor<L>>(
        &self, filter: F, visitor: impl Into<V>
    ) -> V::Output {
        visitor.into().visit_leaf(match self {
            LeafDef::Static(_) => None,
            LeafDef::Leaf(leaf) => L::try_as_ref(leaf).ok()
        })
    }
    fn mut_visit<L: Leaf, F: Filter<L>, M: GraphMutVisitor<L>>(
        &mut self, filter: F, visitor: impl Into<M>
    ) -> M::Output {
        visitor.into().visit_leaf_mut(match self {
            LeafDef::Static(_) => None,
            LeafDef::Leaf(leaf) => L::try_as_mut(leaf).ok()
        })
    }
    fn into_visit<L: Leaf, F: Filter<L>, C: GraphConsumer<L>>(
        self, filter: F, consumer: impl Into<C>
    ) -> C::Output {
        consumer.into().consume_leaf(match self {
            LeafDef::Static(_) => None,
            LeafDef::Leaf(leaf) => L::try_from_value(leaf).ok()
        })
    }
}
impl<I: Leaf, R: StaticRepr, S: Leaf> TypedGraph<I> for LeafDef<I, R, S> {}

#[rustfmt::skip]
impl<I: Leaf, N: Node + Graph,> Graph for NodeDef<I, N> {
    type GraphDef<J: Leaf, R: StaticRepr> = Invalid;
    type Owned = Invalid;

    #[allow(unused)]
    fn graph_def<J: Leaf, L: Leaf, F, M>(
        &self, filter: F, map: impl Into<M>, ctx: &mut GraphContext,
    ) -> Result<Self::GraphDef<J, F::StaticRepr>, GraphError>
            where F: Filter<L>, M: FnMut(L::Ref<'_>) -> J {
        Err(GraphError::GraphDefUnsupported)
    }

    #[allow(unused)]
    fn into_graph_def<J: Leaf, L: Leaf, F, M>(
        self, filter: F, map: impl Into<M>, ctx: &mut GraphContext,
    ) -> Result<Self::GraphDef<J, F::StaticRepr>, GraphError>
            where F: Filter<L>, M: FnMut(LeafCow<'_, L>) -> J {
        Err(GraphError::GraphDefUnsupported)
    }

    fn visit<L: Leaf, F: Filter<L>, V: GraphVisitor<L>>(
        &self, filter: F, visitor: impl Into<V>
    ) -> V::Output {
        match self {
            NodeDef::Leaf(l) => visitor.into().visit_leaf(L::try_as_ref(l).ok()),
            NodeDef::Node(n) => visitor.into().visit_node(View::new(n, filter)),
        }
    }
    fn mut_visit<L: Leaf, F: Filter<L>, M: GraphMutVisitor<L>>(
        &mut self, filter: F, visitor: impl Into<M>
    ) -> M::Output {
        match self {
            NodeDef::Leaf(l) => visitor.into().visit_leaf_mut(L::try_as_mut(l).ok()),
            NodeDef::Node(n) => visitor.into().visit_node_mut(ViewMut::new(n, filter)),
        }
    }
    fn into_visit<L: Leaf, F: Filter<L>, C: GraphConsumer<L>>(
        self, filter: F, consumer: impl Into<C>
    ) -> C::Output {
        match self {
            NodeDef::Leaf(l) => consumer.into().consume_leaf(L::try_from_value(l).ok()),
            NodeDef::Node(n) => consumer.into().consume_node(Bound::new(n, filter))
        }
    }
}

impl<I: Leaf, N> TypedGraph<I> for NodeDef<I, N> where N: Node + Graph {}

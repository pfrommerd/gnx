use crate::graph::KeyRef;

use super::{Graph, GraphError, Leaf, LeafCow};
use std::borrow::Cow;
use std::marker::PhantomData;

pub trait Filter<L: Leaf>: Clone {
    fn matches_ref<'g, G: Graph>(&self, graph: &'g G) -> Result<L::Ref<'g>, &'g G>;
    fn matches_mut<'g, G: Graph>(&self, graph: &'g mut G) -> Result<L::RefMut<'g>, &'g mut G>;
    fn matches_value<G: Graph>(&self, graph: G) -> Result<L, G>;

    // Construct a filter for a given child key
    type ChildFilter<'s>: Filter<L>
    where
        Self: 's;

    fn child<'s>(&'s self, key: KeyRef<'s>) -> Self::ChildFilter<'s>;
}

impl<L: Leaf, F: Filter<L>> Filter<L> for &F {
    fn matches_ref<'g, G: Graph>(&self, graph: &'g G) -> Result<L::Ref<'g>, &'g G> {
        (*self).matches_ref(graph)
    }
    fn matches_mut<'g, G: Graph>(&self, graph: &'g mut G) -> Result<L::RefMut<'g>, &'g mut G> {
        (*self).matches_mut(graph)
    }
    fn matches_value<G: Graph>(&self, graph: G) -> Result<L, G> {
        (*self).matches_value(graph)
    }

    type ChildFilter<'s>
        = F::ChildFilter<'s>
    where
        Self: 's;

    fn child<'s>(&'s self, key: KeyRef<'s>) -> Self::ChildFilter<'s> {
        (*self).child(key)
    }
}

// A non-leaf viewer knows how to handle
// leaves that cannot be coerced to type L
// This is generally a marker type, hence Copy + 'static
pub trait StaticRepr: Copy + 'static {
    // For graphdefs, a filter needs to know what to
    // do with leaves of type L that cannot be coerced
    // to the desired leaf type.
    type Repr<L: Leaf>: Clone + 'static;

    fn try_from_leaf<L: Leaf>(v: LeafCow<L>) -> Result<Self::Repr<L>, GraphError>;
    // Note that the NonLeaf type must be 'static, so
    // we need to be able to try and convert them back
    // without access to self (this conversion may fail!)
    fn try_from_repr<L: Leaf>(v: Cow<'_, Self::Repr<L>>) -> Result<L, GraphError>;
}

#[derive(Clone, Copy)]
pub struct CloneNonLeaf;

impl StaticRepr for CloneNonLeaf {
    type Repr<L: Leaf> = L;
    fn try_from_leaf<L: Leaf>(v: LeafCow<L>) -> Result<Self::Repr<L>, GraphError> {
        Ok(match v {
            LeafCow::Borrowed(b) => L::clone_ref(b),
            LeafCow::Owned(o) => o,
        })
    }
    fn try_from_repr<L: Leaf>(v: Cow<'_, Self::Repr<L>>) -> Result<L, GraphError> {
        match v {
            Cow::Borrowed(b) => Ok(b.clone()),
            Cow::Owned(o) => Ok(o),
        }
    }
}

#[derive(Clone, Copy)]
pub struct DiscardNonLeaf;

impl StaticRepr for DiscardNonLeaf {
    type Repr<L: Leaf> = ();
    fn try_from_leaf<L: Leaf>(_v: LeafCow<L>) -> Result<Self::Repr<L>, GraphError> {
        Ok(())
    }
    fn try_from_repr<L: Leaf>(_v: Cow<'_, Self::Repr<L>>) -> Result<L, GraphError> {
        Err(GraphError::ReprUnsupported)
    }
}

// Basic filter types

#[derive(Clone)]
pub struct OfType<L: Leaf>(PhantomData<L>);

impl<L: Leaf> OfType<L> {
    pub fn filter() -> Self {
        Self(PhantomData)
    }
}

impl<L: Leaf> Filter<L> for OfType<L> {
    fn matches_ref<'g, G: Graph>(&self, graph: &'g G) -> Result<<L as Leaf>::Ref<'g>, &'g G> {
        L::try_from_ref(graph)
    }
    fn matches_mut<'g, G: Graph>(
        &self,
        graph: &'g mut G,
    ) -> Result<<L as Leaf>::RefMut<'g>, &'g mut G> {
        L::try_from_mut(graph)
    }
    fn matches_value<G: Graph>(&self, graph: G) -> Result<L, G> {
        L::try_from_value(graph)
    }

    type ChildFilter<'s>
        = Self
    where
        Self: 's;

    fn child<'s>(&'s self, _key: crate::graph::KeyRef<'s>) -> Self::ChildFilter<'s> {
        self.clone()
    }
}

#[derive(Clone)]
pub struct IgnoreAll<L: Leaf>(PhantomData<L>);
impl<L: Leaf> IgnoreAll<L> {
    pub fn filter() -> Self {
        Self(PhantomData)
    }
}

#[rustfmt::skip]
impl<L: Leaf> Filter<L> for IgnoreAll<L> {
    fn matches_ref<'g, G: Graph>(&self, _graph: &'g G)
      -> Result<<L as Leaf>::Ref<'g>, &'g G> { Err(_graph) }
    fn matches_mut<'g, G: Graph>(&self, _graph: &'g mut G)
      -> Result<<L as Leaf>::RefMut<'g>, &'g mut G> { Err(_graph) }
    fn matches_value<G: Graph>(&self, _graph: G)
      -> Result<L, G> { Err(_graph) }

    type ChildFilter<'s>
        = Self
    where
        Self: 's;

    fn child<'s>(&'s self, _key: crate::graph::KeyRef<'s>) -> Self::ChildFilter<'s> {
        self.clone()
    }
}

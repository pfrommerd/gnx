use super::{Graph, GraphError};
use std::borrow::{Borrow, BorrowMut, Cow};

mod basic;

pub use basic::{NonLeafCloner, Of};

// Usually GraphFilter is implemented
// for the reference type
// (So that Bound has no lifetime parameters)
pub trait GraphFilter<L>: Copy {
    // A reference-like type for the viewer
    // For instance L might be Option<T> and Ref<'l> is Option<&'l T>
    // Thus a GraphFilter could coerce both T and Option<T> to &
    type Ref<'l>: Borrow<L>
    where
        L: 'l;
    type RefMut<'l>: BorrowMut<L>
    where
        L: 'l;
    // A viewer knows how to turn a reference to a leaf into a Self::Ref
    fn as_ref<'l>(&self, leaf: &'l L) -> Self::Ref<'l>;
    fn as_mut<'l>(&self, leaf: &'l mut L) -> Self::RefMut<'l>;

    fn try_as_ref<'g, G: Graph>(&self, graph: &'g G) -> Result<Self::Ref<'g>, &'g G>;
    fn try_as_mut<'g, G: Graph>(&self, graph: &'g mut G) -> Result<Self::RefMut<'g>, &'g mut G>;
    fn try_to_leaf<G: Graph>(&self, g: G) -> Result<L, G>;

    type NonLeafRepr: NonLeafRepr;
}

pub enum GraphCow<R, L>
where
    R: Borrow<L>,
{
    Borrowed(R),
    Owned(L),
}

// A non-leaf viewer knows how to handle
// leaves that cannot be coerced to type L
// This is generally a marker type, hence Copy + 'static
pub trait NonLeafRepr: 'static {
    // For graphdefs, a viewer needs to know what to
    // do with leaves of type T that cannot be coerced
    // to the leaf type L. These will be stored as NonLeaf<T>

    // Ideally we would be able to relax the Clone + 'static
    // on the T parameter so that we can handle
    // non-cloneable "static" fields
    type NonLeaf<T: Clone + 'static>: Into<T> + Clone + 'static;
    fn try_from_value<T: Clone + 'static>(v: Cow<'_, T>) -> Result<Self::NonLeaf<T>, GraphError>;
    // Note that the NonLeaf type must be 'static, so
    // we need to be able to try and convert them back
    // without access to self (this conversion may fail!)
    fn try_from_repr<T: Clone + 'static>(v: Cow<'_, Self::NonLeaf<T>>) -> Result<T, GraphError>;
}

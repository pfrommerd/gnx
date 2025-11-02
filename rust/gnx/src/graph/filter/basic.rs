use crate::graph::{Graph, GraphError, GraphFilter, NonLeafRepr};
use castaway::{LifetimeFree, cast};
use std::borrow::Cow;

pub struct Of<T: LifetimeFree>(std::marker::PhantomData<T>);
impl<T: LifetimeFree> Of<T> {
    fn new() -> Self {
        Self(std::marker::PhantomData)
    }
}
impl<T: LifetimeFree> Clone for Of<T> {
    fn clone(&self) -> Self {
        Self::new()
    }
}

impl<T: LifetimeFree> Copy for Of<T> {}

impl<T: LifetimeFree> GraphFilter<T> for Of<T> {
    type Ref<'r>
        = &'r T
    where
        T: 'r;
    type RefMut<'r>
        = &'r mut T
    where
        T: 'r;

    fn as_ref<'l>(&self, leaf: &'l T) -> Self::Ref<'l> {
        leaf
    }
    fn as_mut<'l>(&self, leaf: &'l mut T) -> Self::RefMut<'l> {
        leaf
    }

    fn try_as_ref<'g, G: Graph>(&self, graph: &'g G) -> Result<Self::Ref<'g>, &'g G> {
        cast!(graph, &T)
    }
    fn try_as_mut<'g, G: Graph>(&self, graph: &'g mut G) -> Result<Self::RefMut<'g>, &'g mut G> {
        cast!(graph, &mut T)
    }
    fn try_to_leaf<G: Graph>(&self, graph: G) -> Result<T, G> {
        cast!(graph, T)
    }
    type NonLeafRepr = NonLeafCloner;
}

pub struct NonLeafCloner;

impl NonLeafRepr for NonLeafCloner {
    type NonLeaf<T: Clone + 'static> = T;
    fn try_from_value<T: Clone + 'static>(v: Cow<'_, T>) -> Result<Self::NonLeaf<T>, GraphError> {
        match v {
            Cow::Borrowed(b) => Ok(b.clone()),
            Cow::Owned(o) => Ok(o),
        }
    }
    fn try_from_repr<T: Clone + 'static>(v: Cow<'_, Self::NonLeaf<T>>) -> Result<T, GraphError> {
        match v {
            Cow::Borrowed(b) => Ok(b.clone()),
            Cow::Owned(o) => Ok(o),
        }
    }
}

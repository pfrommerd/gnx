use std::cell::RefCell;
use std::rc::Rc;

use crate::{ChildrenSource, Error, GraphSource, KeyRef, SourceError};

/// A permissive source that supplies leaves in iterator order.
///
/// All child sources share one cursor. Only `leaf()` advances it; structure and
/// child keys are ignored. For reconstruction, supply the selected leaves in
/// visit order, visiting each shared node only on its first occurrence.
pub struct IterSource<T>(Rc<RefCell<T>>);

impl<T: Iterator> IterSource<T> {
    pub fn new(leaves: impl IntoIterator<IntoIter = T>) -> Self {
        Self(Rc::new(RefCell::new(leaves.into_iter())))
    }
}

impl<I, T: Iterator> GraphSource<I, T::Item> for IterSource<T> {
    type Error = SourceError;
    type ChildrenSource = Self;
    type Permissive = Self;

    fn leaf(self, _: I) -> Result<T::Item, SourceError> {
        self.0
            .borrow_mut()
            .next()
            .ok_or_else(SourceError::missing_leaf)
    }

    fn empty_leaf(self) -> Result<(), SourceError> {
        Ok(())
    }

    fn node(self) -> Result<Self, SourceError> {
        Ok(self)
    }

    fn permissive(self) -> Self {
        self
    }
}

impl<I, T: Iterator> ChildrenSource<I, T::Item> for IterSource<T> {
    type Error = SourceError;
    type ChildSource = Self;

    fn child(&mut self, _: KeyRef<'_>) -> Result<Option<Self>, SourceError> {
        Ok(Some(Self(self.0.clone())))
    }
}

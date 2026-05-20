use crate::{Graph, Leaf, KeyRef};
use std::marker::PhantomData;

pub trait Filter<L: Leaf>: Clone {
    type Owned: Filter<L> + 'static;

    fn owned(&self) -> Self::Owned;

    fn matches_ref<'g, G: Graph>(&self, graph: &'g G) -> Result<L::Ref<'g>, &'g G>;
    fn matches_value<G: Graph>(&self, graph: G) -> Result<L, G>;

    // Construct a filter for a given child key
    type ChildFilter<'s>: Filter<L>
    where
        Self: 's;

    fn child<'s>(&'s self, key: KeyRef<'s>) -> Self::ChildFilter<'s>;

    // helpers
    fn inv(self) -> Invert<L, Self> {
        Invert(self, PhantomData)
    }
}

impl<L: Leaf, F: Filter<L>> Filter<L> for &F {
    type Owned = F::Owned;
    fn owned(&self) -> Self::Owned {
        (*self).owned()
    }

    fn matches_ref<'g, G: Graph>(&self, graph: &'g G) -> Result<L::Ref<'g>, &'g G> {
        (*self).matches_ref(graph)
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

// Basic filter types

#[derive(Clone)]
pub struct Of<L: Leaf>(PhantomData<L>);

impl<L: Leaf> Of<L> {
    pub fn filter() -> Self {
        Self(PhantomData)
    }
}

impl<L: Leaf> Filter<L> for Of<L> {
    type Owned = Self;
    fn owned(&self) -> Self::Owned {
        self.clone()
    }

    fn matches_ref<'g, G: Graph>(&self, graph: &'g G) -> Result<<L as Leaf>::Ref<'g>, &'g G> {
        L::try_from_ref(graph)
    }
    fn matches_value<G: Graph>(&self, graph: G) -> Result<L, G> {
        L::try_from_value(graph)
    }

    type ChildFilter<'s>
        = Self
    where
        Self: 's;

    fn child<'s>(&'s self, _key: KeyRef<'s>) -> Self::ChildFilter<'s> {
        self.clone()
    }
}

#[derive(Clone)]
pub struct Nothing<L: Leaf>(PhantomData<L>);
impl<L: Leaf> Nothing<L> {
    pub fn filter() -> Self {
        Self(PhantomData)
    }
}

#[rustfmt::skip]
impl<L: Leaf> Filter<L> for Nothing<L> {
    type Owned = Self;
    fn owned(&self) -> Self::Owned {
        self.clone()
    }

    fn matches_ref<'g, G: Graph>(&self, _graph: &'g G)
      -> Result<<L as Leaf>::Ref<'g>, &'g G> { Err(_graph) }
    fn matches_value<G: Graph>(&self, _graph: G)
      -> Result<L, G> { Err(_graph) }

    type ChildFilter<'s>
        = Self
    where
        Self: 's;

    fn child<'s>(&'s self, _key: KeyRef<'s>) -> Self::ChildFilter<'s> {
        self.clone()
    }
}

#[derive(Clone)]
pub struct All;

impl Filter<()> for All {
    type Owned = Self;
    fn owned(&self) -> Self::Owned {
        self.clone()
    }

    fn matches_ref<'g, G: Graph>(&self, _graph: &'g G) -> Result<&'g (), &'g G> {
        Ok(&())
    }
    fn matches_value<G: Graph>(&self, _graph: G) -> Result<(), G> {
        Ok(())
    }

    type ChildFilter<'s>
        = Self
    where
        Self: 's;

    fn child<'s>(&'s self, _key: KeyRef<'s>) -> Self::ChildFilter<'s> {
        self.clone()
    }
}

#[derive(Clone)]
pub struct Invert<L: Leaf, F: Filter<L>>(F, PhantomData<L>);

impl<L: Leaf, F: Filter<L>> Filter<()> for Invert<L, F> {
    type Owned = Invert<L, F::Owned>;
    fn owned(&self) -> Self::Owned {
        Invert(self.0.owned(), PhantomData)
    }

    fn matches_ref<'g, G: Graph>(&self, graph: &'g G) -> Result<&'g (), &'g G> {
        match self.0.matches_ref(graph) {
            Ok(_) => Err(graph),
            Err(_) => Ok(&()),
        }
    }
    fn matches_value<G: Graph>(&self, graph: G) -> Result<(), G> {
        match self.0.matches_ref(&graph) {
            Ok(_) => Err(graph),
            Err(_) => Ok(()),
        }
    }

    type ChildFilter<'s>
        = Invert<L, F::ChildFilter<'s>>
    where
        Self: 's;
    fn child<'s>(&'s self, key: KeyRef<'s>) -> Self::ChildFilter<'s> {
        Invert(self.0.child(key), PhantomData)
    }
}

/// Try `Filter::matches_ref` on each graph expression in order; return the first `Ok`.
///
#[macro_export]
macro_rules! filter_matches_ref_any {
    ($filter:expr, $($case:expr),+ $(,)?) => {
        $crate::filter_matches_ref_any!(@match $filter, $($case),+)
    };
    (@match $filter:expr, $case:expr) => {
        $filter.matches_ref($case)
    };
    (@match $filter:expr, $first:expr, $($rest:expr),+) => {
        match $filter.matches_ref($first) {
            Ok(r) => Ok(r),
            Err(_) => $crate::filter_matches_ref_any!(@match $filter, $($rest),+),
        }
    };
}

/// Try `Filter::matches_value` on each graph expression in order; return the first `Ok`.
#[macro_export]
macro_rules! filter_matches_value_seq {
    ($filter:expr, $binder:ident, $($case:expr),+ $(,)?) => {
        $crate::filter_matches_bound_value_seq!(@match $filter, $binder, $binder, $($case),+)
    };
}

#[macro_export]
macro_rules! filter_matches_bound_value_seq {
    ($filter:expr, $binder:ident, $($case:expr),+ $(,)?) => {
        $crate::filter_matches_bound_value_seq!(@match $filter, $binder, $($case),+)
    };
    (@match $filter:expr, $binder:ident, $case:expr) => {
        $filter.matches_value($case)
    };
    (@match $filter:expr, $binder:ident, $first:expr, $($rest:expr),+) => {
        match $filter.matches_value($first) {
            Ok(r) => Ok(r),
            Err($binder) => {$crate::filter_matches_bound_value_seq!(@match $filter, $binder, $($rest),+)},
        }
    };
}
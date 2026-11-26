use ordered_float::OrderedFloat;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::sync::Arc;

use crate::*;

pub use gnx_derive::{GraphEq, GraphPartialEq};

#[derive(Default)]
pub struct GraphEqContext {
    pub left: GraphContext,
    pub right: GraphContext,
    matched: HashSet<(GraphId, GraphId)>,
    left_to_right: HashMap<GraphId, GraphId>,
    right_to_left: HashMap<GraphId, GraphId>,
}

impl GraphEqContext {
    pub fn new() -> Self {
        Self::default()
    }

    fn match_shared(&mut self, left: GraphId, right: GraphId) -> SharedEqVisit {
        if self.matched.contains(&(left, right)) {
            return SharedEqVisit::Seen;
        }
        if self.left_to_right.get(&left).is_some_and(|id| *id != right)
            || self.right_to_left.get(&right).is_some_and(|id| *id != left)
        {
            return SharedEqVisit::Mismatch;
        }
        self.left_to_right.insert(left, right);
        self.right_to_left.insert(right, left);
        self.matched.insert((left, right));
        SharedEqVisit::First
    }
}

enum SharedEqVisit {
    First,
    Seen,
    Mismatch,
}

pub trait GraphPartialEq<Rhs: Graph = Self>: Graph {
    fn graph_partial_eq<L, F>(&self, other: &Rhs, filter: F, ctx: &mut GraphEqContext) -> bool
    where
        L: Leaf,
        F: Filter<L>;

    fn graph_eq_filtered<L, F>(&self, other: &Rhs, filter: F) -> bool
    where
        L: Leaf,
        F: Filter<L>,
    {
        self.graph_partial_eq(other, filter, &mut GraphEqContext::new())
    }
}

pub trait GraphEq: GraphPartialEq<Self> {
    /// Compare two builders without supplying dynamic leaves. Matching holes
    /// are equal; structure, static values, and sharing must agree.
    fn builders_eq<L: Leaf>(
        left: &Self::Builder<L>,
        right: &Self::Builder<L>,
        ctx: &mut GraphEqContext,
    ) -> bool;

    fn graph_eq<L, F>(&self, other: &Self, filter: F, ctx: &mut GraphEqContext) -> bool
    where
        L: Leaf,
        F: Filter<L>,
    {
        self.graph_partial_eq(other, filter, ctx)
    }

    fn builder_eq<L>(&self, other: Self::Builder<L>) -> bool
    where
        L: Leaf,
        <Self as Graph>::Owned: GraphPartialEq<Self>,
    {
        let mut build_ctx = GraphContext::new();
        let Ok(other) = other.build(<Self as TypedGraph<L>>::as_source(self), &mut build_ctx)
        else {
            return false;
        };
        other.graph_eq_filtered::<(), _>(self, Nothing::<()>::filter())
    }
}

impl<V: Leaf + GraphEq> LeafBuilder<V> {
    /// Compare two leaf templates, including any retained static values.
    pub fn graph_eq(&self, other: &Self, ctx: &mut GraphEqContext) -> bool {
        match (self, other) {
            (Self::Leaf, Self::Leaf) => true,
            (Self::Static(a), Self::Static(b)) => a.graph_eq(b, Nothing::<()>::filter(), ctx),
            _ => false,
        }
    }
}

macro_rules! impl_leaf_builder_eq {
    () => {
        fn builders_eq<L: Leaf>(
            left: &Self::Builder<L>,
            right: &Self::Builder<L>,
            ctx: &mut GraphEqContext,
        ) -> bool {
            left.graph_eq(right, ctx)
        }
    };
}

macro_rules! impl_leaf_eq {
    ($($T:ty)*) => {
        $(
            impl GraphPartialEq for $T {
                fn graph_partial_eq<L, F>(
                    &self,
                    other: &Self,
                    filter: F,
                    _ctx: &mut GraphEqContext,
                ) -> bool
                where
                    L: Leaf,
                    F: Filter<L>,
                {
                    self == other
                        || (filter.matches_ref(self).is_ok() && filter.matches_ref(other).is_ok())
                }
            }

            impl GraphEq for $T { impl_leaf_builder_eq!(); }
        )*
    };
}

impl_leaf_eq!(
    () bool char String
    usize u8 u16 u32 u64 u128
    isize i8 i16 i32 i64 i128
);

impl GraphPartialEq for f32 {
    fn graph_partial_eq<L, F>(&self, other: &Self, filter: F, _ctx: &mut GraphEqContext) -> bool
    where
        L: Leaf,
        F: Filter<L>,
    {
        self.to_bits() == other.to_bits()
            || (filter.matches_ref(self).is_ok() && filter.matches_ref(other).is_ok())
    }
}

impl GraphEq for f32 {
    impl_leaf_builder_eq!();
}

impl GraphPartialEq for f64 {
    fn graph_partial_eq<L, F>(&self, other: &Self, filter: F, _ctx: &mut GraphEqContext) -> bool
    where
        L: Leaf,
        F: Filter<L>,
    {
        OrderedFloat::from(*self) == OrderedFloat::from(*other)
            || (filter.matches_ref(self).is_ok() && filter.matches_ref(other).is_ok())
    }
}

impl GraphEq for f64 {
    impl_leaf_builder_eq!();
}

impl<T, U> GraphPartialEq<Vec<U>> for Vec<T>
where
    T: GraphPartialEq<U>,
    U: Graph,
{
    fn graph_partial_eq<L, F>(&self, other: &Vec<U>, filter: F, ctx: &mut GraphEqContext) -> bool
    where
        L: Leaf,
        F: Filter<L>,
    {
        if filter.matches_ref(self).is_ok() && filter.matches_ref(other).is_ok() {
            return true;
        }
        self.len() == other.len()
            && self
                .iter()
                .zip(other)
                .enumerate()
                .all(|(i, (left, right))| {
                    left.graph_partial_eq(right, filter.child(KeyRef::from(i)), ctx)
                })
    }
}

impl<T: GraphEq> GraphEq for Vec<T> {
    fn builders_eq<L: Leaf>(
        left: &Self::Builder<L>,
        right: &Self::Builder<L>,
        ctx: &mut GraphEqContext,
    ) -> bool {
        match (left, right) {
            (NodeBuilder::Leaf, NodeBuilder::Leaf) => true,
            (NodeBuilder::Node(left), NodeBuilder::Node(right)) => {
                left.len() == right.len()
                    && left
                        .iter()
                        .zip(right)
                        .all(|(left, right)| T::builders_eq::<L>(left, right, ctx))
            }
            _ => false,
        }
    }
}

impl<T: GraphPartialEq<U>, U: Graph> GraphPartialEq<Option<U>> for Option<T> {
    fn graph_partial_eq<L: Leaf, F: Filter<L>>(
        &self,
        other: &Option<U>,
        filter: F,
        ctx: &mut GraphEqContext,
    ) -> bool {
        match (self, other) {
            (None, None) => true,
            (Some(a), Some(b)) => a.graph_partial_eq(b, filter, ctx),
            _ => false,
        }
    }
}
impl<T: GraphEq> GraphEq for Option<T> {
    fn builders_eq<L: Leaf>(
        left: &Self::Builder<L>,
        right: &Self::Builder<L>,
        ctx: &mut GraphEqContext,
    ) -> bool {
        match (left, right) {
            (None, None) => true,
            (Some(a), Some(b)) => T::builders_eq::<L>(a, b, ctx),
            _ => false,
        }
    }
}

macro_rules! impl_shared_eq {
    ($W:ident) => {
        impl<T, U> GraphPartialEq<$W<U>> for $W<T>
        where
            T: GraphPartialEq<U>,
            U: Graph,
        {
            fn graph_partial_eq<L, F>(
                &self,
                other: &$W<U>,
                filter: F,
                ctx: &mut GraphEqContext,
            ) -> bool
            where
                L: Leaf,
                F: Filter<L>,
            {
                let left = GraphId::from($W::as_ptr(self) as u64);
                let right = GraphId::from($W::as_ptr(other) as u64);
                match ctx.match_shared(left, right) {
                    SharedEqVisit::Seen => true,
                    SharedEqVisit::Mismatch => false,
                    SharedEqVisit::First => {
                        self.as_ref().graph_partial_eq(other.as_ref(), filter, ctx)
                    }
                }
            }
        }

        impl<T: GraphEq> GraphEq for $W<T> {
            fn builders_eq<L: Leaf>(
                left: &Self::Builder<L>,
                right: &Self::Builder<L>,
                ctx: &mut GraphEqContext,
            ) -> bool {
                match ctx.match_shared(
                    GraphId::from($W::as_ptr(left) as u64),
                    GraphId::from($W::as_ptr(right) as u64),
                ) {
                    SharedEqVisit::Seen => true,
                    SharedEqVisit::Mismatch => false,
                    SharedEqVisit::First => T::builders_eq::<L>(left, right, ctx),
                }
            }
        }
    };
}

impl_shared_eq!(Rc);
impl_shared_eq!(Arc);

impl<T, U> GraphPartialEq<&U> for &T
where
    T: GraphPartialEq<U>,
    U: Graph,
{
    fn graph_partial_eq<L, F>(&self, other: &&U, filter: F, ctx: &mut GraphEqContext) -> bool
    where
        L: Leaf,
        F: Filter<L>,
    {
        let left = GraphId::from(*self as *const T as u64);
        let right = GraphId::from(*other as *const U as u64);
        match ctx.match_shared(left, right) {
            SharedEqVisit::Seen => true,
            SharedEqVisit::Mismatch => false,
            SharedEqVisit::First => (*self).graph_partial_eq(*other, filter, ctx),
        }
    }
}

impl<T: GraphEq> GraphEq for &T {
    fn builders_eq<L: Leaf>(
        left: &Self::Builder<L>,
        right: &Self::Builder<L>,
        ctx: &mut GraphEqContext,
    ) -> bool {
        <Arc<T> as GraphEq>::builders_eq::<L>(left, right, ctx)
    }
}

macro_rules! impl_tuple_eq {
    ($($T:ident $U:ident),* ; $($idx:tt),*) => {
        impl<$($T, $U,)*> GraphPartialEq<($($U,)*)> for ($($T,)*)
        where
            $($T: GraphPartialEq<$U>, $U: Graph,)*
        {
            #[allow(non_snake_case, unused_variables)]
            fn graph_partial_eq<L, F>(
                &self,
                other: &($($U,)*),
                filter: F,
                ctx: &mut GraphEqContext,
            ) -> bool
            where
                L: Leaf,
                F: Filter<L>,
            {
                if filter.matches_ref(self).is_ok() && filter.matches_ref(other).is_ok() {
                    return true;
                }
                let ($($T,)*) = self;
                let ($($U,)*) = other;
                true $(&& $T.graph_partial_eq($U, filter.child(KeyRef::Index($idx)), ctx))*
            }
        }

        impl<$($T: GraphEq,)*> GraphEq for ($($T,)*) {
            #[allow(non_snake_case)]
            fn builders_eq<L: Leaf>(left: &Self::Builder<L>, right: &Self::Builder<L>, ctx: &mut GraphEqContext) -> bool {
                match (left, right) {
                    (NodeBuilder::Leaf, NodeBuilder::Leaf) => true,
                    (NodeBuilder::Node(left), NodeBuilder::Node(right)) => {
                        true $(&& $T::builders_eq::<L>(&left.$idx, &right.$idx, ctx))*
                    }
                    _ => false,
                }
            }
        }
    };
}

impl_tuple_eq!(A U ; 0);
impl_tuple_eq!(A U, B V ; 0, 1);
impl_tuple_eq!(A U, B V, C W ; 0, 1, 2);
impl_tuple_eq!(A U, B V, C W, D X ; 0, 1, 2, 3);
impl_tuple_eq!(A U, B V, C W, D X, E Y ; 0, 1, 2, 3, 4);
impl_tuple_eq!(A U, B V, C W, D X, E Y, G Z ; 0, 1, 2, 3, 4, 5);
impl_tuple_eq!(A U, B V, C W, D X, E Y, G Z, I J ; 0, 1, 2, 3, 4, 5, 6);
impl_tuple_eq!(A U, B V, C W, D X, E Y, G Z, I J, K M ; 0, 1, 2, 3, 4, 5, 6, 7);
impl_tuple_eq!(A U, B V, C W, D X, E Y, G Z, I J, K M, N O ; 0, 1, 2, 3, 4, 5, 6, 7, 8);
impl_tuple_eq!(A U, B V, C W, D X, E Y, G Z, I J, K M, N O, P Q ; 0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
impl_tuple_eq!(A U, B V, C W, D X, E Y, G Z, I J, K M, N O, P Q, R S ; 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
impl_tuple_eq!(A U, B V, C W, D X, E Y, G Z, I J, K M, N O, P Q, R S, AA AB ; 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);

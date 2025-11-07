use ordered_float::OrderedFloat;
use std::hash::{Hash, Hasher};

use crate::graph::*;

pub trait GraphHash {
    fn graph_hash<H, L, F>(&self, hasher: &mut H, filter: F, ctx: &mut GraphContext)
    where
        H: Hasher,
        L: Leaf,
        F: Filter<L>;
}

pub trait GraphEq {
    fn graph_eq<L, F>(&self, other: &Self, filter: F, ctx: &mut GraphContext) -> bool
    where
        L: Leaf,
        F: Filter<L>;
}

pub trait GraphSerialize {}

// Derive Hash for builtin types

macro_rules! derive_traits {
    ($($T:ty)*) => {
        $(
            impl GraphHash for $T {
                fn graph_hash<H, L, F>(&self, hasher: &mut H, exclude: F, _ctx: &mut GraphContext)
                where
                    H: Hasher,
                    L: Leaf,
                    F: Filter<L>,
                {
                    exclude.matches_ref(self).err()
                        .map(|_| self).hash(hasher)
                }
            }

            impl GraphEq for $T {
                fn graph_eq<L, F>(&self, other: &Self, exclude: F, _ctx: &mut GraphContext) -> bool
                where
                    L: Leaf,
                    F: Filter<L>,
                {
                    self == other || (
                        exclude.matches_ref(self).is_ok()
                        && exclude.matches_ref(other).is_ok()
                    )
                }
            }
        )*
    };
}

derive_traits!(
    () bool char String
    usize u8 u16 u32 u64 u128
    isize i8 i16 i32 i64 i128
);
// Graph hash and eq for floats is based on OrderedFloat behavior

#[rustfmt::skip]
impl GraphHash for f32 {
    fn graph_hash<H, L, F>(&self, hasher: &mut H, exclude: F, _ctx: &mut GraphContext)
            where H: Hasher, L: Leaf, F: Filter<L> {
        exclude.matches_ref(self).err()
            .map(|_| OrderedFloat::from(*self)).hash(hasher)
    }
}
#[rustfmt::skip]
impl GraphHash for f64 {
    fn graph_hash<H, L, F>(&self, hasher: &mut H, exclude: F, _ctx: &mut GraphContext)
            where H: Hasher, L: Leaf, F: Filter<L> {
        exclude.matches_ref(self).err()
            .map(|_| OrderedFloat::from(*self)).hash(hasher)
    }
}
#[rustfmt::skip]
impl GraphEq for f32 {
    fn graph_eq<L, F>(&self, other: &Self, exclude: F, _ctx: &mut GraphContext) -> bool
            where L: Leaf, F: Filter<L> {
        self.to_bits() == other.to_bits() || (
            exclude.matches_ref(self).is_ok()
            && exclude.matches_ref(other).is_ok()
        )
    }
}
#[rustfmt::skip]
impl GraphEq for f64 {
    fn graph_eq<L, F>(&self, other: &Self, exclude: F, _ctx: &mut GraphContext) -> bool
            where L: Leaf, F: Filter<L> {
        OrderedFloat::from(*self) == OrderedFloat::from(*other) || (
            exclude.matches_ref(self).is_ok()
            && exclude.matches_ref(other).is_ok()
        )
    }
}

use ordered_float::OrderedFloat;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::Arc;

pub use gnx_derive::GraphHash;

use crate::*;

#[derive(Default)]
pub struct GraphHashContext {
    seen: HashMap<GraphId, usize>,
}

impl GraphHashContext {
    pub fn new() -> Self {
        Self::default()
    }

    fn shared_index(&mut self, id: GraphId) -> SharedHashVisit {
        if let Some(index) = self.seen.get(&id) {
            SharedHashVisit::Seen(*index)
        } else {
            let index = self.seen.len();
            self.seen.insert(id, index);
            SharedHashVisit::First(index)
        }
    }
}

enum SharedHashVisit {
    First(usize),
    Seen(usize),
}

#[derive(Hash)]
pub enum HashMarker {
    Included,
    Excluded,
    SharedRef,
    SharedNode,
}

struct Included<T: ?Sized>(PhantomData<T>);

struct Excluded<T: ?Sized>(PhantomData<T>);

impl<T: ?Sized> Hash for Included<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        HashMarker::Included.hash(state);
        std::any::type_name::<T>().hash(state);
    }
}

impl<T: ?Sized> Hash for Excluded<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        HashMarker::Excluded.hash(state);
        std::any::type_name::<T>().hash(state);
    }
}

pub trait GraphHash: Graph {
    fn graph_hash<H, L, F>(&self, hasher: &mut H, filter: F, ctx: &mut GraphHashContext)
    where
        H: Hasher,
        L: Leaf,
        F: Filter<L>;

    fn hash_builder<H, L>(builder: &Self::Builder<L>, hasher: &mut H, ctx: &mut GraphHashContext)
    where
        H: Hasher,
        L: Leaf;
}

macro_rules! impl_leaf_hash {
    ($($T:ty)*) => {
        $(
            impl GraphHash for $T {
                fn graph_hash<H, L, F>(&self, hasher: &mut H, filter: F, _ctx: &mut GraphHashContext)
                where
                    H: Hasher,
                    L: Leaf,
                    F: Filter<L>,
                {
                    match filter.matches_ref(self) {
                        Ok(_) => Excluded::<Self>(PhantomData).hash(hasher),
                        Err(_) => {
                            Included::<Self>(PhantomData).hash(hasher);
                            self.hash(hasher);
                        }
                    }
                }

                fn hash_builder<H, L>(
                    builder: &Self::Builder<L>,
                    hasher: &mut H,
                    _ctx: &mut GraphHashContext,
                )
                where
                    H: Hasher,
                    L: Leaf,
                {
                    match builder {
                        LeafBuilder::Leaf => Excluded::<L>(PhantomData).hash(hasher),
                        LeafBuilder::Static(value) => {
                            Included::<Self>(PhantomData).hash(hasher);
                            value.hash(hasher);
                        }
                    }
                }
            }
        )*
    };
}

impl_leaf_hash!(
    () bool char String
    usize u8 u16 u32 u64 u128
    isize i8 i16 i32 i64 i128
);

impl GraphHash for f32 {
    fn graph_hash<H, L, F>(&self, hasher: &mut H, filter: F, _ctx: &mut GraphHashContext)
    where
        H: Hasher,
        L: Leaf,
        F: Filter<L>,
    {
        match filter.matches_ref(self) {
            Ok(_) => Excluded::<Self>(PhantomData).hash(hasher),
            Err(_) => {
                Included::<Self>(PhantomData).hash(hasher);
                OrderedFloat::from(*self).hash(hasher);
            }
        }
    }

    fn hash_builder<H, L>(builder: &Self::Builder<L>, hasher: &mut H, _ctx: &mut GraphHashContext)
    where
        H: Hasher,
        L: Leaf,
    {
        match builder {
            LeafBuilder::Leaf => Excluded::<L>(PhantomData).hash(hasher),
            LeafBuilder::Static(value) => {
                Included::<Self>(PhantomData).hash(hasher);
                OrderedFloat::from(*value).hash(hasher);
            }
        }
    }
}

impl GraphHash for f64 {
    fn graph_hash<H, L, F>(&self, hasher: &mut H, filter: F, _ctx: &mut GraphHashContext)
    where
        H: Hasher,
        L: Leaf,
        F: Filter<L>,
    {
        match filter.matches_ref(self) {
            Ok(_) => Excluded::<Self>(PhantomData).hash(hasher),
            Err(_) => {
                Included::<Self>(PhantomData).hash(hasher);
                OrderedFloat::from(*self).hash(hasher);
            }
        }
    }

    fn hash_builder<H, L>(builder: &Self::Builder<L>, hasher: &mut H, _ctx: &mut GraphHashContext)
    where
        H: Hasher,
        L: Leaf,
    {
        match builder {
            LeafBuilder::Leaf => Excluded::<L>(PhantomData).hash(hasher),
            LeafBuilder::Static(value) => {
                Included::<Self>(PhantomData).hash(hasher);
                OrderedFloat::from(*value).hash(hasher);
            }
        }
    }
}

impl<T: GraphHash> GraphHash for Vec<T> {
    fn graph_hash<H, L, F>(&self, hasher: &mut H, filter: F, ctx: &mut GraphHashContext)
    where
        H: Hasher,
        L: Leaf,
        F: Filter<L>,
    {
        match filter.matches_ref(self) {
            Ok(_) => Excluded::<Self>(PhantomData).hash(hasher),
            Err(_) => {
                Included::<Self>(PhantomData).hash(hasher);
                self.len().hash(hasher);
                for (i, item) in self.iter().enumerate() {
                    i.hash(hasher);
                    item.graph_hash(hasher, filter.child(KeyRef::from(i)), ctx);
                }
            }
        }
    }

    fn hash_builder<H, L>(builder: &Self::Builder<L>, hasher: &mut H, ctx: &mut GraphHashContext)
    where
        H: Hasher,
        L: Leaf,
    {
        fn hash_vec_builder<T, H, L>(
            builder: &NodeBuilder<Vec<T::Builder<L>>>,
            hasher: &mut H,
            ctx: &mut GraphHashContext,
        ) where
            T: GraphHash,
            H: Hasher,
            L: Leaf,
        {
            match builder {
                NodeBuilder::Leaf => Excluded::<L>(PhantomData).hash(hasher),
                NodeBuilder::Node(items) => {
                    Included::<Vec<T>>(PhantomData).hash(hasher);
                    items.len().hash(hasher);
                    for (i, item) in items.iter().enumerate() {
                        i.hash(hasher);
                        T::hash_builder(item, hasher, ctx);
                    }
                }
            }
        }
        hash_vec_builder::<T, H, L>(builder, hasher, ctx);
    }
}

macro_rules! impl_shared_hash {
    ($W:ident) => {
        impl<T: GraphHash> GraphHash for $W<T> {
            fn graph_hash<__Hasher, __Leaf, __Filter>(
                &self,
                hasher: &mut __Hasher,
                filter: __Filter,
                ctx: &mut GraphHashContext,
            ) where
                __Hasher: Hasher,
                __Leaf: Leaf,
                __Filter: Filter<__Leaf>,
            {
                let id = GraphId::from($W::as_ptr(self) as u64);
                match ctx.shared_index(id) {
                    SharedHashVisit::Seen(index) => {
                        HashMarker::SharedRef.hash(hasher);
                        index.hash(hasher);
                    }
                    SharedHashVisit::First(index) => {
                        HashMarker::SharedNode.hash(hasher);
                        index.hash(hasher);
                        self.as_ref().graph_hash(hasher, filter, ctx);
                    }
                }
            }

            fn hash_builder<H, L>(
                builder: &Self::Builder<L>,
                hasher: &mut H,
                ctx: &mut GraphHashContext,
            ) where
                H: Hasher,
                L: Leaf,
            {
                let id = GraphId::from($W::as_ptr(builder) as u64);
                match ctx.shared_index(id) {
                    SharedHashVisit::Seen(index) => {
                        HashMarker::SharedRef.hash(hasher);
                        index.hash(hasher);
                    }
                    SharedHashVisit::First(index) => {
                        HashMarker::SharedNode.hash(hasher);
                        index.hash(hasher);
                        T::hash_builder($W::as_ref(builder), hasher, ctx);
                    }
                }
            }
        }
    };
}

impl_shared_hash!(Rc);
impl_shared_hash!(Arc);

impl<T: GraphHash> GraphHash for &T {
    fn graph_hash<H, L, F>(&self, hasher: &mut H, filter: F, ctx: &mut GraphHashContext)
    where
        H: Hasher,
        L: Leaf,
        F: Filter<L>,
    {
        let id = GraphId::from(*self as *const T as u64);
        match ctx.shared_index(id) {
            SharedHashVisit::Seen(index) => {
                HashMarker::SharedRef.hash(hasher);
                index.hash(hasher);
            }
            SharedHashVisit::First(index) => {
                HashMarker::SharedNode.hash(hasher);
                index.hash(hasher);
                (*self).graph_hash(hasher, filter, ctx);
            }
        }
    }

    fn hash_builder<H, L>(builder: &Self::Builder<L>, hasher: &mut H, ctx: &mut GraphHashContext)
    where
        H: Hasher,
        L: Leaf,
    {
        let id = GraphId::from(Arc::as_ptr(builder) as u64);
        match ctx.shared_index(id) {
            SharedHashVisit::Seen(index) => {
                HashMarker::SharedRef.hash(hasher);
                index.hash(hasher);
            }
            SharedHashVisit::First(index) => {
                HashMarker::SharedNode.hash(hasher);
                index.hash(hasher);
                T::hash_builder(Arc::as_ref(builder), hasher, ctx);
            }
        }
    }
}

macro_rules! impl_tuple_hash {
    ($($T:ident),* ; $($idx:tt),*) => {
        impl<$($T: GraphHash,)*> GraphHash for ($($T,)*) {
            #[allow(non_snake_case, unused_variables)]
            fn graph_hash<__Hasher, __Leaf, __Filter>(
                &self,
                hasher: &mut __Hasher,
                filter: __Filter,
                ctx: &mut GraphHashContext,
            )
            where
                __Hasher: Hasher,
                __Leaf: Leaf,
                __Filter: Filter<__Leaf>,
            {
                match filter.matches_ref(self) {
                    Ok(_) => Excluded::<Self>(PhantomData).hash(hasher),
                    Err(_) => {
                        Included::<Self>(PhantomData).hash(hasher);
                        let ($($T,)*) = self;
                        $(
                            $idx.hash(hasher);
                            $T.graph_hash(hasher, filter.child(KeyRef::Index($idx)), ctx);
                        )*
                    }
                }
            }
            #[allow(non_snake_case, unused_variables)]
            fn hash_builder<__Hasher, __Leaf>(
                builder: &Self::Builder<__Leaf>,
                hasher: &mut __Hasher,
                ctx: &mut GraphHashContext,
            )
            where
                __Hasher: Hasher,
                __Leaf: Leaf,
            {
                match builder {
                    NodeBuilder::Leaf => Excluded::<__Leaf>(PhantomData).hash(hasher),
                    NodeBuilder::Node(($($T,)*)) => {
                        Included::<Self>(PhantomData).hash(hasher);
                        $(
                            $idx.hash(hasher);
                            $T::hash_builder($T, hasher, ctx);
                        )*
                    }
                }
            }
        }
    };
}

impl_tuple_hash!(A ; 0);
impl_tuple_hash!(A, B ; 0, 1);
impl_tuple_hash!(A, B, C ; 0, 1, 2);
impl_tuple_hash!(A, B, C, D ; 0, 1, 2, 3);
impl_tuple_hash!(A, B, C, D, E ; 0, 1, 2, 3, 4);
impl_tuple_hash!(A, B, C, D, E, F ; 0, 1, 2, 3, 4, 5);
impl_tuple_hash!(A, B, C, D, E, F, G ; 0, 1, 2, 3, 4, 5, 6);
impl_tuple_hash!(A, B, C, D, E, F, G, I ; 0, 1, 2, 3, 4, 5, 6, 7);
impl_tuple_hash!(A, B, C, D, E, F, G, I, J ; 0, 1, 2, 3, 4, 5, 6, 7, 8);
impl_tuple_hash!(A, B, C, D, E, F, G, I, J, K ; 0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
impl_tuple_hash!(A, B, C, D, E, F, G, I, J, K, M ; 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
impl_tuple_hash!(A, B, C, D, E, F, G, I, J, K, M, N ; 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);

use crate::graph::{Graph, Leaf};

#[rustfmt::skip]
pub trait Callable<InLeaf: Leaf, OutLeaf: Leaf,
                   Input: Graph<InLeaf>, Output: Graph<OutLeaf>> {
    fn call(&self, input: Input) -> Output;
}

macro_rules! impl_tuple_arg_fn {
    ($($T:ty)*) => {
        paste::paste! {
            impl<InLeaf: Leaf, OutLeaf: Leaf, $($T: Graph<InLeaf>,)* OUT: Graph<OutLeaf>, FUNC> Callable<InLeaf, OutLeaf, ($($T,)*), OUT> for FUNC
            where
                FUNC: Fn($($T,)*) -> OUT,
            {
                fn call(&self, input: ($($T,)*)) -> OUT {
                    let ($([<$T:lower>],)*) = input;
                    self($([<$T:lower>],)*)
                }
            }
        }
    };
}

impl_tuple_arg_fn!();
impl_tuple_arg_fn!(A);
impl_tuple_arg_fn!(A B);
impl_tuple_arg_fn!(A B C);
impl_tuple_arg_fn!(A B C D);
impl_tuple_arg_fn!(A B C D E);
impl_tuple_arg_fn!(A B C D E F);
impl_tuple_arg_fn!(A B C D E F G);
impl_tuple_arg_fn!(A B C D E F G H);
impl_tuple_arg_fn!(A B C D E F G H I);
impl_tuple_arg_fn!(A B C D E F G H I J);
impl_tuple_arg_fn!(A B C D E F G H I J K);
impl_tuple_arg_fn!(A B C D E F G H I J K L);

use crate::graph::Graph;

#[rustfmt::skip]
pub trait Callable<Input: Graph, Output: Graph> {
    fn call(&self, input: Input) -> Output;
}

macro_rules! impl_tuple_arg_fn {
    ($($T:ty)*) => {
        paste::paste! {
            impl<$($T: Graph,)* OUT: Graph, FUNC> Callable<($($T,)*), OUT> for FUNC
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

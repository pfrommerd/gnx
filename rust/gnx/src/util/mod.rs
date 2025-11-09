use std::fmt::Display;

use crate::graph::{Graph, Key};

pub use castaway::LifetimeFree;
pub use castaway::cast as try_specialize;

pub trait Error : Sized + std::error::Error {
    fn custom<T: Display>(msg: T) -> Self;
    
    fn invalid_type(unexp_value: Option<impl Display>, unexp_type: impl Display, expected: impl Display) -> Self {
        Self::custom(format!("Invalid type, got {unexp_type}, expected {expected}"))
    }
    fn missing_child(key: Key) -> Self {
        Self::custom(format!("Missing child {key}"))
    }
}

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

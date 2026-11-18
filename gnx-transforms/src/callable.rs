use gnx_graph::{Graph, OwnedGraph, util::GraphResult};
use std::convert::Infallible;

/// The input and output graph types shared by the callable interfaces.
pub trait Function<I: Graph> {
    type Output: Graph;
}

/// A callable that returns its output directly.
/// Implementing `TryCallable` does not automatically implement `Callable`,
/// even when the error is `Infallible`.
///
/// ```compile_fail
/// use gnx_transforms::{Callable, Function, TryCallable};
/// use std::convert::Infallible;
/// struct OnlyTry;
/// impl Function<()> for OnlyTry { type Output = u32; }
/// impl TryCallable<()> for OnlyTry {
///     type Error = Infallible;
///     fn try_invoke(&self, _: ()) -> Result<u32, Infallible> { Ok(1) }
/// }
/// assert_eq!(Callable::invoke(&OnlyTry, ()), 1);
/// ```
pub trait Callable<I: Graph>: Function<I> {
    fn invoke(&self, input: I) -> Self::Output;
}

/// A callable whose successful result is a graph. Errors need not be graphs.
pub trait TryCallable<I: Graph>: Function<I> {
    type Error;

    fn try_invoke(&self, input: I) -> Result<Self::Output, Self::Error>;
}

/// An infallible callable that can also accept rebuilt, owned inputs.
pub trait Jitable<I: Graph>:
    Callable<I, Output: Graph<Owned = <Self as Function<I>>::Output>>
{
    fn invoke_owned(&self, input: I::Owned) -> Self::Output;
}

/// A fallible callable that can also accept rebuilt, owned inputs.
pub trait TryJitable<I: Graph>:
    TryCallable<I, Output: Graph<Owned = <Self as Function<I>>::Output>>
{
    fn try_invoke_owned(&self, input: I::Owned) -> Result<Self::Output, Self::Error>;
}

pub mod as_fn {
    use super::{Callable, TryCallable};
    use gnx_graph::Graph;
    macro_rules! callable_ext {
        ($num:expr, $($T:ty)*) => {
            paste::paste! {
                pub trait [< CallableExt $num >]<$($T: Graph,)*> : Callable<($($T,)*)> {
                    fn as_fn(&self) -> impl Fn($($T,)*) -> Self::Output {
                        |$([<$T:lower>],)*| self.invoke(($([<$T:lower>],)*))
                    }
                    fn into_fn(self) -> impl Fn($($T,)*) -> Self::Output where Self: Sized {
                        move |$([<$T:lower>],)*| self.invoke(($([<$T:lower>],)*))
                    }
                }
                impl<$($T: Graph,)* FUNC: Callable<($($T,)*)>> [< CallableExt $num >]<$($T,)*> for FUNC {}

                pub trait [< TryCallableExt $num >]<$($T: Graph,)*>: TryCallable<($($T,)*)> {
                    fn as_try_fn(&self) -> impl Fn($($T,)*) -> Result<Self::Output, Self::Error> {
                        |$([<$T:lower>],)*| self.try_invoke(($([<$T:lower>],)*))
                    }
                    fn into_try_fn(self) -> impl Fn($($T,)*) -> Result<Self::Output, Self::Error> where Self: Sized {
                        move |$([<$T:lower>],)*| self.try_invoke(($([<$T:lower>],)*))
                    }
                }
                impl<$($T: Graph,)* FUNC: TryCallable<($($T,)*)>> [< TryCallableExt $num >]<$($T,)*> for FUNC {}
            }
        };
    }
    callable_ext!(0,);
    callable_ext!(1, A);
    callable_ext!(2, A B);
    callable_ext!(3, A B C);
    callable_ext!(4, A B C D);
    callable_ext!(5, A B C D E);
    callable_ext!(6, A B C D E F);
    callable_ext!(7, A B C D E F G);
    callable_ext!(8, A B C D E F G H);
    callable_ext!(9, A B C D E F G H I);
    callable_ext!(10, A B C D E F G H I J);
    callable_ext!(11, A B C D E F G H I J K);
    callable_ext!(12, A B C D E F G H I J K L);
}

macro_rules! impl_tuple_arg_fn {
    ($($T:ty)*) => {
        paste::paste! {
            impl<$($T: Graph,)* FUNC, RET: GraphResult> Function<($($T,)*)> for FUNC
            where
                FUNC: Fn($($T,)*) -> RET,
            {
                type Output = RET::Output;
            }

            impl<$($T: Graph,)* FUNC, RET: GraphResult> TryCallable<($($T,)*)> for FUNC
            where
                FUNC: Fn($($T,)*) -> RET,
            {
                type Error = RET::Error;

                fn try_invoke(&self, input: ($($T,)*)) -> Result<Self::Output, Self::Error> {
                    let ($([<$T:lower>],)*) = input;
                    self($([<$T:lower>],)*).into_result()
                }
            }

            impl<$($T: Graph,)* FUNC, RET: GraphResult<Error = Infallible>>
                Callable<($($T,)*)> for FUNC
            where
                FUNC: Fn($($T,)*) -> RET,
            {
                fn invoke(&self, input: ($($T,)*)) -> Self::Output {
                    self.try_invoke(input).unwrap_or_else(|never| match never {})
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

macro_rules! impl_tuple_arg_jitable_fn {
    ($($T:ty)*) => {
        paste::paste! {
            impl<$($T: Graph<Owned = $T>,)* FUNC, RET: GraphResult>
                TryJitable<($($T,)*)> for FUNC
            where
                FUNC: Fn($($T,)*) -> RET,
                RET::Output: OwnedGraph,
            {
                fn try_invoke_owned(&self, input: ($($T,)*)) -> Result<Self::Output, Self::Error> {
                    self.try_invoke(input)
                }
            }

            impl<$($T: Graph<Owned = $T>,)* FUNC, RET: GraphResult<Error = Infallible>>
                Jitable<($($T,)*)> for FUNC
            where
                FUNC: Fn($($T,)*) -> RET,
                RET::Output: OwnedGraph,
            {
                fn invoke_owned(&self, input: ($($T,)*)) -> Self::Output {
                    self.invoke(input)
                }
            }
        }
    };
}
impl_tuple_arg_jitable_fn!();
impl_tuple_arg_jitable_fn!(A);
impl_tuple_arg_jitable_fn!(A B);
impl_tuple_arg_jitable_fn!(A B C);
impl_tuple_arg_jitable_fn!(A B C D);
impl_tuple_arg_jitable_fn!(A B C D E);
impl_tuple_arg_jitable_fn!(A B C D E F);
impl_tuple_arg_jitable_fn!(A B C D E F G);
impl_tuple_arg_jitable_fn!(A B C D E F G H);
impl_tuple_arg_jitable_fn!(A B C D E F G H I);
impl_tuple_arg_jitable_fn!(A B C D E F G H I J);
impl_tuple_arg_jitable_fn!(A B C D E F G H I J K);
impl_tuple_arg_jitable_fn!(A B C D E F G H I J K L);

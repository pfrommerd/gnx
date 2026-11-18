use gnx_graph::Graph;

#[rustfmt::skip]
pub trait Callable<Input: Graph> {
    type Output: Graph;

    fn invoke(&self, input: Input) -> Self::Output;
}

// A Jitable is any Callable that can also be called
// with an owned version of its input and where the output is owned.
// E.g. a fn(&u32, &u32) -> u32 is Jitable with (Arc<u32>, Arc<u32>) -> u32
//
// This is not always true. For instance (Vec<&u32>) -> u32 is not Jitable
// as we cannot coerce a Vec<Arc<u32>> into a Vec<&u32>.
pub trait Jitable<I: Graph>: Callable<I, Output = Self::OwnedOutput> {
    type OwnedOutput: Graph<Owned = Self::OwnedOutput>;

    fn invoke_owned(&self, input: I::Owned) -> Self::Output;
}

/// A callable whose successful result is a graph. Errors need not be graphs.
pub trait TryCallable<I: Graph> {
    type Output: Graph;
    type Error;

    fn try_invoke(&self, input: I) -> Result<Self::Output, Self::Error>;
}

/// A fallible callable that can also accept the owned form of its input graph.
pub trait TryJitable<I: Graph>: TryCallable<I, Output = Self::OwnedOutput> {
    type OwnedOutput: Graph<Owned = Self::OwnedOutput>;

    fn try_invoke_owned(&self, input: I::Owned) -> Result<Self::Output, Self::Error>;
}

pub mod as_fn {
    use super::Callable;
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
            impl<$($T: Graph,)* FUNC, OUT: Graph> Callable<($($T,)*)> for FUNC
            where
                FUNC: Fn($($T,)*) -> OUT,
            {
                type Output = OUT;
                fn invoke(&self, input: ($($T,)*)) -> Self::Output {
                    let ($([<$T:lower>],)*) = input;
                    self($([<$T:lower>],)*)
                }
            }

            impl<$($T: Graph,)* FUNC, OUT: Graph, ERR> TryCallable<($($T,)*)> for FUNC
            where
                FUNC: Fn($($T,)*) -> Result<OUT, ERR>,
            {
                type Output = OUT;
                type Error = ERR;

                fn try_invoke(&self, input: ($($T,)*)) -> Result<Self::Output, Self::Error> {
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

macro_rules! impl_tuple_arg_jitable_fn {
    ($($T:ty)*) => {
        paste::paste! {
            impl<$($T: Graph<Owned = $T>,)* FUNC, OUT: Graph<Owned = OUT>> Jitable<($($T,)*)> for FUNC
            where
                FUNC: Fn($($T,)*) -> OUT,
            {
                type OwnedOutput = OUT;

                fn invoke_owned(&self, input: ($($T,)*)) -> Self::Output {
                    let ($([<$T:lower>],)*) = input;
                    self($([<$T:lower>],)*)
                }
            }

            impl<$($T: Graph<Owned = $T>,)* FUNC, OUT: Graph<Owned = OUT>, ERR>
                TryJitable<($($T,)*)> for FUNC
            where
                FUNC: Fn($($T,)*) -> Result<OUT, ERR>,
            {
                type OwnedOutput = OUT;

                fn try_invoke_owned(&self, input: ($($T,)*)) -> Result<Self::Output, Self::Error> {
                    self.try_invoke(input)
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

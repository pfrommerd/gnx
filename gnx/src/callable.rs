use gnx::graph::Graph;

#[rustfmt::skip]
pub trait Callable<Input: Graph> {
    type Output: Graph;

    fn invoke(&self, input: Input) -> Self::Output;
}

pub mod as_fn {
    use super::Callable;
    use gnx::graph::Graph;
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
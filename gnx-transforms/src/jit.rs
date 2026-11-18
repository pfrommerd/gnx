use crate::{Callable, Jitable};

pub use gnx_derive::{jit, jit_fn, jittable};

use gnx_graph::Graph;

/// Callable wrapper used by the function attribute and closure macro.
///
/// ```
/// #[gnx_transforms::jit]
/// fn increment(value: u32) -> u32 { value + 1 }
/// let offset = 2;
/// let callable = gnx_transforms::jit_fn!(|value: u32| increment(value) + offset);
/// assert_eq!(callable(3), 6);
/// ```
///
/// The attribute currently accepts no options:
/// ```compile_fail
/// #[gnx_transforms::jit(unsupported)]
/// fn example(value: u32) -> u32 { value }
/// ```
pub struct Jit<F> {
    func: F,
}

impl<I: Graph, F: Jitable<I>> Callable<I> for Jit<F> {
    type Output = <F as Callable<I>>::Output;
    fn invoke(&self, input: I) -> Self::Output {
        self.func.invoke(input)
    }
}

pub fn jit<I: Graph, F: Jitable<I>>(func: F) -> Jit<F> {
    Jit { func }
}

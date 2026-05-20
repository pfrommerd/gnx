use crate::{Callable, Jitable};

pub use gnx_derive::{jit, jittable, transform};

use gnx_graph::Graph;

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

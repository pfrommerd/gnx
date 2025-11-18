use crate::Callable;

pub use gnx_derive::{jit, transform};

use gnx::graph::Graph;


pub struct Jit<F> {
    func: F,
}

impl<I: Graph, F: Callable<I>> Callable<I> for Jit<F> {
    type Output = F::Output;
    fn invoke(&self, input: I) -> Self::Output {
        self.func.invoke(input)
    }
}

pub fn jit<I: Graph, F: Callable<I>>(func: F) -> Jit<F> {
    Jit { func }
}
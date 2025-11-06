use crate::graph::*;

mod dag;
mod traits;

pub use dag::{Dag, ToDag};
pub use traits::{GraphEq, GraphHash};

#[rustfmt::skip]
pub trait GraphExt: Graph {
    fn to_dag<L: Leaf, F: Filter<L>, V, M: FnMut(L::Ref<'_>) -> V>(
        &self, filter: F, map: M
    ) -> Dag<V> {
        self.visit(filter, ToDag(map))
    }
}

impl<G: Graph> GraphExt for G {}

// pub struct GraphDef<I: Leaf, L: Leaf, B: Builder<L>> {
//     builder: B,
//     values: Dag<I>,
//     _phantom: PhantomData<L>,
// }

// impl<L: Leaf, B: Builder<L>, I: Leaf> GraphDef<I, L, B> {}

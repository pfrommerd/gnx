mod dag;
mod traits;
mod callable;

pub use castaway::LifetimeFree;
pub use castaway::cast as try_specialize;

use crate::{Graph, Leaf, Filter};

pub use dag::{Dag, ToDag};
pub use traits::{GraphEq, GraphHash};
pub use callable::Callable;

#[rustfmt::skip]
pub trait GraphExt: Graph {
    fn to_dag<L: Leaf, F: Filter<L>, V, M: FnMut(L::Ref<'_>) -> V>(
        &self, filter: F, map: M
    ) -> Dag<V> {
        self.visit(filter, ToDag(map))
    }
}

impl<G: Graph> GraphExt for G {}
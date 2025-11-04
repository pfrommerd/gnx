use crate::graph::{Filter, Graph, Leaf, LeafCow};

mod dag;
use dag::Dag;

struct ToDag;

#[rustfmt::skip]
pub trait GraphExt: Graph {
    fn to_dag<L: Leaf, F: Filter<L>, V, M: FnMut(L::Ref<'_>) -> V>(
        &self, filter: F, map: M
    ) -> Dag<V> {
        todo!()
    }
    fn into_dag<L: Leaf, F: Filter<L>, V, M: FnMut(LeafCow<'_, L>) -> V>(
        self, filter: F, map: M
    ) -> Dag<V> where Self: Sized {
        todo!()
    }
}

impl<G: Graph> GraphExt for G {}

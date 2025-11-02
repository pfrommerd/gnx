use crate::graph::{Graph, GraphFilter, Node};

use super::dag::Dag;

struct ToDag;

pub trait GraphExt: Graph {
    fn to_dag<L, V: GraphFilter<L>>(&self, viewer: V) -> Dag<L> {
        todo!()
    }
    fn to_dag_recurse<L, V: GraphFilter<L>>(&self, viewer: V) -> Dag<L>
    where
        L: Node,
    {
        todo!()
    }
}

impl<G: Graph> GraphExt for G {}

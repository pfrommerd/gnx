use crate::graph::{Filter, Graph, Node};

use super::dag::Dag;

struct ToDag;

pub trait GraphExt: Graph {
    fn to_dag<L, V: Filter<L>>(&self, viewer: V) -> Dag<L> {
        todo!()
    }
    fn to_dag_recurse<L, V: Filter<L>>(&self, viewer: V) -> Dag<L>
    where
        L: Node,
    {
        todo!()
    }
}

impl<G: Graph> GraphExt for G {}

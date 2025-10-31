use super::*;

use super::path::Dag;

struct ToDag;

pub trait GraphExt: Graph {
    fn to_dag<L, V: GraphViewer<L>>(&self, viewer: V) -> Dag<L> {
        todo!()
    }
    fn to_dag_recurse<L, V: GraphViewer<L>>(&self, viewer: V) -> Dag<L>
    where
        L: Node,
    {
        todo!()
    }
}
impl<G: Graph> GraphExt for G {}

// This file also contains Graph-specific
// PartialEq, Eq, and Hash traits
//
// Unlike their std counterparts, these consider
// also the underlying DAG structure e.g.
//
// if a = b = c = d then:
//
//     (&a, &b) == (&c, &d)
//     (&a, &a) != (&c, &d)

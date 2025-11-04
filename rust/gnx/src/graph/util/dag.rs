use crate::graph::Key;

use std::collections::HashMap;
use std::sync::Arc;

pub enum Dag<V> {
    Leaf(V),
    Node { children: HashMap<Key, Arc<Dag<V>>> },
}

// A generic dag container
pub enum Dag<V> {
    Leaf(V),
    Node {
        value: Option<V>,
        children: HashMap<Key, Arc<Dag<V>>>,
    },
}

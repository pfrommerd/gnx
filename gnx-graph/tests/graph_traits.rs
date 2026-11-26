use std::rc::Rc;

use gnx_graph::anyhash::HasherWrite;
use gnx_graph::{
    Graph, GraphContext, GraphEq, GraphEqContext, GraphHash, GraphHashContext, GraphPartialEq,
    Leaf, Nothing, Of, SourceError,
};

#[derive(Clone, Leaf, GraphHash, GraphPartialEq, GraphEq)]
struct Hashable(u32);

#[derive(Default)]
struct HashBytes(Vec<u8>);

impl HasherWrite for HashBytes {
    fn write(&mut self, bytes: &[u8]) {
        self.0.extend_from_slice(bytes);
    }
}

fn hash_graph<G, L, F>(graph: &G, filter: F) -> Vec<u8>
where
    G: GraphHash,
    L: Leaf,
    F: gnx_graph::Filter<L>,
{
    let mut hasher = HashBytes::default();
    graph.graph_hash::<_, L, _>(&mut hasher, filter, &mut GraphHashContext::new());
    hasher.0
}

fn hash_builder<G, L>(builder: &G::Builder<L>) -> Vec<u8>
where
    G: GraphHash,
    L: Leaf,
{
    let mut hasher = HashBytes::default();
    G::hash_builder(builder, &mut hasher, &mut GraphHashContext::new());
    hasher.0
}

#[test]
fn optional_values_keep_the_inner_path_and_rebuild_the_variant() {
    use gnx_graph::{Builder, GraphSource, NullSource, TypedGraph};
    let some = Some(3u32);
    let none = None::<u32>;
    let value: u32 = some.as_source().leaf(()).unwrap();
    assert_eq!(value, 3);
    let builder = |x: &Option<u32>| {
        x.builder::<u32, _, SourceError>(Of::<u32>::filter(), &mut GraphContext::new())
            .unwrap()
    };
    let (present, absent) = (builder(&some), builder(&none));
    assert_ne!(
        hash_builder::<Option<u32>, u32>(&present),
        hash_builder::<Option<u32>, u32>(&absent)
    );
    assert_eq!(
        hash_graph::<_, u32, _>(&some, Of::<u32>::filter()),
        hash_builder::<Option<u32>, u32>(&present)
    );
    assert!(!some.graph_eq_filtered::<u32, _>(&none, Of::<u32>::filter()));
    assert_eq!(
        present
            .build(
                TypedGraph::<u32>::as_source(&9u32),
                &mut GraphContext::new()
            )
            .unwrap(),
        Some(9)
    );
    assert_eq!(
        Builder::<u32>::build(absent, NullSource, &mut GraphContext::new()).unwrap(),
        None
    );
    let node = Some((1u32, 2u32));
    assert_eq!(
        node.replace(
            Of::<u32>::filter(),
            (4u32, 5u32).as_source(),
            &mut GraphContext::new()
        )
        .unwrap(),
        Some((4, 5))
    );
}

#[test]
fn graph_hash_hashes_builtin_values() {
    let left = vec![1u32, 2, 3];
    let same = vec![1u32, 2, 3];
    let different = vec![1u32, 2, 4];

    assert_eq!(
        hash_graph::<_, (), _>(&left, Nothing::<()>::filter()),
        hash_graph::<_, (), _>(&same, Nothing::<()>::filter())
    );
    assert_ne!(
        hash_graph::<_, (), _>(&left, Nothing::<()>::filter()),
        hash_graph::<_, (), _>(&different, Nothing::<()>::filter())
    );
}

#[test]
fn graph_hash_accounts_for_excluded_structure() {
    let two = vec![Hashable(1), Hashable(2)];
    let three = vec![Hashable(1), Hashable(2), Hashable(3)];

    assert_ne!(
        hash_graph::<_, Hashable, _>(&two, Of::<Hashable>::filter()),
        hash_graph::<_, Hashable, _>(&three, Of::<Hashable>::filter())
    );
}

#[test]
fn graph_hash_distinguishes_shared_structure() {
    let shared = Rc::new(7u32);
    let shared_graph = vec![shared.clone(), shared.clone()];
    let tree_graph = vec![Rc::new(7u32), Rc::new(7u32)];

    assert_ne!(
        hash_graph::<_, (), _>(&shared_graph, Nothing::<()>::filter()),
        hash_graph::<_, (), _>(&tree_graph, Nothing::<()>::filter())
    );
}

#[test]
fn graph_hash_builder_matches_filtered_graph_hash() {
    let graph = vec![1u32, 2, 3];
    let builder = graph
        .builder::<u32, _, SourceError>(Of::<u32>::filter(), &mut GraphContext::new())
        .unwrap();

    assert_eq!(
        hash_graph::<_, u32, _>(&graph, Of::<u32>::filter()),
        hash_builder::<Vec<u32>, u32>(&builder)
    );
}

#[test]
fn graph_hash_builder_matches_graph_hash_under_same_filter() {
    let graph = vec![1u32, 2, 3];
    let builder = graph
        .builder::<(), _, SourceError>(Nothing::<()>::filter(), &mut GraphContext::new())
        .unwrap();

    assert_eq!(
        hash_graph::<_, (), _>(&graph, Nothing::<()>::filter()),
        hash_builder::<Vec<u32>, ()>(&builder)
    );
}

#[test]
fn graph_partial_eq_filters_leaf_values() {
    let left = vec![1u32, 2, 3];
    let right = vec![4u32, 5, 6];
    let shorter = vec![4u32, 5];

    assert!(left.graph_eq_filtered::<u32, _>(&right, Of::<u32>::filter()));
    assert!(!left.graph_eq_filtered::<u32, _>(&shorter, Of::<u32>::filter()));
}

#[test]
fn graph_eq_compares_shared_graph_structure() {
    let left_shared = Rc::new(9u32);
    let right_shared = Rc::new(9u32);

    let left = vec![left_shared.clone(), left_shared.clone()];
    let right = vec![right_shared.clone(), right_shared.clone()];
    let tree = vec![Rc::new(9u32), Rc::new(9u32)];

    assert!(left.graph_eq_filtered::<(), _>(&right, Nothing::<()>::filter()));
    assert!(!left.graph_eq_filtered::<(), _>(&tree, Nothing::<()>::filter()));
}

#[test]
fn graph_eq_builder_compares_prefiltered_builder() {
    let graph = vec![1u32, 2, 3];
    let builder = graph
        .builder::<u32, _, SourceError>(Of::<u32>::filter(), &mut GraphContext::new())
        .unwrap();

    assert!(graph.builder_eq::<u32>(builder));
}

#[test]
fn direct_builder_equality_ignores_holes_but_preserves_static_data_and_sharing() {
    type G = (Vec<Rc<u32>>, bool);
    let builder = |graph: G| {
        graph
            .builder::<u32, _, SourceError>(Of::<u32>::filter(), &mut GraphContext::new())
            .unwrap()
    };
    let compare = |a, b| G::builders_eq::<u32>(&a, &b, &mut GraphEqContext::new());
    let packed = |value: u32, flag: bool| {
        let value = Rc::new(value);
        builder((vec![value.clone(), value], flag))
    };
    assert!(compare(packed(1, true), packed(2, true)));
    assert!(!compare(packed(1, true), packed(1, false)));
    assert!(!compare(
        packed(1, true),
        builder((vec![Rc::new(1), Rc::new(1)], true)),
    ));
}

#[test]
fn derived_builder_equality_compares_static_values_and_holes() {
    use gnx_graph::LeafBuilder::{Leaf, Static};
    let compare = |a, b| Hashable::builders_eq::<Hashable>(&a, &b, &mut GraphEqContext::new());
    assert!(compare(Leaf, Leaf));
    assert!(compare(Static(Hashable(1)), Static(Hashable(1))));
    assert!(!compare(Static(Hashable(1)), Static(Hashable(2))));
    assert!(!compare(Leaf, Static(Hashable(1))));
}

#[test]
fn dag_mapping_uses_insertion_order_and_maps_shared_leaves_once() {
    use gnx_graph::Key;
    use gnx_graph::util::{Dag, DagChild, DagNode};
    use std::sync::Arc;
    let shared = Arc::new(Dag::Leaf(Some(7u32)));
    let graph = Dag::Node(DagNode::Map(
        [
            ("b".into(), DagChild::Owned(Dag::Leaf(Some(2)))),
            ("d".into(), DagChild::Shared(shared.clone())),
            ("a".into(), DagChild::Shared(shared)),
            ("c".into(), DagChild::Owned(Dag::Leaf(Some(3)))),
        ]
        .into(),
    ));
    let mut visited = Vec::new();
    let mapped = graph.map_leaves(|leaf| {
        visited.push(*leaf);
        visited.len() - 1
    });
    assert_eq!(visited, vec![2, 7, 3]);
    let Dag::Node(DagNode::Map(children)) = mapped else {
        panic!("expected map")
    };
    let (DagChild::Shared(a), DagChild::Shared(d)) =
        (&children[&Key::from("a")], &children[&Key::from("d")])
    else {
        panic!("expected shared leaves")
    };
    assert!(Arc::ptr_eq(a, d));
}

#[test]
fn derive_graph_traits_hash_and_compare_fields() {
    let left = Hashable(11);
    let same = Hashable(11);
    let different = Hashable(12);

    assert!(left.graph_eq_filtered::<(), _>(&same, Nothing::<()>::filter()));
    assert!(!left.graph_eq_filtered::<(), _>(&different, Nothing::<()>::filter()));
    assert_eq!(
        hash_graph::<_, (), _>(&left, Nothing::<()>::filter()),
        hash_graph::<_, (), _>(&same, Nothing::<()>::filter())
    );
    assert_ne!(
        hash_graph::<_, (), _>(&left, Nothing::<()>::filter()),
        hash_graph::<_, (), _>(&different, Nothing::<()>::filter())
    );
}

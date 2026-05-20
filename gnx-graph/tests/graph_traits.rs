use std::collections::hash_map::DefaultHasher;
use std::hash::Hasher;
use std::rc::Rc;

use gnx_graph::{
    Graph, GraphContext, GraphEq, GraphHash, GraphHashContext, GraphPartialEq, Leaf, Nothing, Of,
    SourceError,
};

#[derive(Clone, Leaf, GraphHash, GraphPartialEq, GraphEq)]
struct Hashable(u32);

fn hash_graph<G, L, F>(graph: &G, filter: F) -> u64
where
    G: GraphHash,
    L: Leaf,
    F: gnx_graph::Filter<L>,
{
    let mut hasher = DefaultHasher::new();
    graph.graph_hash::<_, L, _>(&mut hasher, filter, &mut GraphHashContext::new());
    hasher.finish()
}

fn hash_builder<G, L>(builder: &G::Builder<L>) -> u64
where
    G: GraphHash,
    L: Leaf,
{
    let mut hasher = DefaultHasher::new();
    G::hash_builder(builder, &mut hasher, &mut GraphHashContext::new());
    hasher.finish()
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

use gnx_graph::util::{Dag, DagChild, DagNode, GraphExt, IterSource};
use gnx_graph::{Builder, Error, Graph, GraphContext, Key, Of, SourceError};
use std::sync::Arc;

#[test]
fn rebuilds_nested_graphs_and_consumes_each_shared_node_once() {
    let shared = Arc::new(1u32);
    let graph = (("static".to_owned(), shared.clone()), (shared, 2u32));
    let filter = Of::<u32>::filter();
    assert_eq!(graph.collect_leaves(&filter), vec![1, 2]);
    let builder = graph
        .builder::<u32, _, SourceError>(filter, &mut GraphContext::new())
        .unwrap();
    let mut leaves = [7u32, 8].into_iter();
    let ((label, a), (b, tail)) = builder
        .clone()
        .build(IterSource::new(&mut leaves), &mut GraphContext::new())
        .unwrap();
    assert_eq!(label, "static");
    assert!(Arc::ptr_eq(&a, &b));
    assert_eq!((*a, tail), (7, 8));
    assert_eq!(leaves.next(), None);
    let error = builder
        .build(IterSource::new([7u32]), &mut GraphContext::new())
        .unwrap_err();
    assert_eq!(error, SourceError::missing_leaf());
}

#[test]
fn dag_map_visit_and_build_follow_insertion_order() {
    let shared = Arc::new(Dag::Leaf(Some(1u32)));
    let graph = Dag::Node(DagNode::Map(
        [
            ("z".into(), DagChild::Owned(Dag::Leaf(Some(3u32)))),
            ("b".into(), DagChild::Shared(shared.clone())),
            ("a".into(), DagChild::Shared(shared)),
            ("c".into(), DagChild::Owned(Dag::Leaf(None))),
            ("y".into(), DagChild::Owned(Dag::Leaf(Some(2u32)))),
        ]
        .into(),
    ));
    let filter = Of::<u32>::filter();
    assert_eq!(graph.collect_leaves(&filter), vec![3, 1, 2]);
    let rebuilt = graph
        .builder::<u32, _, SourceError>(&filter, &mut GraphContext::new())
        .unwrap()
        .build(IterSource::new(vec![4u32, 5, 6]), &mut GraphContext::new())
        .unwrap();
    assert_eq!(rebuilt.collect_leaves(&filter), vec![4, 5, 6]);
    let Dag::Node(DagNode::Map(children)) = rebuilt else {
        panic!("expected a map");
    };
    let (DagChild::Shared(a), DagChild::Shared(b)) =
        (&children[&Key::from("a")], &children[&Key::from("b")])
    else {
        panic!("expected shared children");
    };
    assert!(Arc::ptr_eq(a, b));
    assert!(matches!(
        children[&Key::from("c")],
        DagChild::Owned(Dag::Leaf(None))
    ));
    assert!(matches!(
        children[&Key::from("z")],
        DagChild::Owned(Dag::Leaf(Some(4)))
    ));
}

#[test]
fn converting_to_a_dag_preserves_mixed_key_order() {
    let graph: Dag<u32> = gnx_graph::util::dag! {
        0: 4u32,
        "middle": 5u32,
        1: 6u32,
    };
    let copied = graph.to_dag(Of::<u32>::filter(), |value| *value);
    let Dag::Node(DagNode::Map(children)) = copied else {
        panic!("expected a map");
    };
    assert_eq!(
        children.keys().cloned().collect::<Vec<_>>(),
        vec![Key::from(0), Key::from("middle"), Key::from(1)],
    );
}

use std::sync::Arc;

use gnx_graph::{
    Key,
    util::{Dag, DagChild, DagNode, dag},
};

#[test]
fn dag_macro_builds_leaves_and_empty_leaves() {
    let leaf: Dag<i32> = dag!(7);
    assert!(matches!(leaf, Dag::Leaf(Some(7))));

    let empty: Dag<i32> = dag!(());
    assert!(matches!(empty, Dag::Leaf(None)));
}

#[test]
fn dag_macro_builds_lists_and_maps() {
    let bar = "bar";
    let c = 3;
    let d = 4;
    let dag: Dag<i32> = dag! {
        "foo": [1, 2],
        0: (),
        bar: [c, d],
    };

    let Dag::Node(DagNode::Map(children)) = dag else {
        panic!("expected map node");
    };

    assert!(children.contains_key(&Key::from("foo")));
    assert!(children.contains_key(&Key::from(0)));
    assert!(children.contains_key(&Key::from(bar)));

    let DagChild::Owned(Dag::Node(DagNode::List(foo))) = &children[&Key::from("foo")] else {
        panic!("expected foo list");
    };
    assert_eq!(foo.len(), 2);
}

#[test]
fn dag_macro_let_bindings_create_shared_children() {
    let dag: Dag<i32> = dag! {
        let a = [1, 2];
        "foo": a,
        "bar": a,
    };

    let Dag::Node(DagNode::Map(children)) = dag else {
        panic!("expected map node");
    };

    let DagChild::Shared(foo) = &children[&Key::from("foo")] else {
        panic!("expected shared foo child");
    };
    let DagChild::Shared(bar) = &children[&Key::from("bar")] else {
        panic!("expected shared bar child");
    };
    assert!(Arc::ptr_eq(foo, bar));

    let dag: Dag<i32> = dag! {
        let b = 2;
        b, b
    };
    let Dag::Node(DagNode::List(children)) = dag else {
        panic!("expected list node");
    };
    let DagChild::Shared(left) = &children[0] else {
        panic!("expected shared left child");
    };
    let DagChild::Shared(right) = &children[1] else {
        panic!("expected shared right child");
    };
    assert!(Arc::ptr_eq(left, right));
}

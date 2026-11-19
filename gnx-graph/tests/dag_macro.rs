use std::sync::Arc;

use gnx_graph::{
    Key,
    util::{Dag, DagChild, DagNode, dag},
};

#[test]
fn dag_macro_builds_leaves_and_empty_leaves() {
    let leaf: Dag<i32> = dag!(7);
    assert!(matches!(leaf, Dag::Leaf(Some(7))));

    let empty: Dag<i32> = dag!(*);
    assert!(matches!(empty, Dag::Leaf(None)));
}

#[test]
fn dag_macro_builds_lists_and_maps() {
    let bar = "bar";
    let c = 3;
    let d = 4;
    let dag: Dag<i32> = dag! {
        "foo": [1, 2],
        0: *,
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

#[test]
fn dag_macro_star_is_an_empty_leaf_anywhere_a_value_goes() {
    fn is_empty(child: &DagChild<i32>) -> bool {
        matches!(child, DagChild::Owned(Dag::Leaf(None)))
    }
    let Dag::Node(DagNode::List(children)) = dag!(*, 1, *) else {
        panic!("expected list node");
    };
    assert!(is_empty(&children[0]) && is_empty(&children[2]));
    assert!(matches!(children[1], DagChild::Owned(Dag::Leaf(Some(1)))));

    let Dag::Node(DagNode::List(children)) = dag!([*, (2, *)]) else {
        panic!("expected list node");
    };
    assert!(is_empty(&children[0]));

    let dag: Dag<i32> = dag! { "a": *, "b": 3 };
    let Dag::Node(DagNode::Map(children)) = dag else {
        panic!("expected map node");
    };
    assert!(is_empty(&children[&Key::from("a")]));

    // `()` is the unit value, not an empty leaf.
    assert!(matches!(dag!(()), Dag::<()>::Leaf(Some(()))));
    let Dag::Node(DagNode::List(units)) = dag!((), *) else {
        panic!("expected list node");
    };
    assert!(matches!(
        units[0],
        DagChild::Owned(Dag::<()>::Leaf(Some(())))
    ));
    assert!(matches!(units[1], DagChild::Owned(Dag::Leaf(None))));

    // `*` followed by an operand is still a dereference.
    let value = &5;
    assert!(matches!(dag!(*value), Dag::Leaf(Some(5))));
}

use gnx_graph::util::{Dag, DagChild, DagNode, GraphExt, dag};
use gnx_graph::{Filter, Graph, Key, KeyRef, Leaf, Of};
use std::sync::Arc;

#[derive(Clone)]
struct Name(usize);
impl From<u8> for Name {
    fn from(n: u8) -> Self {
        Self(n.into())
    }
}
impl From<bool> for Name {
    fn from(b: bool) -> Self {
        Self(b.into())
    }
}

#[test]
fn typed_dag_converts_payloads_but_not_keys_or_shared_bindings() {
    let tree = dag!(as Name; let flags=[true, 7u8, *]; "a":flags, "b":flags);
    let Dag::Node(DagNode::Map(nodes)) = tree else {
        panic!()
    };
    let (DagChild::Shared(a), DagChild::Shared(b)) =
        (&nodes[&Key::from("a")], &nodes[&Key::from("b")])
    else {
        panic!()
    };
    assert!(Arc::ptr_eq(a, b));
    let Dag::Node(DagNode::List(items)) = a.as_ref() else {
        panic!()
    };
    assert!(matches!(
        items[..],
        [
            DagChild::Owned(Dag::Leaf(Some(Name(1)))),
            DagChild::Owned(Dag::Leaf(Some(Name(7)))),
            DagChild::Owned(Dag::Leaf(None))
        ]
    ));
    let _: Dag<u8> = dag!(1u8, *);
}

/// Return the borrowed matched string together with its index within the selected subtree.
#[derive(Clone)]
struct Indexed(isize, std::rc::Rc<std::cell::Cell<usize>>);
impl Filter for Indexed {
    type Leaf = String;
    type Annotation<'g> = (isize, &'g str);
    type Owned = Self;
    type ChildFilter<'s> = Self;
    fn owned(&self) -> Self {
        self.clone()
    }
    fn matches_ref<'g, G: Graph>(&self, g: &'g G) -> Result<&'g String, &'g G> {
        Of::<String>::filter().matches_ref(g)
    }
    fn matches_value<G: Graph>(&self, g: G) -> Result<String, G> {
        String::try_from_value(g)
    }
    fn annotate<'g>(&self, s: <String as Leaf>::Ref<'g>) -> Self::Annotation<'g> {
        self.1.set(self.1.get() + 1);
        (self.0, s.as_str())
    }
    fn child<'s>(&'s self, k: KeyRef<'s>) -> Self {
        match k {
            KeyRef::Index(i) => Self(i, self.1.clone()),
            _ => self.clone(),
        }
    }
}
#[test]
fn child_annotations_borrow_original_leaves_and_preserve_sharing() {
    let shared = Arc::new(vec![String::from("x"), String::from("y")]);
    let graph = (vec![String::from("skip")], shared.clone(), shared.clone());
    let cursor = Indexed(-1, Default::default());
    let selected = graph.collect_leaves_with_paths(dag!(*, &cursor, &cursor));
    assert_eq!(
        selected,
        vec![
            (vec![Key::Index(1), Key::Index(0)], "x".to_owned()),
            (vec![Key::Index(1), Key::Index(1)], "y".to_owned())
        ]
    );
    assert_eq!(
        cursor.1.get(),
        0,
        "ordinary leaf collection does not annotate"
    );

    struct Annotate;
    impl<'g> gnx_graph::GraphVisitor<'g, String, String, (isize, &'g str)> for Annotate {
        type Output = (isize, &'g str);
        fn visit_leaf<F: Filter<Leaf = String, Annotation<'g> = Self::Output>>(
            self,
            value: &'g String,
            filter: F,
        ) -> Self::Output {
            filter.annotate(value)
        }
        fn visit_static<S: Leaf>(self, _: S::Ref<'g>) -> Self::Output {
            unreachable!()
        }
        fn visit_node<
            N: gnx_graph::Node,
            F: Filter<Leaf = String, Annotation<'g> = Self::Output>,
        >(
            self,
            _: gnx_graph::View<'g, N, F>,
        ) -> Self::Output {
            unreachable!()
        }
        fn visit_shared<S: Graph, F: Filter<Leaf = String, Annotation<'g> = Self::Output>>(
            self,
            _: gnx_graph::GraphId,
            _: gnx_graph::View<'g, S, F>,
        ) -> Self::Output {
            unreachable!()
        }
    }
    let (index, text) = shared[1].visit(cursor.child(KeyRef::Index(1)), Annotate);
    assert_eq!(index, 1);
    assert_eq!(text.as_ptr(), shared[1].as_ptr());
    assert_eq!(cursor.1.get(), 1);
}

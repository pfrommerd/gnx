use super::{Dag, DagChild, DagNode};
use crate::{Filter, Graph, Key, KeyRef, Leaf};
use std::borrow::Cow;

/// The current routing subtree, and the path below its payload (if already reached).
#[derive(Clone)]
pub struct DagFilterCursor<'a, F: Clone> {
    tree: Option<Cow<'a, Dag<F>>>,
    suffix: Vec<Key>,
}

fn step<'a, F: Clone>(
    tree: Option<&'a Dag<F>>,
    suffix: &[Key],
    key: KeyRef<'_>,
) -> DagFilterCursor<'a, F> {
    let mut suffix = suffix.to_vec();
    let tree = tree.and_then(|tree| {
        let child = match tree {
            Dag::Leaf(None) => return None,
            Dag::Leaf(Some(_)) => {
                suffix.push(key.to_value());
                return Some(tree);
            }
            Dag::Node(_) if matches!(key, KeyRef::Transparent) => return Some(tree),
            Dag::Node(DagNode::Map(m)) => m.get(&key.to_value()),
            Dag::Node(DagNode::List(v)) => match key {
                KeyRef::Index(i) => usize::try_from(i).ok().and_then(|i| v.get(i)),
                _ => None,
            },
        }?;
        Some(match child {
            DagChild::Owned(d) => d,
            DagChild::Shared(d) => d.as_ref(),
        })
    });
    DagFilterCursor {
        tree: tree.map(Cow::Borrowed),
        suffix,
    }
}

fn match_ref<'g, L: Leaf, F: Filter<Leaf = L>, G: Graph>(
    f: &F,
    path: &[Key],
    g: &'g G,
) -> Result<L::Ref<'g>, &'g G> {
    match path.split_first() {
        None => f.matches_ref(g),
        Some((k, p)) => match_ref(&f.child(k.as_ref()), p, g),
    }
}
fn annotation<'g, L: Leaf, F: Filter<Leaf = L>>(
    f: &F,
    path: &[Key],
    leaf: L::Ref<'g>,
) -> F::Annotation<'g> {
    match path.split_first() {
        None => f.annotate(leaf),
        Some((k, p)) => annotation(&f.child(k.as_ref()), p, leaf),
    }
}
fn match_value<L: Leaf, F: Filter<Leaf = L>, G: Graph>(f: &F, path: &[Key], g: G) -> Result<L, G> {
    match path.split_first() {
        None => f.matches_value(g),
        Some((k, p)) => match_value(&f.child(k.as_ref()), p, g),
    }
}

impl<L: Leaf, F: Filter<Leaf = L>> Filter for DagFilterCursor<'_, F> {
    type Leaf = L;
    type Owned = DagFilterCursor<'static, F::Owned>;
    type Annotation<'g> = F::Annotation<'g>;
    type ChildFilter<'s>
        = DagFilterCursor<'s, F>
    where
        Self: 's;
    fn owned(&self) -> Self::Owned {
        DagFilterCursor {
            tree: self
                .tree
                .as_deref()
                .map(|d| Cow::Owned(d.map_leaves(Filter::owned))),
            suffix: self.suffix.clone(),
        }
    }
    fn matches_ref<'g, G: Graph>(&self, graph: &'g G) -> Result<L::Ref<'g>, &'g G> {
        match self.tree.as_deref() {
            Some(Dag::Leaf(Some(f))) => match_ref(f, &self.suffix, graph),
            _ => Err(graph),
        }
    }
    fn matches_value<G: Graph>(&self, graph: G) -> Result<L, G> {
        match self.tree.as_deref() {
            Some(Dag::Leaf(Some(f))) => match_value(f, &self.suffix, graph),
            _ => Err(graph),
        }
    }
    fn annotate<'g>(&self, leaf: L::Ref<'g>) -> Self::Annotation<'g> {
        match self.tree.as_deref() {
            Some(Dag::Leaf(Some(f))) => annotation(f, &self.suffix, leaf),
            _ => unreachable!("only matched leaves have annotations"),
        }
    }
    fn child<'s>(&'s self, key: KeyRef<'s>) -> Self::ChildFilter<'s> {
        step(self.tree.as_deref(), &self.suffix, key)
    }
}

impl<L: Leaf, F: Filter<Leaf = L>> Filter for Dag<F> {
    type Leaf = L;
    type Annotation<'g> = F::Annotation<'g>;
    type Owned = Dag<F::Owned>;
    type ChildFilter<'s>
        = DagFilterCursor<'s, F>
    where
        Self: 's;
    fn owned(&self) -> Self::Owned {
        self.map_leaves(Filter::owned)
    }
    fn matches_ref<'g, G: Graph>(&self, g: &'g G) -> Result<L::Ref<'g>, &'g G> {
        match self {
            Dag::Leaf(Some(f)) => f.matches_ref(g),
            _ => Err(g),
        }
    }
    fn matches_value<G: Graph>(&self, g: G) -> Result<L, G> {
        match self {
            Dag::Leaf(Some(f)) => f.matches_value(g),
            _ => Err(g),
        }
    }
    fn annotate<'g>(&self, l: L::Ref<'g>) -> Self::Annotation<'g> {
        match self {
            Dag::Leaf(Some(f)) => f.annotate(l),
            _ => unreachable!("only matched leaves have annotations"),
        }
    }
    fn child<'s>(&'s self, k: KeyRef<'s>) -> Self::ChildFilter<'s> {
        step(Some(self), &[], k)
    }
}

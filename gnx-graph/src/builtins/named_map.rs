//! String-keyed graph nodes. Hash iteration order must never affect a trace signature.
use super::*;
use crate::anyhash::HasherWrite;
use std::{borrow::Borrow, collections::HashMap, hash::Hash};

trait Name: Borrow<str> + Hash + Eq + Clone + 'static {}
impl<K: Borrow<str> + Hash + Eq + Clone + 'static> Name for K {}
fn sorted<K: Name, V>(m: &HashMap<K, V>) -> Vec<(&K, &V)> {
    let mut entries: Vec<_> = m.iter().collect();
    entries.sort_unstable_by(|a, b| a.0.borrow().cmp(b.0.borrow()));
    entries
}
#[derive(Clone)]
pub struct NamedMapBuilder<K, B>(Vec<(K, B)>);
impl<K: Name, V: Graph> Graph for HashMap<K, V> {
    type Owned = HashMap<K, V::Owned>;
    type Builder<L: Leaf> = NodeBuilder<NamedMapBuilder<K, V::Builder<L>>>;
    fn builder<L: Leaf, F: Filter<L>, E: Error>(
        &self,
        filter: F,
        ctx: &mut GraphContext,
    ) -> Result<Self::Builder<L>, E> {
        if filter.matches_ref(self).is_ok() {
            return Ok(NodeBuilder::Leaf);
        }
        Ok(NodeBuilder::Node(NamedMapBuilder(
            sorted(self)
                .into_iter()
                .map(|(k, v)| {
                    Ok((
                        k.clone(),
                        v.builder(filter.child(KeyRef::from(k.borrow())), ctx)?,
                    ))
                })
                .collect::<Result<_, E>>()?,
        )))
    }
    fn replace<'g, L: Leaf, F: Filter<L>, S: GraphSource<L::Ref<'g>, L>>(
        &'g self,
        filter: F,
        source: S,
        ctx: &mut GraphContext,
    ) -> Result<Self::Owned, S::Error> {
        if let Ok(r) = filter.matches_ref(self) {
            return source
                .leaf(r)?
                .try_into_value()
                .map_err(|_| S::Error::invalid_leaf());
        }
        let mut src = source.node()?;
        sorted(self)
            .into_iter()
            .map(|(k, v)| {
                let key = KeyRef::from(k.borrow());
                Ok((
                    k.clone(),
                    v.replace(filter.child(key), src.expect_child(key)?, ctx)?,
                ))
            })
            .collect()
    }
    fn visit<'g, L: Leaf, F: Filter<L>, Vv: GraphVisitor<'g, Self, L, F::Annotation<'g>>>(
        &'g self,
        filter: F,
        v: Vv,
    ) -> Vv::Output {
        match filter.matches_ref(self) {
            Ok(r) => v.visit_leaf(r, filter),
            Err(x) => v.visit_node(View::new(x, filter)),
        }
    }
    fn visit_into<L: Leaf, F: Filter<L>, C: GraphConsumer<Self, L>>(
        self,
        filter: F,
        c: C,
    ) -> C::Output {
        match filter.matches_value(self) {
            Ok(l) => c.consume_leaf(l),
            Err(x) => c.consume_node(Bound::new(x, filter)),
        }
    }
}
impl<K: Name, V: Graph> Node for HashMap<K, V> {
    fn visit_children<
        'g,
        L: Leaf,
        F: Filter<L>,
        C: ChildrenVisitor<'g, Self, L, F::Annotation<'g>>,
    >(
        &'g self,
        f: F,
        mut c: C,
    ) -> C::Output {
        for (k, v) in sorted(self) {
            let key = KeyRef::from(k.borrow());
            c.visit_child(key, View::new(v, f.child(key)));
        }
        c.finish()
    }
    fn visit_into_children<L: Leaf, F: Filter<L>, C: ChildrenConsumer<Self, L>>(
        self,
        f: F,
        mut c: C,
    ) -> C::Output {
        let mut entries: Vec<_> = self.into_iter().collect();
        entries.sort_unstable_by(|a, b| a.0.borrow().cmp(b.0.borrow()));
        for (k, v) in entries {
            let key = KeyRef::from(k.borrow());
            c.consume_child(key.to_value(), Bound::new(v, f.child(key)));
        }
        c.finish()
    }
}
impl<K: Name, L: Leaf, B: Builder<L>> Builder<L> for NamedMapBuilder<K, B> {
    type Graph = HashMap<K, B::Graph>;
    fn build<S: GraphSource<(), L>>(
        self,
        s: S,
        ctx: &mut GraphContext,
    ) -> Result<Self::Graph, S::Error> {
        let mut src = s.node()?;
        self.0
            .into_iter()
            .map(|(k, b)| {
                let v = b.build(src.expect_child(KeyRef::from(k.borrow()))?, ctx)?;
                Ok((k, v))
            })
            .collect()
    }
}
impl<K: Name, V: GraphHash> GraphHash for HashMap<K, V> {
    fn graph_hash<H: HasherWrite, L: Leaf, F: Filter<L>>(
        &self,
        h: &mut H,
        f: F,
        ctx: &mut GraphHashContext,
    ) {
        if f.matches_ref(self).is_err() {
            for (k, v) in sorted(self) {
                crate::anyhash::Hash::hash(k.borrow(), h);
                v.graph_hash(h, f.child(KeyRef::from(k.borrow())), ctx);
            }
        }
    }
    fn hash_builder<H: HasherWrite, L: Leaf>(
        b: &Self::Builder<L>,
        h: &mut H,
        ctx: &mut GraphHashContext,
    ) {
        if let NodeBuilder::Node(items) = b {
            for (k, v) in &items.0 {
                crate::anyhash::Hash::hash(k.borrow(), h);
                V::hash_builder::<H, L>(v, h, ctx);
            }
        }
    }
}
impl<K: Name, V: GraphEq> GraphPartialEq for HashMap<K, V> {
    fn graph_partial_eq<L: Leaf, F: Filter<L>>(
        &self,
        o: &Self,
        f: F,
        ctx: &mut GraphEqContext,
    ) -> bool {
        if f.matches_ref(self).is_ok() && f.matches_ref(o).is_ok() {
            return true;
        }
        self.len() == o.len()
            && sorted(self).into_iter().all(|(k, v)| {
                o.get(k.borrow())
                    .is_some_and(|w| v.graph_partial_eq(w, f.child(KeyRef::from(k.borrow())), ctx))
            })
    }
}
impl<K: Name, V: GraphEq> GraphEq for HashMap<K, V> {
    fn builders_eq<L: Leaf>(
        a: &Self::Builder<L>,
        b: &Self::Builder<L>,
        ctx: &mut GraphEqContext,
    ) -> bool {
        match (a, b) {
            (NodeBuilder::Node(a), NodeBuilder::Node(b)) => {
                a.0.len() == b.0.len()
                    && a.0
                        .iter()
                        .zip(&b.0)
                        .all(|((k, v), (l, w))| k == l && V::builders_eq::<L>(v, w, ctx))
            }
            (NodeBuilder::Leaf, NodeBuilder::Leaf) => true,
            _ => false,
        }
    }
}

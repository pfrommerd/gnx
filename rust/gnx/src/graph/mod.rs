mod callable;
mod impls;
mod path;

pub use gnx_derive::Union;

pub use callable::Callable;
pub use path::{GraphId, Key};

use std::any::{Any, TypeId};
use std::borrow::Borrow;
use std::collections::{HashMap, HashSet};
use std::error::Error;

pub trait Leaf: Graph<Self> {
    type LeafDef: Clone;

    fn leaf_def(&self) -> Self::LeafDef;
}

pub trait VisitGraph<L: Leaf, G: Graph<L>> {
    type Output;
    fn leaf(self, value: impl Borrow<L>) -> Self::Output;
    // Note the inherent lifetime of N may be restricted b
    // the inherent G
    fn node<'g, N>(self, node: impl Borrow<N>) -> Self::Output
    where
        N: Node<L> + 'g,
        G: 'g;
    fn shared<'g, I>(self, id: GraphId, shared: impl Borrow<I>) -> Self::Output
    where
        I: Graph<L> + 'g,
        G: 'g;
}
pub trait VisitChildren<'g, L: Leaf> {
    type Output;
    fn child<Child>(&mut self, key: Key, child: impl Borrow<Child>) -> &mut Self
    where
        Child: Graph<L> + 'g;
    fn finish(self) -> Self::Output;
}

pub trait MapGraph<L: Leaf, G: Graph<L>> {
    type Output;
    fn leaf(self, value: impl Into<L>) -> Self::Output;
    fn node<'g, N>(self, node: impl Into<N>) -> Self::Output
    where
        N: Node<L> + 'g,
        G: 'g;
    // Once we hit a "Shared," we may have to switch
    // (1) to visiting the inner graph definition,
    // (2) cloning and continue mapping
    // (3) stop e.g. if we have seen this GraphId before
    fn shared<'g, I>(self, id: GraphId, shared: impl Borrow<I>) -> Self::Output
    where
        I: Graph<L> + 'g,
        G: 'g;
}
pub trait MapChildren<'g, L: Leaf> {
    type Output;
    fn child<Child: Graph<L> + 'g>(&mut self, key: Key, child: Child) -> &mut Self;
    fn finish(self) -> Self::Output;
}

pub trait VisitGraphDef<L: Leaf, Def: GraphDef<L>> {
    type Output;
    fn leaf(self, def: impl Borrow<L::LeafDef>) -> Self::Output;
    fn node<'g, N>(self, def: impl Borrow<N>) -> Self::Output
    where
        N: NodeDef<L> + 'g,
        Def: 'g;
    fn shared<'g, I>(self, id: GraphId, inner: impl Borrow<I>) -> Self::Output
    where
        I: GraphDef<L> + 'g,
        Def: 'g;
}
pub trait VisitChildrenDef<'g, L: Leaf> {
    type Output;
    fn child<C>(&mut self, key: Key, child_def: impl Borrow<C>) -> &mut Self
    where
        C: GraphDef<L> + 'g;
    fn finish(self) -> Self::Output;
}

pub trait GraphSource<L: Leaf> {
    type Error: Error + From<&'static str>;

    // Whether this is a shared node
    fn id(&self) -> Option<GraphId>;
    fn leaf(self, leaf_def: &L::LeafDef) -> Result<L, Self::Error>;
    fn node(self) -> impl NodeSource<L, Error = Self::Error>;
}
pub trait NodeSource<L: Leaf> {
    type Error: Error + From<&'static str>;
    fn child<Def: GraphDef<L>>(
        &mut self,
        key: Key,
        child_def: &Def,
        ctx: &mut GraphContext,
    ) -> Result<Option<Def::Graph>, Self::Error>;
}

pub trait Graph<L: Leaf>: Clone {
    // The owned version of this graph node
    // Note that the GraphDef is of the Owned type!
    type Owned: Graph<L, GraphDef = Self::GraphDef>;
    type GraphDef: GraphDef<L, Graph = Self::Owned>;

    fn graph_def(&self) -> Self::GraphDef;

    fn visit<V: VisitGraph<L, Self>>(&self, visitor: V) -> V::Output;
    fn map<M: MapGraph<L, Self>>(self, consumer: M) -> M::Output;
}

pub trait Node<L: Leaf>: Graph<L> {
    fn visit_children<'g, V: VisitChildren<'g, L>>(&self, visitor: V) -> V::Output
    where
        Self: 'g;
    fn map_children<'g, M: MapChildren<'g, L>>(self, consumer: M) -> M::Output
    where
        Self: 'g;
}

pub trait GraphDef<L: Leaf>: Clone {
    type Graph: Graph<L>;

    #[rustfmt::skip]
    fn id(&self) -> Option<GraphId> { None }

    fn visit<'r, V: VisitGraphDef<L, Self>>(&'r self, visitor: V) -> V::Output;
    fn build<S: GraphSource<L>>(
        &self,
        source: S,
        ctx: &mut GraphContext,
    ) -> Result<Self::Graph, S::Error>;
}

pub trait NodeDef<L: Leaf>: GraphDef<L> {
    fn visit_children<'g, V: VisitChildrenDef<'g, L>>(&self, visitor: V) -> V::Output
    where
        Self: 'g;
}

// A GraphContext stores already-constructed graph nodes
// in a generic type-erased way
#[derive(Default)]
pub struct GraphContext {
    // The boxes contain HashMap<GraphId, T>
    maps: HashMap<TypeId, Box<dyn Any>>,
    // All the GraphIds we've seen so far
    seen: HashSet<GraphId>,
}

impl GraphContext {
    pub fn new() -> Self {
        Self::default()
    }
    // Will return Err(()) on type mismatch
    pub fn lookup<T: Clone>(&mut self, id: GraphId) -> Result<Option<T>, &'static str> {
        todo!()
    }
    // Will return Err(()) on type mismatch
    // or if the id was already present (i.e. )
    pub fn put<T: Clone>(&mut self, id: GraphId, value: T) -> Result<(), &'static str> {
        todo!()
    }
}

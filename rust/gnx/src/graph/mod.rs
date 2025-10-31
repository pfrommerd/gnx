mod callable;
mod impls;
mod path;
mod traits;

pub use gnx_derive::LeafUnion;
pub use traits::GraphExt;

pub use callable::Callable;
pub use path::{GraphId, Key, KeyRef, Path};

use std::any::{Any, TypeId};
use std::borrow::Borrow;
use std::collections::{HashMap, HashSet};
use std::error::Error;

pub trait Graph {
    // The owned version of this graph node
    // Note that the GraphDef is of the Owned type!
    type GraphDef<I>: GraphDef<I, Graph = Self::Owned>;
    type Owned: Graph;

    fn graph_def<I, L, V, F>(
        &self,
        viewer: V,
        map: F,
        ctx: &mut GraphContext,
    ) -> Result<Self::GraphDef<I>, GraphError>
    where
        V: GraphViewer<L>,
        F: FnMut(V::Ref<'_>) -> I;

    fn visit<L, V, M>(&self, view: V, visitor: M) -> M::Output
    where
        V: GraphViewer<L>,
        M: GraphVisitor<L, V>;
    fn map<L, V, M>(self, view: V, consumer: M) -> M::Output
    where
        V: GraphViewer<L>,
        M: GraphMap<L, V>;
}

pub trait Node: Graph {
    fn visit_children<L, V, M>(&self, viewer: V, visitor: M) -> M::Output
    where
        V: GraphViewer<L>,
        M: ChildrenVisitor<L, V>;
    fn map_children<L, V, M>(self, viewer: V, map: M) -> M::Output
    where
        V: GraphViewer<L>,
        M: ChildrenMap<L, V>;
}

pub trait GraphDef<I> {
    type Graph: Graph;

    fn visit<V: DefVisitor<I>>(&self, visitor: V) -> V::Output;

    fn build<L, B, S>(
        &self,
        builder: B,
        source: S,
        ctx: &mut GraphContext,
    ) -> Result<Self::Graph, S::Error>
    where
        B: LeafBuilder<I, L>,
        S: GraphSource<I, L>;
}

pub trait NodeDef<I>: GraphDef<I> {
    fn visit_children<V: DefVisitorChildren<I>>(&self, visitor: V) -> V::Output;
}

pub trait GraphVisitor<L, V: GraphViewer<L>> {
    type Output;
    // None if the node is logically a leaf,
    // but it has no value under the current View
    fn leaf(self, value: Option<V::Ref<'_>>) -> Self::Output;
    fn node<N: Node>(self, node: View<'_, N, V>) -> Self::Output;
    fn shared<S: Graph>(self, id: GraphId, shared: View<'_, S, V>) -> Self::Output;
}

pub trait ChildrenVisitor<L, V: GraphViewer<L>> {
    type Output;
    fn child<G: Graph>(&mut self, key: KeyRef<'_>, child: View<'_, G, V>) -> &mut Self;
    fn finish(self) -> Self::Output;
}

pub trait GraphMap<L, V: GraphViewer<L>> {
    type Output;
    fn leaf(self, value: Option<L>) -> Self::Output;
    fn node<N: Node>(self, node: Bound<N, V>) -> Self::Output;
    // Once we hit a "Shared," we may have to switch to
    // (1) to visiting the inner graph definition,
    // (2) cloning and continue mapping
    // (3) stop e.g. if we have seen this GraphId before
    fn shared<S: Graph>(self, id: GraphId, shared: View<'_, S, V>) -> Self::Output;
}

pub trait ChildrenMap<L, V: GraphViewer<L>> {
    type Output;
    fn child<G: Graph>(&mut self, key: Key, child: Bound<G, V>) -> &mut Self;
    fn finish(self) -> Self::Output;
}

pub trait GraphSource<I, L> {
    type Error: Error + From<GraphError>;
    // Whether this is a shared node
    fn id(&self) -> Option<GraphId>;
    // Try to construct a leaf given a value
    fn leaf(self, info: &I) -> Result<L, Self::Error>;
    fn node(self) -> impl ChildrenSource<I, L, Error = Self::Error>;
}

pub trait ChildrenSource<I, L> {
    type Error: Error + From<&'static str>;

    #[rustfmt::skip]
    fn child(&mut self, key: KeyRef<'_>) -> Result<Option<
        impl GraphSource<I, L, Error=Self::Error>
    >, Self::Error>;
}

// Usually GraphViewer is implemented
// for the reference type
// (So that Bound has no lifetime parameters)
pub trait GraphViewer<L>: Copy {
    // A reference-like type for the viewer
    // For instance L might be Option<T> and Ref<'l> is Option<&'l T>
    // Thus a GraphViewer could coerce both T and Option<T> to &
    type Ref<'l>: Borrow<L> + 'l;
    // A viewer knows how to turn a reference to a leaf into a Self::Ref
    fn borrow_leaf<'l>(&self, leaf: &'l L) -> Self::Ref<'l>;

    fn try_as_leaf<'g, G: Graph>(&self, graph: &'g G) -> Result<Self::Ref<'g>, &'g G>;
    fn try_to_leaf<G: Graph>(&self, g: G) -> Result<L, G>;
}
// LeafBuilder is the counterpart to GraphViewer.
// It allows constructing arbitrary Graph types given an associated Def and a value
pub trait LeafBuilder<I, L>: Copy {
    fn try_build<D: GraphDef<I>>(&self, def: &D, value: L) -> Result<D::Graph, GraphError>;
}

pub struct View<'g, G: Graph, V> {
    pub graph: &'g G,
    pub viewer: V,
}

impl<'g, G: Graph, V> View<'g, G, V> {
    fn new(graph: &'g G, viewer: V) -> View<'g, G, V> {
        View { graph, viewer }
    }
}
pub struct Bound<G: Graph, V> {
    pub graph: G,
    pub viewer: V,
}
impl<G: Graph, V> Bound<G, V> {
    fn new(graph: G, viewer: V) -> Bound<G, V> {
        Bound { graph, viewer }
    }
}

pub enum GraphError {
    MissingNode,
    UnsupportedLeafDef,
}

pub trait DefVisitor<I> {
    type Output;
    fn leaf(self, value: Option<&I>) -> Self::Output;
    fn node<N: NodeDef<I>>(self, def: &N) -> Self::Output;
    fn shared<S: GraphDef<I>>(self, id: GraphId, shared: &S) -> Self::Output;
}

pub trait DefVisitorChildren<I> {
    type Output;
    fn child<C>(&mut self, key: KeyRef, child_def: &C) -> &mut Self
    where
        C: GraphDef<I>;
    fn finish(self) -> Self::Output;
}

// A GraphContext stores already-constructed graph nodes
// in a generic type-erased way
// The proper way of interacting with it is through the ctx_build! macro,
// which releases the &mut borrow on the context while building child nodes
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

    // For use by the ctx_build_shared macro
    // through which you should interact with the GraphContext
    // The macro released the &mut borrow while building child nodes
    pub fn _reserve<T: Clone>(&mut self, id: GraphId) -> Result<Option<T>, GraphError> {
        self.seen.insert(id);
    }
    pub fn _finish<T: Clone>(&mut self, id: GraphId, value: T) -> Result<(), GraphError> {
        todo!()
    }
}

// Use like ctx_build_shared!(ctx, id, { ... body ... }) -> Result<T, E>
pub use gnx_derive::ctx_build_shared;

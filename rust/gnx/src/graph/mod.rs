mod callable;
mod impls;
mod path;
mod traits;

pub use gnx_derive::{Leaf, ctx_build_shared, impl_leaf};
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
    type GraphDef<I: Clone + 'static, R: NonLeafRepr>: GraphDef<I, Graph = Self::Owned>;
    // The "owned" graph needs to be static and cloneable
    type Owned: Graph + Clone + 'static;

    fn graph_def<I, L, V, F>(
        &self,
        viewer: V,
        map: F,
        ctx: &mut GraphContext,
    ) -> Result<Self::GraphDef<I, V::NonLeafRepr>, GraphError>
    where
        I: Clone + 'static,
        V: GraphViewer<L>,
        F: FnMut(V::Ref<'_>) -> I;

    fn visit<L, V, M>(&self, view: V, visitor: M) -> M::Output
    where
        V: GraphViewer<L>,
        M: GraphVisitor<L, V>;

    // visit_mut will visit all mutablely accessible children, mutably.
    // When a node contains immutable shared children (e.g. Rc<T>)
    // these can be traversed using visit_mut_inner, which takes &self
    // and acts like visit(), but will attempt to recursively
    // visit children like RefCell<T> mutably if possible.
    // This allows us to track mutation state

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

pub trait GraphDef<I: Clone + 'static>: Clone + 'static {
    type Graph: Graph + 'static;

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

    type NonLeafRepr: NonLeafRepr;
}

// A non-leaf viewer knows how to handle
// leaves that cannot be coerced to type L
// This is generally a marker type, hence Copy + 'static
pub trait NonLeafRepr: 'static {
    // For graphdefs, a viewer needs to know what to
    // do with leaves of type T that cannot be coerced
    // to the leaf type L. These will be stored as NonLeaf<T>

    // Ideally we would be able to relax the Clone + 'static
    // on the T parameter so that we can handle
    // non-cloneable "static" fields
    type NonLeaf<T: Clone + 'static>: Into<T> + Clone + 'static;
    fn try_to_nonleaf<T: Clone + 'static>(v: &T) -> Result<Self::NonLeaf<T>, GraphError>;
    // Note that the NonLeaf type must be 'static, so
    // we need to be able to try and convert them back
    // without access to self (this conversion may fail!)
    fn try_from_nonleaf<T: Clone + 'static>(v: &Self::NonLeaf<T>) -> Result<T, GraphError>;
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

// LeafBuilder is the counterpart to GraphViewer.
// It allows constructing arbitrary Graph types given an associated Def and a value
pub trait LeafBuilder<I: Clone + 'static, L>: Copy {
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
    ContextError,
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
    pub fn _reserve<T: Clone + 'static>(&mut self, id: GraphId) -> Result<Option<T>, GraphError> {
        if self.seen.contains(&id) {
            return Err(GraphError::ContextError);
        }
        self.seen.insert(id);
        let map = self
            .maps
            .entry(TypeId::of::<T>())
            .or_insert_with(|| Box::new(HashMap::<GraphId, T>::new()));
        let map = map
            .downcast_mut::<HashMap<GraphId, T>>()
            .ok_or(GraphError::ContextError)?;
        if let Some(s) = map.get(&id) {
            Ok(Some(s.clone()))
        } else {
            Ok(None)
        }
    }
    pub fn _finish<T: Clone + 'static>(&mut self, id: GraphId, value: T) -> Result<(), GraphError> {
        let map = self
            .maps
            .entry(TypeId::of::<T>())
            .or_insert_with(|| Box::new(HashMap::<GraphId, T>::new()));
        // self.seen.insert(id);
        let map = map
            .downcast_mut::<HashMap<GraphId, T>>()
            .ok_or(GraphError::ContextError)?;
        map.insert(id, value);
        Ok(())
    }
}

// A static non-leaf viewer

struct NonLeafCloner;

impl NonLeafRepr for NonLeafCloner {
    type NonLeaf<T: Clone + 'static> = T;
    fn try_to_nonleaf<T: Clone + 'static>(v: &T) -> Result<Self::NonLeaf<T>, GraphError> {
        Ok(v.clone())
    }
    fn try_from_nonleaf<T: Clone + 'static>(v: &Self::NonLeaf<T>) -> Result<T, GraphError> {
        Ok(v.clone())
    }
}

// Helper types for defining GraphDefs
// A LeafDef is either a leaf value (identified by I)
// or a static value of type T

pub enum LeafDef<I, T, R>
where
    I: Clone + 'static,
    T: Graph + Clone + 'static,
    R: NonLeafRepr,
{
    Leaf(I),
    NonLeaf(R::NonLeaf<T>),
}

impl<I, T, R> Clone for LeafDef<I, T, R>
where
    I: Clone + 'static,
    T: Graph + Clone + 'static,
    R: NonLeafRepr,
{
    fn clone(&self) -> Self {
        match self {
            LeafDef::Leaf(i) => LeafDef::Leaf(i.clone()),
            LeafDef::NonLeaf(nl) => LeafDef::NonLeaf(nl.clone()),
        }
    }
}

impl<I, T, R> GraphDef<I> for LeafDef<I, T, R>
where
    I: Clone + 'static,
    T: Graph + Clone + 'static,
    R: NonLeafRepr,
{
    type Graph = T;

    fn build<L, B, S>(
        &self,
        builder: B,
        source: S,
        _ctx: &mut GraphContext,
    ) -> Result<Self::Graph, S::Error>
    where
        B: LeafBuilder<I, L>,
        S: GraphSource<I, L>,
    {
        match self {
            LeafDef::Leaf(leaf) => Ok(builder.try_build(self, source.leaf(leaf)?)?),
            LeafDef::NonLeaf(value) => Ok(R::try_from_nonleaf(value)?),
        }
    }
}

#[derive(Clone)]
pub enum NodeDef<I, N>
where
    I: Clone + 'static,
    N: GraphDef<I>,
{
    Leaf(I),
    Node(N),
}

impl<I: Clone + 'static, N: GraphDef<I>> GraphDef<I> for NodeDef<I, N> {
    type Graph = N::Graph;

    fn build<L, B, S>(
        &self,
        builder: B,
        source: S,
        ctx: &mut GraphContext,
    ) -> Result<Self::Graph, S::Error>
    where
        B: LeafBuilder<I, L>,
        S: GraphSource<I, L>,
    {
        match self {
            NodeDef::Leaf(leaf) => Ok(builder.try_build(self, source.leaf(leaf)?)?),
            NodeDef::Node(node_def) => node_def.build(builder, source, ctx),
        }
    }
}

// LeafDef and BoundDef themselves are graphs!
// This way we can use a graphdef to select axes to vmap over.
// let graph = ...; // some graph
// let target_axes = graph.graph_def(Of<Array>, |a| 0)
//
// zip(&mut graph, target_shapes).visit_mut(
//    LeafMap::where(Of<(Array, Shape)>, |(x, target_shape)| x.reshape(target_shape)
// ))

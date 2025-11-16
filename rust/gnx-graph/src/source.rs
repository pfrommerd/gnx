use super::*;

pub trait GraphSource<I, L> {
    type Error: Error;
    type ChildrenSource: ChildrenSource<I, L, Error = Self::Error>;
    // Try to construct a leaf given a value
    fn leaf(self, info: I) -> Result<L, Self::Error>;
    fn empty_leaf(self) -> Result<(), Self::Error>;
    fn node(self) -> Result<Self::ChildrenSource, Self::Error>;

    // Convert this source into a permissive graph source,
    // i.e. one that only errors on missing leaf() calls
    type Permissive: GraphSource<I, L, Error = Self::Error>;
    fn permissive(self) -> Self::Permissive;
}

pub trait ChildrenSource<I, L> {
    type Error: Error;
    type ChildSource: GraphSource<I, L, Error = Self::Error>;
    #[rustfmt::skip]
    fn child(&mut self, key: KeyRef<'_>) -> Result<Option<Self::ChildSource>, Self::Error>;

    fn expect_child(&mut self, key: KeyRef<'_>) -> Result<Self::ChildSource, Self::Error> {
        match self.child(key)? {
            Some(s) => Ok(s),
            None => Err(Self::Error::missing_child(key.to_value())),
        }
    }
}

pub struct NullSource;

impl<I, L> GraphSource<I, L> for NullSource {
    type Error = SourceError;
    type ChildrenSource = NullSource;

    fn leaf(self, _info: I) -> Result<L, Self::Error> {
        Err(SourceError::missing_leaf())
    }
    fn empty_leaf(self) -> Result<(), Self::Error> {
        Ok(())
    }
    fn node(self) -> Result<NullSource, Self::Error> {
        Ok(NullSource)
    }
    type Permissive = Self;
    fn permissive(self) -> Self {
        self
    }
}
impl<I, L> ChildrenSource<I, L> for NullSource {
    type Error = SourceError;
    type ChildSource = NullSource;

    fn child(&mut self, _key: KeyRef<'_>) -> Result<Option<NullSource>, Self::Error> {
        Ok(Some(NullSource))
    }
}

pub enum PermissiveSource<S> {
    Source(S),
    Null,
}

impl<S> From<S> for PermissiveSource<S> {
    fn from(value: S) -> Self {
        PermissiveSource::Source(value)
    }
}

impl<I, L, S: GraphSource<I, L>> GraphSource<I, L> for PermissiveSource<S> {
    type Error = S::Error;
    type ChildrenSource = PermissiveSource<S::ChildrenSource>;

    fn leaf(self, info: I) -> Result<L, Self::Error> {
        use PermissiveSource::*;
        match self {
            Null => Err(Self::Error::missing_leaf()),
            Source(s) => s.leaf(info),
        }
    }
    fn empty_leaf(self) -> Result<(), Self::Error> {
        Ok(())
    }
    fn node(self) -> Result<Self::ChildrenSource, Self::Error> {
        use PermissiveSource::*;
        match self {
            Source(s) => Ok(Source(s.node()?)),
            Null => Ok(Null),
        }
    }
    type Permissive = Self;
    fn permissive(self) -> Self::Permissive { self }
}
impl<I, L, S: ChildrenSource<I, L>> ChildrenSource<I, L> for PermissiveSource<S> {
    type Error = S::Error;
    type ChildSource = PermissiveSource<S::ChildSource>;

    fn child(&mut self, key: KeyRef<'_>) -> Result<Option<Self::ChildSource>, Self::Error> {
        use PermissiveSource::*;
        match self {
            Null => Ok(Some(Null)),
            Source(s) => match s.child(key)? {
                Some(s) => Ok(Some(Source(s))),
                None => Ok(Some(Null))
            },
        }
    }
}

// Wrap any graph with a filter in an AsSource to turn it into a graph source!
pub struct AsSource<'s, G: ?Sized, F>(&'s G, F);
impl<'g, G: ?Sized, F> AsSource<'g, G, F> {
    pub fn new(graph: &'g G, filter: F) -> Self {
        Self(graph, filter)
    }
}

impl<'g, A, L: Leaf, G: Graph, F: Filter<L>> GraphSource<A, L> for AsSource<'g, G, F> {
    type Error = SourceError;
    type ChildrenSource = GenericChildren<'g, L>;
    fn leaf(self, _: A) -> Result<L, Self::Error> {
        self.0.visit(&self.1, ToLeaf)
    }
    fn empty_leaf(self) -> Result<(), Self::Error> {
        if self.0.visit(&self.1, IsStaticLeaf) {
            Ok(())
        } else {
            Err(SourceError::missing_static_leaf())
        }
    }
    fn node(self) -> Result<Self::ChildrenSource, Self::Error> {
        self.0.visit(&self.1, ToChildren)
    }

    type Permissive = PermissiveSource<Self>;
    fn permissive(self) -> Self::Permissive {
        self.into()
    }
}

// The dyn-compatible machinery to allow for
// nodes with heterogenous children types
// to be represented as a "ChildrenSource"

use std::collections::HashMap;
pub struct GenericChildren<'g, L>(HashMap<Key, GenericSource<'g, L>>);
type GenericSource<'g, L> = Box<dyn _AnySource<'g, L> + 'g>;

impl<'g, A, L: 'g> ChildrenSource<A, L> for GenericChildren<'g, L> {
    type Error = SourceError;
    type ChildSource = Box<dyn _AnySource<'g, L> + 'g>;

    #[rustfmt::skip]
    fn child(&mut self, key: KeyRef<'_>) -> Result<Option<Self::ChildSource>, Self::Error> {
        Ok(self.0.remove(&key.to_value()))
    }
}

// Visitors to convert a graph to either a Leaf L or a GenericChildren
struct ToLeaf;
struct IsStaticLeaf;
struct ToChildren;

#[rustfmt::skip]
impl<'g, G: Graph, L: Leaf> GraphVisitor<'g, G, L> for ToLeaf {
    type Output = Result<L, SourceError>;

    fn visit_leaf(self, value: L::Ref<'_>)
        -> Self::Output { Ok(L::clone_ref(value)) }
    fn visit_static<S: Leaf>(self, _: S::Ref<'_>)
        -> Self::Output { Err(SourceError::expected_leaf()) }
    fn visit_shared<S: Graph, F: Filter<L>>(
        self, _id: GraphId, shared: View<'g, S, F>,
    ) -> Self::Output { shared.visit(self) }
    fn visit_node<N: Node, F: Filter<L>>(self, _: View<'g, N, F>) -> Self::Output {
        Err(SourceError::expected_leaf())
    }
}

#[rustfmt::skip]
impl<'g, G: Graph, L: Leaf> GraphVisitor<'g, G, L> for IsStaticLeaf {
    type Output = bool;

    fn visit_leaf(self, _: L::Ref<'_>)
        -> Self::Output { false }
    fn visit_static<S: Leaf>(self, _: S::Ref<'_>)
        -> Self::Output { true }
    fn visit_shared<S: Graph, F: Filter<L>>(
        self, _id: GraphId, shared: View<'g, S, F>,
    ) -> Self::Output { shared.visit(self) }
    fn visit_node<N: Node, F: Filter<L>>(self, _: View<'g, N, F>)
    -> Self::Output { false }
}

#[rustfmt::skip]
impl<'g, G: Graph, L: Leaf> GraphVisitor<'g, G, L> for ToChildren {
    type Output = Result<GenericChildren<'g, L>, SourceError>;

    fn visit_leaf(self, _value: L::Ref<'_>)
        -> Self::Output { Err(SourceError::expected_node()) }
    fn visit_static<S: Leaf>(self, _value: S::Ref<'_>)
        -> Self::Output { Err(SourceError::expected_node()) }
    fn visit_shared<S: Graph, F: Filter<L>>(
        self, _id: GraphId, shared: View<'g, S, F>,
    ) -> Self::Output { shared.visit(self) }
    fn visit_node<N: Node, F: Filter<L>>(self, node: View<'g, N, F>) -> Self::Output {
        Ok(node.visit_children(CollectChildren::default()))
    }
}
struct CollectChildren<'g, L>(HashMap<Key, Box<dyn _AnySource<'g, L> + 'g>>);
impl<'g, L> Default for CollectChildren<'g, L> {
    fn default() -> Self {
        Self(HashMap::new())
    }
}

#[rustfmt::skip]
impl<'g, N: Node, L: Leaf> ChildrenVisitor<'g, N, L> for CollectChildren<'g, L> {
    type Output = GenericChildren<'g, L>;
    fn visit_child<C: Graph, F: Filter<L>>(
        &mut self,
        key: KeyRef<'_>,
        child: View<'g, C, F>,
    ) -> &mut Self {
        let filter = child.filter.owned();
        let child = Box::new(AsSource(child.graph, filter));
        self.0.insert(key.to_value(), child);
        self
    }
    fn finish(self) -> Self::Output {
        GenericChildren(self.0)
    }
}

// Internal type-erasure machinery used to convert AsSource<'g, G, F>
// into a a type-erased Box<dyn _AnySource<'g, L>>
pub trait _AnySource<'g, L> {
    fn leaf_any(&self) -> Result<L, SourceError>;
    fn empty_leaf_any(&self) -> Result<(), SourceError>;
    fn node_any(&self) -> Result<GenericChildren<'g, L>, SourceError>;
}
impl<'g, L: Leaf, G: Graph, F: Filter<L>> _AnySource<'g, L> for AsSource<'g, G, F> {
    fn leaf_any(&self) -> Result<L, SourceError> {
        self.0.visit(&self.1, ToLeaf)
    }
    fn empty_leaf_any(&self) -> Result<(), SourceError> {
        if self.0.visit(&self.1, IsStaticLeaf) {
            Ok(())
        } else {
            Err(SourceError::missing_static_leaf())
        }
    }
    fn node_any(&self) -> Result<GenericChildren<'g, L>, SourceError> {
        self.0.visit(&self.1, ToChildren)
    }
}
impl<'g, A, L: 'g> GraphSource<A, L> for Box<dyn _AnySource<'g, L> + 'g> {
    type Error = SourceError;
    type ChildrenSource = GenericChildren<'g, L>;

    fn leaf(self, _: A) -> Result<L, Self::Error> {
        self.leaf_any()
    }
    fn empty_leaf(self) -> Result<(), Self::Error> {
        self.empty_leaf_any()
    }
    fn node(self) -> Result<Self::ChildrenSource, Self::Error> {
        self.node_any()
    }

    type Permissive = PermissiveSource<Self>;
    fn permissive(self) -> Self::Permissive {
        self.into()
    }
}

// Erased error type
#[derive(Debug, Eq, PartialEq, Hash, Clone)]
pub enum SourceError {
    ExpectedLeaf,
    ExpectedStaticLeaf,
    ExpectedNode,
    Custom(String),
    Other,
}

impl std::fmt::Display for SourceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SourceError::ExpectedNode => write!(f, "Not a node!"),
            SourceError::ExpectedLeaf => write!(f, "Not a leaf!"),
            SourceError::ExpectedStaticLeaf => write!(f, "Not a static leaf!"),
            SourceError::Custom(v) => write!(f, "{}", v),
            SourceError::Other => write!(f, "Other"),
        }
    }
}
impl std::error::Error for SourceError {}

impl Error for SourceError {
    fn custom<T: std::fmt::Display>(msg: T) -> Self {
        SourceError::Custom(msg.to_string())
    }
}
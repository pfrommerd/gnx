// Blanket Graph implementations for common rust types

use super::*;
use super::util::try_specialize;

mod containers;
mod refs;

    
macro_rules! impl_leaf {
    ($name:ty) => {
        impl Graph for $name {
            type Owned = Self;
            type Builder<L: Leaf> = LeafBuilder<Self>;

            fn replace<'g, L: Leaf, F: Filter<L>, S: GraphSource<L::Ref<'g>, L>>(
                &'g self, filter: F, source: S, _ctx: &mut GraphContext,
            ) -> Result<Self::Owned, S::Error> {
                match filter.matches_ref(self) {
                    Ok(r) => Ok(source.leaf(r)?
                        .try_into_value()
                        .or_else(<Self as Leaf>::try_from_value)
                        .map_err(|_| S::Error::invalid_leaf())?),
                    Err(_) => Ok(self.clone())
                }
            }
            fn builder<'g, L: Leaf, F: Filter<L>, E: Error>(
                &'g self, filter: F, _ctx: &mut GraphContext
            ) -> Result<Self::Builder<L>, E> {
                match filter.matches_ref(self) {
                    Ok(_) => Ok(LeafBuilder::Leaf),
                    Err(_) => Ok(LeafBuilder::Static(self.clone())),
                }
            }

            fn visit<'g, L: Leaf, F: Filter<L>, V: GraphVisitor<'g, Self, L>>(
                &'g self, filter: F, visitor: V
            ) -> V::Output {
                match filter.matches_ref(self) {
                    Ok(r) => visitor.visit_leaf(r),
                    Err(s) => visitor.visit_static::<Self>(<Self as Leaf>::as_ref(s))
                }
            }
            fn visit_into<L: Leaf, F: Filter<L>, C: GraphConsumer<Self, L>>(
                self, filter: F, consumer: C
            ) -> C::Output {
                match filter.matches_value(self) {
                    Ok(v) => consumer.consume_leaf(v),
                    Err(s) => consumer.consume_static::<Self>(s)
                }
            }
        }
        impl TypedGraph<$name> for $name {}
        impl Leaf for $name {
            type Ref<'l> = &'l Self
                where Self: 'l;
            fn as_ref<'l>(&'l self) -> Self::Ref<'l> { self }
            fn clone_ref(v: Self::Ref<'_>) -> Self { v.clone() }
            fn try_from_value<V>(g: V) -> Result<Self, V> {
                try_specialize!(g, Self)
            }
            fn try_from_ref<'v, V>(graph: &'v V) -> Result<Self::Ref<'v>, &'v V> {
                try_specialize!(graph, Self::Ref<'v>)
            }
            fn try_into_value<V: 'static>(self) -> Result<V, Self> {
                try_specialize!(self, V)
            }
        }
    };
}

impl_leaf!(());
impl_leaf!(char);
impl_leaf!(usize);
impl_leaf!(u8);
impl_leaf!(u16);
impl_leaf!(u32);
impl_leaf!(u64);
impl_leaf!(u128);

impl_leaf!(isize);
impl_leaf!(i8);
impl_leaf!(i16);
impl_leaf!(i32);
impl_leaf!(i64);
impl_leaf!(i128);

impl_leaf!(bool);
impl_leaf!(f32);
impl_leaf!(f64);

impl_leaf!(String);

// Owned builders helper types for leaves and nodes

#[derive(Clone)]
pub enum LeafBuilder<V: Leaf> {
    Leaf,
    Static(V),
}

#[derive(Clone)]
pub enum NodeBuilder<B> {
    Leaf,    // Build from a leaf value
    Node(B), // The owned builder for the node
}

impl<L: Leaf, V: Leaf> Builder<L> for LeafBuilder<V> {
    type Graph = V;
    fn build<S: GraphSource<(), L>>(
        self,
        source: S,
        _ctx: &mut GraphContext,
    ) -> Result<V, S::Error> {
        match self {
            // First try the converter associated
            // with the produced leaf type L, then
            // try the converter associated with V
            LeafBuilder::Leaf => match L::try_into_value::<V>(source.leaf(())?) {
                Ok(v) => Ok(v),
                Err(l) => Ok(V::try_from_value(l).map_err(|_| S::Error::invalid_leaf())?),
            },
            LeafBuilder::Static(v) => {
                source.empty_leaf()?;
                Ok(v)
            }
        }
    }
}
impl<L, N, B> Builder<L> for NodeBuilder<B>
where
    L: Leaf,
    N: Graph + Clone + 'static,
    B: Builder<L, Graph = N> + Clone + 'static,
{
    type Graph = N;
    fn build<S: GraphSource<(), L>>(
        self,
        source: S,
        ctx: &mut GraphContext,
    ) -> Result<N, S::Error> {
        match self {
            NodeBuilder::Leaf => Ok(
                L::try_into_value::<N>(source.leaf(())?).map_err(|_| S::Error::invalid_leaf())?
            ),
            NodeBuilder::Node(b) => b.build(source, ctx),
        }
    }
}
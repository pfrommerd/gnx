// Blanket Graph implementations for common rust types

use super::*;

mod containers;
mod rc;

use gnx_derive::impl_leaf;

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
                Err(l) => Ok(V::try_from_value(l).map_err(|_| GraphError::InvalidLeaf)?),
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
                L::try_into_value::<N>(source.leaf(())?).map_err(|_| GraphError::InvalidLeaf)?
            ),
            NodeBuilder::Node(b) => b.build(source, ctx),
        }
    }
}

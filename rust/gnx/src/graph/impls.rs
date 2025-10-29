// Blanket Graph implementations for common rust types
use super::*;

use std::sync::Arc;

// Builtin leaf types
impl<L: Leaf> Graph<L> for u8 {}

impl Graph<u8> for u8 {}

// A blanket impl for &T
impl<L: Leaf, T: Graph<L>> Graph<L> for &T {
    // We use Arc so that if &T is Sync,
    // GraphDef and Owned are also Sync
    type GraphDef = T::GraphDef;
    type Owned = T::Owned;
    fn graph_def(&self) -> Self::GraphDef {
        (*self).graph_def()
    }
    fn visit<V: VisitGraph<L, Self>>(&self, visitor: V) -> V::Output {
        let id: GraphId = ((*self as *const T) as u64).into();
        visitor.shared::<T>(id, *self)
    }
    fn map<M: MapGraph<L, Self>>(self, map: M) -> M::Output {
        let id: GraphId = ((self as *const T) as u64).into();
        map.shared::<T>(id, self)
    }
}

// Vec<T: Graph<L>> implements Graph<L>
impl<L: Leaf, T: Graph<L>> Graph<L> for Vec<T> {
    type GraphDef = Vec<T::GraphDef>;
    type Owned = Vec<T::Owned>;
    fn graph_def(&self) -> Self::GraphDef {
        self.iter().map(|t| t.graph_def()).collect()
    }
    fn visit<V: VisitGraph<L, Self>>(&self, visitor: V) -> V::Output {
        visitor.node::<Self>(self)
    }
    fn map<M: MapGraph<L, Self>>(self, map: M) -> M::Output {
        map.node::<Self>(self)
    }
}
impl<L: Leaf, T: Graph<L>> Node<L> for Vec<T> {
    fn visit_children<'g, V: VisitChildren<'g, L>>(&self, mut visitor: V) -> V::Output
    where
        Self: 'g,
    {
        for (idx, child) in self.iter().enumerate() {
            visitor.child::<T>(Key::Index(idx), child);
        }
        visitor.finish()
    }
    fn map_children<'g, M: MapChildren<'g, L>>(self, mut map: M) -> M::Output
    where
        Self: 'g,
    {
        for (idx, child) in self.into_iter().enumerate() {
            map.child(Key::Index(idx), child);
        }
        map.finish()
    }
}

// Vec<D: GraphDef<L>> implements GraphDef<L>
impl<L: Leaf, D: GraphDef<L>> GraphDef<L> for Vec<D> {
    type Graph = Vec<D::Graph>;
    fn visit<V: VisitGraphDef<L, Self>>(&self, visitor: V) -> V::Output {
        visitor.node::<Self>(self)
    }

    fn build<S: GraphSource<L>>(
        &self,
        source: S,
        mut ctx: &mut GraphContext,
    ) -> Result<Self::Graph, S::Error> {
        let mut ns = source.node();
        let r = self
            .iter()
            .enumerate()
            .map(|(i, d)| ns.child(Key::Index(i), d, &mut ctx))
            .collect::<Result<Option<Vec<D::Graph>>, S::Error>>();
        r.transpose()
            .unwrap_or(Err(S::Error::from("Not enough elements to build Vec!")))
    }
}
#[rustfmt::skip]
impl<L: Leaf, D: GraphDef<L>> NodeDef<L> for Vec<D> {
    fn visit_children<'g, V: VisitChildrenDef<'g, L>>(&self, visitor: V) -> V::Output
        where Self: 'g { visitor.finish() }
}

// For arrays we can now use const generics to implement for all sizes!

// Macro to implement for tuples up to arity 10
macro_rules! impl_tuple_graph {
    ($($T:ty)*, $($idx:expr)*) => {
        paste::paste! {
            impl<L: Leaf, $($T: Graph<L>,)*> Graph<L> for ($($T,)*) {
                type GraphDef = ($($T::GraphDef,)*);
                type Owned = ($($T::Owned,)*);

                fn graph_def(&self) -> Self::GraphDef {
                    let ($([<$T:lower>],)*) = self;
                    ($([<$T:lower>].graph_def(),)*)
                }
                fn visit<V: VisitGraph<L, Self>>(&self, visitor: V) -> V::Output {
                    visitor.node::<Self>(self)
                }
                fn map<M: MapGraph<L, Self>>(self, consumer: M) -> M::Output {
                    consumer.node::<Self>(self)
                }
            }
            impl<L: Leaf, $($T: Graph<L>,)*> Node<L> for ($($T,)*) {
                #[allow(unused_mut)]
                fn visit_children<'g, V: VisitChildren<'g, L>>(&self, mut visitor: V) -> V::Output
                        where Self: 'g {
                    let ($([<$T:lower>],)*) = self;
                    $(
                        visitor.child::<$T>(Key::Index($idx), [<$T:lower>]);
                    )*
                    visitor.finish()
                }
                #[allow(unused_mut)]
                fn map_children<'g, M: MapChildren<'g, L>>(self, mut map: M) -> M::Output
                        where Self: 'g {
                    let ($([<$T:lower>],)*) = self;
                    $(
                        map.child(Key::Index($idx), [<$T:lower>]);
                    )*
                    map.finish()
                }
            }
            impl<L: Leaf, $($T: GraphDef<L>,)*> GraphDef<L> for ($($T,)*) {
                type Graph = ($($T::Graph,)*);

                fn visit<V: VisitGraphDef<L, Self>>(&self, visitor: V) -> V::Output {
                    visitor.node::<Self>(self)
                }
                #[allow(unused_mut, unused_variables)]
                fn build<S: GraphSource<L>>(&self, source: S, mut ctx: &mut GraphContext) -> Result<Self::Graph, S::Error> {
                    let mut ns = source.node();
                    $(
                        let [<$T:lower>] = ns.child(Key::Index($idx), &self.$idx, &mut ctx)?
                            .ok_or(S::Error::from("Not enough elements to build tuple!"))?;
                    )*
                    Ok(($( [<$T:lower>], )*))
                }
            }
            impl<L: Leaf, $($T: GraphDef<L>,)*> NodeDef<L> for ($($T,)*) {
                #[allow(unused_mut)]
                fn visit_children<'g, V: VisitChildrenDef<'g, L>>(&self, mut visitor: V) -> V::Output
                        where Self: 'g {
                    let ($([<$T:lower>],)*) = self;
                    $(
                        visitor.child::<$T>(Key::Index($idx), [<$T:lower>]);
                    )*
                    visitor.finish()
                }
            }
        }
    };
}

impl_tuple_graph!(,);
impl_tuple_graph!(A, 0);
impl_tuple_graph!(A B, 0 1);
impl_tuple_graph!(A B C, 0 1 2);
impl_tuple_graph!(A B C D, 0 1 2 3);
impl_tuple_graph!(A B C D E, 0 1 2 3 4);
impl_tuple_graph!(A B C D E F, 0 1 2 3 4 5);
impl_tuple_graph!(A B C D E F G, 0 1 2 3 4 5 6);
impl_tuple_graph!(A B C D E F G H, 0 1 2 3 4 5 6 7);
impl_tuple_graph!(A B C D E F G H I, 0 1 2 3 4 5 6 7 8);
impl_tuple_graph!(A B C D E F G H I J, 0 1 2 3 4 5 6 7 8 9);
impl_tuple_graph!(A B C D E F G H I J K, 0 1 2 3 4 5 6 7 8 9 10);
impl_tuple_graph!(A B C D E F G H I J K N, 0 1 2 3 4 5 6 7 8 9 10 11);

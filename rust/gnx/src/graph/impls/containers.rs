use super::*;

impl<T: Graph> Graph for Vec<T> {
    type Builder<'g, L: Leaf, F: Filter<L>>
        = ViewBuilder<'g, Self, F>
    where
        Self: 'g;
    type OwnedBuilder<L: Leaf> = NodeBuilder<Vec<T::OwnedBuilder<L>>>;
    type Owned = Vec<T::Owned>;

    fn builder<'g, L: Leaf, F: Filter<L>>(&'g self, filter: F) -> Self::Builder<'g, L, F> {
        ViewBuilder::new(self, filter)
    }

    fn visit<'g, L: Leaf, F: Filter<L>, V: GraphVisitor<'g, Self, L>>(
        &'g self,
        filter: F,
        visitor: V,
    ) -> V::Output {
        match filter.matches_ref(self) {
            Ok(leaf) => visitor.visit_leaf(leaf),
            Err(graph) => visitor.visit_node(View::new(graph, filter)),
        }
    }
    fn mut_visit<'g, L: Leaf, F: Filter<L>, V: GraphMutVisitor<'g, Self, L>>(
        &'g mut self,
        filter: F,
        visitor: V,
    ) -> V::Output {
        match filter.matches_mut(self) {
            Ok(leaf) => visitor.visit_leaf_mut(leaf),
            Err(graph) => visitor.visit_node_mut(ViewMut::new(graph, &filter)),
        }
    }
    fn into_visit<L: Leaf, F: Filter<L>, C: GraphConsumer<Self, L>>(
        self,
        filter: F,
        consumer: C,
    ) -> C::Output {
        match filter.matches_value(self) {
            Ok(leaf) => consumer.consume_leaf(leaf),
            Err(graph) => consumer.consume_node(Bound::new(graph, filter)),
        }
    }
}
impl<L: Leaf, T: TypedGraph<L>> TypedGraph<L> for Vec<T> {}

impl<T: Graph> Node for Vec<T> {
    fn visit_children<'g, L: Leaf, F: Filter<L>, V: ChildrenVisitor<'g, Self, L>>(
        &'g self,
        filter: F,
        mut visitor: V,
    ) -> V::Output {
        self.iter().enumerate().for_each(|(i, x)| {
            let r = KeyRef::Index(i);
            visitor.visit_child(r, View::new(x, filter.child(r)));
        });
        visitor.finish()
    }
    fn mut_visit_children<'g, L: Leaf, F: Filter<L>, V: ChildrenMutVisitor<'g, Self, L>>(
        &'g mut self,
        filter: F,
        mut visitor: V,
    ) -> V::Output {
        self.iter_mut().enumerate().for_each(|(i, x)| {
            visitor.visit_child_mut(KeyRef::Index(i), ViewMut::new(x, &filter));
        });
        visitor.finish()
    }
    fn into_visit_children<L: Leaf, F: Filter<L>, C: ChildrenConsumer<Self, L>>(
        self,
        filter: F,
        mut consumer: C,
    ) -> C::Output {
        self.into_iter().enumerate().for_each(|(i, x)| {
            consumer.consume_child(Key::Index(i), Bound::new(x, &filter));
        });
        consumer.finish()
    }
}

// Vec<D: GraphDef<L>> implements GraphDef<L>
impl<'g, L: Leaf, F: Filter<L>, T: Graph> Builder<L> for ViewBuilder<'g, Vec<T>, F> {
    type Graph = Vec<T::Owned>;
    type Owned = NodeBuilder<Vec<T::OwnedBuilder<L>>>;

    fn build<S: GraphSource<(), L>>(
        self,
        source: S,
        ctx: &mut GraphContext,
    ) -> Result<Self::Graph, S::Error> {
        match self.filter.matches_ref(self.graph) {
            Ok(_) => Ok(source
                .leaf(())?
                .try_into_value()
                .map_err(|_| GraphError::InvalidLeaf)?),
            Err(node) => {
                let mut ns = source.node()?;
                node.iter()
                    .enumerate()
                    .map(|(i, g)| {
                        let key = KeyRef::Index(i);
                        let builder = g.builder(self.filter.child(key));
                        builder.build(ns.child(key)?, ctx)
                    })
                    .collect()
            }
        }
    }
    fn to_owned_builder(&self, mut ctx: &mut GraphContext) -> Self::Owned {
        NodeBuilder::Node(
            self.graph
                .iter()
                .enumerate()
                .map(|(i, g)| {
                    let key = KeyRef::Index(i);
                    let builder = g.builder(self.filter.child(key));
                    builder.to_owned_builder(&mut ctx)
                })
                .collect(),
        )
    }
}
impl<'g, L: Leaf, B: Builder<L>> Builder<L> for Vec<B> {
    type Graph = Vec<B::Graph>;
    type Owned = Vec<B::Owned>;

    fn build<S: GraphSource<(), L>>(
        self,
        source: S,
        ctx: &mut GraphContext,
    ) -> Result<Self::Graph, S::Error> {
        let mut ns = source.node()?;
        self.into_iter()
            .enumerate()
            .map(|(i, b)| {
                let key = KeyRef::Index(i);
                b.build(ns.child(key)?, ctx)
            })
            .collect()
    }
    fn to_owned_builder(&self, ctx: &mut GraphContext) -> Self::Owned {
        self.iter().map(|b| b.to_owned_builder(ctx)).collect()
    }
}

macro_rules! impl_tuple_graph {
    ($($T:ty)*, $($idx:expr)*) => {
        paste::paste! {
            impl<$($T: Graph,)*> Graph for ($($T,)*) {
                type Owned = ($($T::Owned,)*);
                type Builder<'g, L: Leaf, F: Filter<L>>
                    = ViewBuilder<'g, Self, F>
                where
                    Self: 'g;
                type OwnedBuilder<L: Leaf> = NodeBuilder<($($T::OwnedBuilder<L>,)*)>;

                fn builder<'g, L: Leaf, F: Filter<L>>(&'g self, filter: F) -> Self::Builder<'g, L, F> {
                    ViewBuilder::new(self, filter)
                }

                fn visit<'g, L: Leaf, F: Filter<L>, V: GraphVisitor<'g, Self, L>>(
                    &'g self, filter: F, visitor: V
                ) -> V::Output {
                    match filter.matches_ref(self) {
                        Ok(leaf) => visitor.visit_leaf(leaf),
                        Err(graph) => visitor.visit_node(View::new(graph, filter))
                    }
                }
                fn mut_visit<'g, L: Leaf, F: Filter<L>, V: GraphMutVisitor<'g, Self, L>>(
                    &'g mut self, filter: F, visitor: V
                ) -> V::Output {
                    match filter.matches_mut(self) {
                        Ok(leaf) => visitor.visit_leaf_mut(leaf),
                        Err(graph) => visitor.visit_node_mut(ViewMut::new(graph, filter))
                    }
                }
                fn into_visit<L: Leaf, F: Filter<L>, C: GraphConsumer<Self, L>>(
                    self, filter: F, consumer: C
                ) -> C::Output {
                    match filter.matches_value(self) {
                        Ok(leaf) => consumer.consume_leaf(leaf),
                        Err(graph) => consumer.consume_node(Bound::new(graph, filter))
                    }
                }
            }
            impl<L: Leaf, $($T: TypedGraph<L>,)*> TypedGraph<L> for ($($T,)*) {}
            impl<$($T: Graph,)*> Node for ($($T,)*) {
                #[allow(unused_variables, unused_mut)]
                fn visit_children<'g, L: Leaf, F: Filter<L>, V: ChildrenVisitor<'g, Self, L>>(
                    &'g self, filter: F, mut visitor: V
                ) -> V::Output {
                    let ($([<$T:lower>],)*) = self;
                    $(
                      visitor.visit_child(KeyRef::Index($idx), View::new([<$T:lower>], &filter));
                    )*
                    visitor.finish()
                }
                #[allow(unused_variables, unused_mut)]
                fn mut_visit_children<'g, L: Leaf, F: Filter<L>, V: ChildrenMutVisitor<'g, Self, L>>(
                    &'g mut self, filter: F, mut visitor: V
                ) -> V::Output {
                    let ($([<$T:lower>],)*) = self;
                    $(
                      visitor.visit_child_mut(KeyRef::Index($idx), ViewMut::new([<$T:lower>], &filter));
                    )*
                    visitor.finish()
                }
                #[allow(unused_variables, unused_mut)]
                fn into_visit_children<L: Leaf, F: Filter<L>, C: ChildrenConsumer<Self, L>>(
                    self, filter: F, mut consumer: C
                ) -> C::Output {
                    let ($([<$T:lower>],)*) = self;
                    $(
                      consumer.consume_child(Key::Index($idx), Bound::new([<$T:lower>], &filter));
                    )*
                    consumer.finish()
                }
            }
            impl<'g, L: Leaf, F: Filter<L>, $($T: Graph,)*> Builder<L> for ViewBuilder<'g, ($($T,)*), F> {
                type Graph = ($($T::Owned,)*);
                type Owned = NodeBuilder<($($T::OwnedBuilder<L>,)*)>;

                #[allow(unused_variables, unused_mut)]
                fn build<S>(
                    self,
                    mut source: S,
                    mut ctx: &mut GraphContext,
                ) -> Result<Self::Graph, S::Error>
                where
                    S: GraphSource<(), L>,
                {
                    let ($([<$T:lower>],)*) = self.graph;
                    let mut ns = source.node()?;
                    $(
                        let [<$T:lower>] = [<$T:lower>].builder(&self.filter).build(
                            ns.child(KeyRef::Index($idx))?, &mut ctx
                        )?;
                    )*
                    Ok(($([<$T:lower>],)*))
                }

                #[allow(unused_variables, unused_mut)]
                fn to_owned_builder(&self, mut ctx: &mut GraphContext) -> Self::Owned {
                    let ($([<$T:lower>],)*) = self.graph;
                    NodeBuilder::Node((
                        $(
                            [<$T:lower>].builder(&self.filter).to_owned_builder(&mut ctx),
                        )*
                    ))
                }
            }
            impl<L: Leaf, $($T: Builder<L>,)*> Builder<L> for ($($T,)*) {
                type Graph = ($($T::Graph,)*);
                type Owned = ($($T::Owned,)*);

                #[allow(unused_variables, unused_mut)]
                fn build<S>(
                    self,
                    mut source: S,
                    mut ctx: &mut GraphContext,
                ) -> Result<Self::Graph, S::Error>
                where
                    S: GraphSource<(), L>,
                {
                    let ($([<$T:lower>],)*) = self;
                    let mut ns = source.node()?;
                    $(
                        let [<$T:lower>] = [<$T:lower>].build(
                            ns.child(KeyRef::Index($idx))?, &mut ctx
                        )?;
                    )*
                    Ok(($([<$T:lower>],)*))
                }

                #[allow(unused_variables, unused_mut)]
                fn to_owned_builder(&self, mut ctx: &mut GraphContext) -> Self::Owned {
                    let ($([<$T:lower>],)*) = self;
                    (
                        $(
                            [<$T:lower>].to_owned_builder(&mut ctx),
                        )*
                    )
                }
            }
        }
    };
}

impl_tuple_graph!(A, 0);
impl_tuple_graph!(A BB, 0 1);
impl_tuple_graph!(A BB CC, 0 1 2);
impl_tuple_graph!(A BB CC D, 0 1 2 3);
impl_tuple_graph!(A BB CC D E, 0 1 2 3 4);
impl_tuple_graph!(A BB CC D E FF, 0 1 2 3 4 5);
impl_tuple_graph!(A BB CC D E FF G, 0 1 2 3 4 5 6);
impl_tuple_graph!(A BB CC D E FF G H, 0 1 2 3 4 5 6 7);
impl_tuple_graph!(A BB CC D E FF G H II, 0 1 2 3 4 5 6 7 8);
impl_tuple_graph!(A BB CC D E FF G H II J, 0 1 2 3 4 5 6 7 8 9);
impl_tuple_graph!(A BB CC D E FF G H II J K, 0 1 2 3 4 5 6 7 8 9 10);
impl_tuple_graph!(A BB CC D E FF G H II J K LL, 0 1 2 3 4 5 6 7 8 9 10 11);

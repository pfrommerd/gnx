use super::*;

impl<T: Graph> Graph for Vec<T> {
    type Owned = Vec<T::Owned>;
    type Builder<L: Leaf> = NodeBuilder<Vec<T::Builder<L>>>;

    fn builder<'g, L: Leaf, F: Filter<L>>(
        &'g self,
        filter: F,
        mut ctx: &mut GraphContext,
    ) -> Result<Self::Builder<L>, GraphError> {
        match filter.matches_ref(self) {
            Ok(_) => Ok(NodeBuilder::Leaf),
            Err(graph) => Ok(NodeBuilder::Node(
                graph
                    .iter()
                    .enumerate()
                    .map(|(i, g)| {
                        let key = KeyRef::Index(i);
                        g.builder(filter.child(key), &mut ctx)
                    })
                    .collect::<Result<Vec<T::Builder<L>>, GraphError>>()?,
            )),
        }
    }
    fn replace<'g, L: Leaf, F: Filter<L>, S: GraphSource<L::Ref<'g>, L>>(
        &'g self,
        filter: F,
        source: S,
        ctx: &mut GraphContext,
    ) -> Result<Self::Owned, S::Error> {
        match filter.matches_ref(self) {
            Ok(r) => Ok(source
                .leaf(r)?
                .try_into_value()
                .map_err(|_| GraphError::InvalidLeaf)?),
            Err(graph) => {
                let mut ns = source.node()?;
                graph
                    .iter()
                    .enumerate()
                    .map(|(i, g)| {
                        let key = KeyRef::Index(i);
                        g.replace(filter.child(key), ns.child(key)?, ctx)
                    })
                    .collect()
            }
        }
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
    fn visit_into<L: Leaf, F: Filter<L>, C: GraphConsumer<Self, L>>(
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
    fn visit_into_children<L: Leaf, F: Filter<L>, C: ChildrenConsumer<Self, L>>(
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

impl<'g, L: Leaf, B: Builder<L>> Builder<L> for Vec<B> {
    type Graph = Vec<B::Graph>;

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
}

macro_rules! impl_tuple_graph {
    ($($T:ty)*, $($idx:expr)*) => {
        paste::paste! {
            impl<$($T: Graph,)*> Graph for ($($T,)*) {
                type Owned = ($($T::Owned,)*);

                type Builder<L: Leaf> = NodeBuilder<($($T::Builder<L>,)*)>;

                fn builder<'g, L: Leaf, F: Filter<L>>(
                    &'g self,
                    filter: F,
                    mut ctx: &mut GraphContext,
                ) -> Result<Self::Builder<L>, GraphError> {
                    match filter.matches_ref(self) {
                        Ok(_) => Ok(NodeBuilder::Leaf),
                        Err(_) => Ok(NodeBuilder::Node({
                            let ($([<$T:lower>],)*) = self;
                            $(let [<$T:lower>] = [<$T:lower>].builder(
                                filter.child(KeyRef::Index($idx)), &mut ctx
                            )?;)*
                            ($([<$T:lower>],)*)
                        })),
                    }
                }
                fn replace<'g, L: Leaf, F: Filter<L>, S: GraphSource<L::Ref<'g>, L>>(
                    &'g self,
                    filter: F,
                    source: S,
                    mut ctx: &mut GraphContext,
                ) -> Result<Self::Owned, S::Error> {
                    match filter.matches_ref(self) {
                        Ok(r) => Ok(source.leaf(r)?.try_into_value().map_err(|_| GraphError::InvalidLeaf)?),
                        Err(_) => {
                            let ($([<$T:lower>],)*) = self;
                            let mut ns = source.node()?;
                            $(
                                let [<$T:lower>] = [<$T:lower>].replace(
                                    filter.child(KeyRef::Index($idx)), ns.child(KeyRef::Index($idx))?, &mut ctx
                                )?;
                            )*
                            Ok(($([<$T:lower>],)*))
                        }
                    }
                }

                fn visit<'g, L: Leaf, F: Filter<L>, V: GraphVisitor<'g, Self, L>>(
                    &'g self, filter: F, visitor: V
                ) -> V::Output {
                    match filter.matches_ref(self) {
                        Ok(leaf) => visitor.visit_leaf(leaf),
                        Err(graph) => visitor.visit_node(View::new(graph, filter))
                    }
                }
                fn visit_into<L: Leaf, F: Filter<L>, C: GraphConsumer<Self, L>>(
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
                fn visit_into_children<L: Leaf, F: Filter<L>, C: ChildrenConsumer<Self, L>>(
                    self, filter: F, mut consumer: C
                ) -> C::Output {
                    let ($([<$T:lower>],)*) = self;
                    $(
                      consumer.consume_child(Key::Index($idx), Bound::new([<$T:lower>], &filter));
                    )*
                    consumer.finish()
                }
            }
            impl<L: Leaf, $($T: Builder<L>,)*> Builder<L> for ($($T,)*) {
                type Graph = ($($T::Graph,)*);

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

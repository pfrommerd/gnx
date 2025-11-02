use super::*;

impl<T: Graph> Graph for Vec<T> {
    type GraphDef<I: Clone + 'static, R: NonLeafRepr> = NodeDef<I, Vec<T::GraphDef<I, R>>>;
    type Owned = Vec<T::Owned>;

    fn graph_def<I, L, F, M>(
        &self,
        filter: F,
        mut map: M,
        mut ctx: &mut GraphContext,
    ) -> Result<Self::GraphDef<I, F::NonLeafRepr>, GraphError>
    where
        I: Clone + 'static,
        F: GraphFilter<L>,
        M: FnMut(F::Ref<'_>) -> I,
    {
        match filter.try_as_ref(self) {
            Ok(leaf) => Ok(NodeDef::Leaf(map(leaf))),
            Err(graph) => Ok(NodeDef::Node(
                graph
                    .iter()
                    .map(|x| x.graph_def(filter, &mut map, &mut ctx))
                    .collect::<Result<Vec<T::GraphDef<I, F::NonLeafRepr>>, GraphError>>()?,
            )),
        }
    }
    fn into_graph_def<I, L, F, M>(
        self,
        filter: F,
        mut map: M,
        ctx: &mut GraphContext,
    ) -> Result<Self::GraphDef<I, F::NonLeafRepr>, GraphError>
    where
        I: Clone + 'static,
        F: GraphFilter<L>,
        M: FnMut(GraphCow<F::Ref<'_>, L>) -> I,
    {
        match filter.try_to_leaf(self) {
            Ok(leaf) => Ok(NodeDef::Leaf(map(GraphCow::Owned(leaf)))),
            Err(graph) => Ok(NodeDef::Node(
                graph
                    .into_iter()
                    .map(|x| x.into_graph_def(filter, &mut map, ctx))
                    .collect::<Result<Vec<T::GraphDef<I, F::NonLeafRepr>>, GraphError>>()?,
            )),
        }
    }

    fn visit<L, F, V>(&self, filter: F, visitor: impl Into<V>) -> V::Output
    where
        F: GraphFilter<L>,
        V: GraphVisitor<L, F>,
    {
        let visitor = visitor.into();
        match filter.try_as_ref(self) {
            Ok(leaf) => visitor.visit_leaf(Some(leaf)),
            Err(graph) => visitor.visit_node(View::new(graph, filter)),
        }
    }
    fn mut_visit<L, F, V>(&mut self, filter: F, visitor: impl Into<V>) -> V::Output
    where
        F: GraphFilter<L>,
        V: GraphMutVisitor<L, F>,
    {
        let visitor = visitor.into();
        match filter.try_as_mut(self) {
            Ok(leaf) => visitor.visit_leaf_mut(Some(leaf)),
            Err(graph) => visitor.visit_node_mut(ViewMut::new(graph, filter)),
        }
    }
    fn into_visit<L, F, C>(self, filter: F, consumer: impl Into<C>) -> C::Output
    where
        F: GraphFilter<L>,
        C: GraphConsumer<L, F>,
    {
        let consumer = consumer.into();
        match filter.try_to_leaf(self) {
            Ok(leaf) => consumer.consume_leaf(Some(leaf)),
            Err(graph) => consumer.consume_node(Bound::new(graph, filter)),
        }
    }
}

impl<T: Graph> Node for Vec<T> {
    fn visit_children<L, F, V>(&self, filter: F, visitor: impl Into<V>) -> V::Output
    where
        F: GraphFilter<L>,
        V: ChildrenVisitor<L, F>,
    {
        let mut visitor = visitor.into();
        self.iter().enumerate().for_each(|(i, x)| {
            visitor.visit_child(KeyRef::Index(i), View::new(x, filter));
        });
        visitor.finish()
    }
    fn visit_children_mut<L, F, V>(&mut self, filter: F, visitor: impl Into<V>) -> V::Output
    where
        F: GraphFilter<L>,
        V: ChildrenMutVisitor<L, F>,
    {
        let mut visitor = visitor.into();
        self.iter_mut().enumerate().for_each(|(i, x)| {
            visitor.visit_child_mut(KeyRef::Index(i), ViewMut::new(x, filter));
        });
        visitor.finish()
    }
    fn consume_children<L, F, C>(self, filter: F, consumer: impl Into<C>) -> C::Output
    where
        F: GraphFilter<L>,
        C: ChildrenConsumer<L, F>,
    {
        let mut consumer = consumer.into();
        self.into_iter().enumerate().for_each(|(i, x)| {
            consumer.consume_child(Key::Index(i), Bound::new(x, filter));
        });
        consumer.finish()
    }
}

// Vec<D: GraphDef<L>> implements GraphDef<L>

impl<I: Clone + 'static, T: GraphDef<I>> GraphDef<I> for Vec<T> {
    type Graph = Vec<T::Graph>;

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
        let mut ns = source.node();
        self.iter()
            .enumerate()
            .map(|(i, def)| {
                def.build(
                    builder,
                    ns.child(KeyRef::Index(i))?.ok_or(GraphError::MissingNode)?,
                    ctx,
                )
            })
            .collect()
    }
    fn into_build<L, B, S>(
        self,
        builder: B,
        source: S,
        ctx: &mut GraphContext,
    ) -> Result<Self::Graph, S::Error>
    where
        B: LeafBuilder<I, L>,
        S: GraphSource<I, L>,
    {
        let mut ns = source.node();
        self.into_iter()
            .enumerate()
            .map(|(i, def)| {
                def.into_build(
                    builder,
                    ns.child(KeyRef::Index(i))?.ok_or(GraphError::MissingNode)?,
                    ctx,
                )
            })
            .collect()
    }
}

macro_rules! impl_tuple_graph {
    ($($T:ty)*, $($idx:expr)*) => {
        paste::paste! {
            impl<$($T: Graph,)*> Graph for ($($T,)*) {
                type GraphDef<I: Clone + 'static, R: NonLeafRepr> = ($($T::GraphDef<I, R>,)*);
                type Owned = ($($T::Owned,)*);

                #[allow(unused_variables, unused_mut)]
                fn graph_def<I, L, F, M>(&self, filter: F, mut map: M, mut ctx: &mut GraphContext)
                    -> Result<Self::GraphDef<I, F::NonLeafRepr>, GraphError>
                where
                    I: Clone +'static,
                    F: GraphFilter<L>,
                    M: FnMut(F::Ref<'_>) -> I
                {
                    let ($([<$T:lower>],)*) = self;
                    Ok(($([<$T:lower>].graph_def(filter, &mut map, &mut ctx)?,)*))
                }
                #[allow(unused_variables, unused_mut)]
                fn into_graph_def<I, L, F, M>(self, filter: F, mut map: M, mut ctx: &mut GraphContext)
                    -> Result<Self::GraphDef<I, F::NonLeafRepr>, GraphError>
                where
                    I: Clone +'static,
                    F: GraphFilter<L>,
                    M: FnMut(GraphCow<F::Ref<'_>, L>) -> I
                {
                    let ($([<$T:lower>],)*) = self;
                    Ok(($([<$T:lower>].into_graph_def(filter, &mut map, &mut ctx)?,)*))
                }

                fn visit<L, F, V>(&self, filter: F, visitor: impl Into<V>) -> V::Output
                where
                    F: GraphFilter<L>,
                    V: GraphVisitor<L, F>
                {
                    let visitor = visitor.into();
                    match filter.try_as_ref(self) {
                        Ok(leaf) => visitor.visit_leaf(Some(leaf)),
                        Err(graph) => visitor.visit_node(View::new(graph, filter))
                    }
                }
                fn mut_visit<L, F, V>(&mut self, filter: F, visitor: impl Into<V>) -> V::Output
                where
                    F: GraphFilter<L>,
                    V: GraphMutVisitor<L, F>
                {
                    let visitor = visitor.into();
                    match filter.try_as_mut(self) {
                        Ok(leaf) => visitor.visit_leaf_mut(Some(leaf)),
                        Err(graph) => visitor.visit_node_mut(ViewMut::new(graph, filter))
                    }
                }
                fn into_visit<L, F, C>(self, filter: F, consumer: impl Into<C>) -> C::Output
                where
                    F: GraphFilter<L>,
                    C: GraphConsumer<L, F>
                {
                    let consumer = consumer.into();
                    match filter.try_to_leaf(self) {
                        Ok(leaf) => consumer.consume_leaf(Some(leaf)),
                        Err(graph) => consumer.consume_node(Bound::new(graph, filter))
                    }
                }
            }
            impl<$($T: Graph,)*> Node for ($($T,)*) {
                #[allow(unused_variables, unused_mut)]
                fn visit_children<L, F, V>(&self, filter: F, visitor: impl Into<V>) -> V::Output
                where
                    F: GraphFilter<L>,
                    V: ChildrenVisitor<L, F>
                {
                    let mut visitor = visitor.into();
                    let ($([<$T:lower>],)*) = self;
                    $(
                      visitor.visit_child(KeyRef::Index($idx), View::new([<$T:lower>], filter));
                    )*
                    visitor.finish()
                }
                #[allow(unused_variables, unused_mut)]
                fn visit_children_mut<L, F, V>(&mut self, filter: F, visitor: impl Into<V>) -> V::Output
                where
                    F: GraphFilter<L>,
                    V: ChildrenMutVisitor<L, F>
                {
                    let mut visitor = visitor.into();
                    let ($([<$T:lower>],)*) = self;
                    $(
                      visitor.visit_child_mut(KeyRef::Index($idx), ViewMut::new([<$T:lower>], filter));
                    )*
                    visitor.finish()
                }
                #[allow(unused_variables, unused_mut)]
                fn consume_children<L, F, C>(self, filter: F, consumer: impl Into<C>) -> C::Output
                where
                    F: GraphFilter<L>,
                    C: ChildrenConsumer<L, F>
                {
                    let mut consumer = consumer.into();
                    let ($([<$T:lower>],)*) = self;
                    $(
                      consumer.consume_child(Key::Index($idx), Bound::new([<$T:lower>], filter));
                    )*
                    consumer.finish()
                }
            }
            impl<I: Clone + 'static, $($T: GraphDef<I>,)*> GraphDef<I> for ($($T,)*) {
                type Graph = ($($T::Graph,)*);

                #[allow(unused_variables, unused_mut)]
                fn build<L, B, S>(
                    &self,
                    builder: B,
                    mut source: S,
                    mut ctx: &mut GraphContext,
                ) -> Result<Self::Graph, S::Error>
                where
                    B: LeafBuilder<I, L>,
                    S: GraphSource<I, L>,
                {
                    let ($([<$T:lower>],)*) = self;
                    let mut ns = source.node();
                    $(
                        let [<$T:lower>] = [<$T:lower>].build(
                            builder, ns.child(KeyRef::Index($idx))?.ok_or(
                                GraphError::MissingNode
                            )?, &mut ctx
                        )?;
                    )*
                    Ok(($([<$T:lower>],)*))
                }

                #[allow(unused_variables, unused_mut)]
                fn into_build<L, B, S>(
                    self,
                    builder: B,
                    mut source: S,
                    mut ctx: &mut GraphContext,
                ) -> Result<Self::Graph, S::Error>
                where
                    B: LeafBuilder<I, L>,
                    S: GraphSource<I, L>,
                {
                    let ($([<$T:lower>],)*) = self;
                    let mut ns = source.node();
                    $(
                        let [<$T:lower>] = [<$T:lower>].into_build(
                            builder, ns.child(KeyRef::Index($idx))?.ok_or(
                                GraphError::MissingNode
                            )?, &mut ctx
                        )?;
                    )*
                    Ok(($([<$T:lower>],)*))
                }
            }
        }
    };
}

impl_tuple_graph!(,);
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

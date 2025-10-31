// Blanket Graph implementations for common rust types
use super::*;

use std::rc::Rc;
use std::sync::Arc;

// Builtin types

#[derive(Clone)]
pub enum LeafDef<I, T> {
    Leaf(I),
    Static(T),
}

impl<I, T: Graph + Clone> GraphDef<I> for LeafDef<I, T> {
    type Graph = T;

    fn visit<V: DefVisitor<I>>(&self, visitor: V) -> V::Output {
        match self {
            LeafDef::Leaf(leaf) => visitor.leaf(Some(leaf)),
            LeafDef::Static(_) => visitor.leaf(None),
        }
    }
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
            LeafDef::Static(value) => Ok(value.clone()),
        }
    }
}

#[derive(Clone)]
pub enum BoundDef<I, N: NodeDef<I>> {
    Leaf(I),
    Node(N),
}

impl<I, N: NodeDef<I>> GraphDef<I> for BoundDef<I, N> {
    type Graph = N::Graph;

    fn visit<V: DefVisitor<I>>(&self, visitor: V) -> V::Output {
        match self {
            BoundDef::Leaf(leaf) => visitor.leaf(Some(leaf)),
            BoundDef::Node(node) => visitor.node(node),
        }
    }
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
            BoundDef::Leaf(leaf) => Ok(builder.try_build(self, source.leaf(leaf)?)?),
            BoundDef::Node(node) => node.build(builder, source, ctx),
        }
    }
}

macro_rules! basic_impl {
    ($T:ty) => {
        impl Graph for $T {
            type GraphDef<I> = LeafDef<I, $T>;
            type Owned = $T;

            fn graph_def<I, L, V, F>(
                &self,
                viewer: V,
                mut map: F,
                _ctx: &mut GraphContext,
            ) -> Result<Self::GraphDef<I>, GraphError>
            where
                V: GraphViewer<L>,
                F: FnMut(V::Ref<'_>) -> I,
            {
                match viewer.try_as_leaf(self) {
                    Ok(leaf) => Ok(LeafDef::Leaf(map(leaf))),
                    Err(_graph) => Ok(LeafDef::Static(self.clone())),
                }
            }
            fn visit<L, V, M>(&self, view: V, visitor: M) -> M::Output
            where
                V: GraphViewer<L>,
                M: GraphVisitor<L, V>,
            {
                match view.try_as_leaf(self) {
                    Ok(leaf) => visitor.leaf(Some(leaf)),
                    Err(_graph) => visitor.leaf(None),
                }
            }
            fn map<L, V, M>(self, view: V, map: M) -> M::Output
            where
                V: GraphViewer<L>,
                M: GraphMap<L, V>,
            {
                match view.try_to_leaf(self) {
                    Ok(leaf) => map.leaf(Some(leaf)),
                    Err(_graph) => map.leaf(None),
                }
            }
        }
    };
}

basic_impl!(u8);

impl<T: Graph> Graph for &T {
    type GraphDef<I> = Arc<T::GraphDef<I>>;
    type Owned = Arc<T::Owned>;

    fn graph_def<I, L, V, F>(
        &self,
        viewer: V,
        map: F,
        ctx: &mut GraphContext,
    ) -> Result<Self::GraphDef<I>, GraphError>
    where
        V: GraphViewer<L>,
        F: FnMut(V::Ref<'_>) -> I,
    {
        let id: GraphId = ((*self) as *const T as u64).into();
        ctx_build_shared!(ctx, Self::GraphDef<I>, id, {
            let g = T::graph_def(self, viewer, map, ctx);
            g.map(Arc::new)
        })
    }

    fn visit<L, V, M>(&self, view: V, visitor: M) -> M::Output
    where
        V: GraphViewer<L>,
        M: GraphVisitor<L, V>,
    {
        let id: GraphId = ((*self) as *const T as u64).into();
        visitor.shared::<T>(id, View::new(self, view))
    }
    fn map<L, V, M>(self, view: V, map: M) -> M::Output
    where
        V: GraphViewer<L>,
        M: GraphMap<L, V>,
    {
        let id: GraphId = (self as *const T as u64).into();
        map.shared::<T>(id, View::new(self, view))
    }
}

macro_rules! impl_rc {
    ($W:ident) => {
        // Arc<T: Graph<L>> implements Graph<L>
        impl<T: Graph> Graph for $W<T> {
            type GraphDef<I> = $W<T::GraphDef<I>>;
            type Owned = $W<T::Owned>;

            fn graph_def<I, L, V, F>(
                &self,
                viewer: V,
                map: F,
                ctx: &mut GraphContext,
            ) -> Result<Self::GraphDef<I>, GraphError>
            where
                V: GraphViewer<L>,
                F: FnMut(V::Ref<'_>) -> I,
            {
                let id: GraphId = ($W::as_ptr(self) as u64).into();
                ctx_build_shared!(ctx, Self::GraphDef<I>, id, {
                    let v = T::graph_def(self, viewer, map, ctx);
                    v.map($W::new)
                })
            }
            fn visit<L, V, M>(&self, viewer: V, visitor: M) -> M::Output
            where
                V: GraphViewer<L>,
                M: GraphVisitor<L, V>,
            {
                let id: GraphId = (($W::as_ptr(self) as *const T) as u64).into();
                visitor.shared::<T>(id, View::new(self, viewer))
            }
            fn map<L, V, M>(self, viewer: V, map: M) -> M::Output
            where
                V: GraphViewer<L>,
                M: GraphMap<L, V>,
            {
                let id: GraphId = (($W::as_ptr(&self) as *const T) as u64).into();
                map.shared::<T>(id, View::new(&self, viewer))
            }
        }
        impl<I, T: GraphDef<I>> GraphDef<I> for $W<T> {
            type Graph = $W<T::Graph>;

            fn visit<V: DefVisitor<I>>(&self, visitor: V) -> V::Output {
                let id: GraphId = (($W::as_ptr(self) as *const T) as u64).into();
                visitor.shared::<T>(id, self)
            }
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
                let id: GraphId = (($W::as_ptr(self) as *const T) as u64).into();
                ctx_build_shared!(ctx, Self::Graph, id, {
                    let g = (**self).build(builder, source, ctx)?;
                    Ok($W::new(g))
                })
            }
        }
    };
}

impl_rc!(Rc);
impl_rc!(Arc);

// Vec<T: Graph<L>> implements Graph<L>
impl<T: Graph> Graph for Vec<T> {
    type GraphDef<I> = BoundDef<I, Vec<T::GraphDef<I>>>;
    type Owned = Vec<T::Owned>;

    fn graph_def<I, L, V, F>(
        &self,
        viewer: V,
        mut map: F,
        mut ctx: &mut GraphContext,
    ) -> Result<Self::GraphDef<I>, GraphError>
    where
        V: GraphViewer<L>,
        F: FnMut(V::Ref<'_>) -> I,
    {
        match viewer.try_as_leaf(self) {
            Ok(leaf) => Ok(BoundDef::Leaf(map(leaf))),
            Err(_graph) => Ok(BoundDef::Node(
                self.iter()
                    .map(|x| x.graph_def(viewer, &mut map, &mut ctx))
                    .collect::<Result<Vec<T::GraphDef<I>>, GraphError>>()?,
            )),
        }
    }
    fn visit<L, V, M>(&self, viewer: V, visitor: M) -> M::Output
    where
        V: GraphViewer<L>,
        M: GraphVisitor<L, V>,
    {
        match viewer.try_as_leaf(self) {
            Ok(leaf) => visitor.leaf(Some(leaf)),
            Err(graph) => visitor.node(View::new(graph, viewer)),
        }
    }
    fn map<L, V, M>(self, viewer: V, map: M) -> M::Output
    where
        V: GraphViewer<L>,
        M: GraphMap<L, V>,
    {
        match viewer.try_to_leaf(self) {
            Ok(leaf) => map.leaf(Some(leaf)),
            Err(graph) => map.node(Bound::new(graph, viewer)),
        }
    }
}

impl<T: Graph> Node for Vec<T> {
    fn visit_children<L, V, M>(&self, viewer: V, mut visitor: M) -> M::Output
    where
        V: GraphViewer<L>,
        M: ChildrenVisitor<L, V>,
    {
        self.iter().enumerate().for_each(|(i, x)| {
            visitor.child(KeyRef::Index(i), View::new(x, viewer));
        });
        visitor.finish()
    }
    fn map_children<L, V, M>(self, viewer: V, mut map: M) -> M::Output
    where
        V: GraphViewer<L>,
        M: ChildrenMap<L, V>,
    {
        self.into_iter().enumerate().for_each(|(i, x)| {
            map.child(Key::Index(i), Bound::new(x, viewer));
        });
        map.finish()
    }
}

// Vec<D: GraphDef<L>> implements GraphDef<L>

impl<I, T: GraphDef<I>> GraphDef<I> for Vec<T> {
    type Graph = Vec<T::Graph>;

    fn visit<V>(&self, visitor: V) -> V::Output
    where
        V: DefVisitor<I>,
    {
        visitor.node(self)
    }
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
}

impl<I, T: GraphDef<I>> NodeDef<I> for Vec<T> {
    fn visit_children<V: DefVisitorChildren<I>>(&self, mut visitor: V) -> V::Output {
        self.iter().enumerate().for_each(|(i, x)| {
            visitor.child(KeyRef::Index(i), x);
        });
        visitor.finish()
    }
}

// For arrays we can now use const generics to implement for all sizes!

// Macro to implement for tuples up to arity 10
macro_rules! impl_tuple_graph {
    ($($T:ty)*, $($idx:expr)*) => {
        paste::paste! {
            impl<$($T: Graph,)*> Graph for ($($T,)*) {
                type GraphDef<I> = ($($T::GraphDef<I>,)*);
                type Owned = ($($T::Owned,)*);

                #[allow(unused_variables, unused_mut)]
                fn graph_def<I, L, V, F>(&self, viewer: V, mut map: F, mut ctx: &mut GraphContext)
                                        -> Result<Self::GraphDef<I>, GraphError>
                        where V: GraphViewer<L>, F: FnMut(V::Ref<'_>) -> I {
                    let ($([<$T:lower>],)*) = self;
                    Ok(($([<$T:lower>].graph_def(viewer, &mut map, &mut ctx)?,)*))
                }
                fn visit<L, V, M>(&self, view: V, visitor: M) -> M::Output
                where
                    V: GraphViewer<L>,
                    M: GraphVisitor<L, V>
                {
                    match view.try_as_leaf(self) {
                        Ok(leaf) => visitor.leaf(Some(leaf)),
                        Err(graph) => visitor.node(View::new(graph, view))
                    }
                }
                fn map<L, V, M>(self, view: V, map: M) -> M::Output
                where
                    V: GraphViewer<L>,
                    M: GraphMap<L, V>
                {
                    match view.try_to_leaf(self) {
                        Ok(leaf) => map.leaf(Some(leaf)),
                        Err(graph) => map.node(Bound::new(graph, view))
                    }
                }
            }
            impl<$($T: Graph,)*> Node for ($($T,)*) {
                #[allow(unused_variables, unused_mut)]
                fn visit_children<L, V, M>(&self, viewer: V, mut visitor: M) -> M::Output
                where
                    V: GraphViewer<L>,
                    M: ChildrenVisitor<L, V>
                {
                    let ($([<$T:lower>],)*) = self;
                    $(
                      visitor.child(KeyRef::Index($idx), View::new([<$T:lower>], viewer));
                    )*
                    visitor.finish()
                }
                #[allow(unused_variables, unused_mut)]
                fn map_children<L, V, M>(self, viewer: V, mut map: M) -> M::Output
                where
                    V: GraphViewer<L>,
                    M: ChildrenMap<L, V>
                {
                    let ($([<$T:lower>],)*) = self;
                    $(
                      map.child(Key::Index($idx), Bound::new([<$T:lower>], viewer));
                    )*
                    map.finish()
                }
            }
            impl<I, $($T: GraphDef<I>,)*> GraphDef<I> for ($($T,)*) {
                type Graph= ($($T::Graph,)*);

                fn visit<V>(&self, visitor: V) -> V::Output
                where
                    V: DefVisitor<I>
                {
                    visitor.node(self)
                }
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
            }
            impl<I, $($T: GraphDef<I>,)*> NodeDef<I> for ($($T,)*) {
                #[allow(unused_variables, unused_mut)]
                fn visit_children<M>(&self, mut visitor: M) -> M::Output
                where
                    M: DefVisitorChildren<I>,
                {
                    let ($([<$T:lower>],)*) = self;
                    $(
                    visitor.child(KeyRef::Index($idx), [<$T:lower>]);
                    )*
                    visitor.finish()
                }
            }
        }
    };
}

impl_tuple_graph!(,);
impl_tuple_graph!(A, 0);
impl_tuple_graph!(A BB, 0 1);
impl_tuple_graph!(A BB C, 0 1 2);
impl_tuple_graph!(A BB C D, 0 1 2 3);
impl_tuple_graph!(A BB C D E, 0 1 2 3 4);
impl_tuple_graph!(A BB C D E FF, 0 1 2 3 4 5);
impl_tuple_graph!(A BB C D E FF G, 0 1 2 3 4 5 6);
impl_tuple_graph!(A BB C D E FF G H, 0 1 2 3 4 5 6 7);
impl_tuple_graph!(A BB C D E FF G H II, 0 1 2 3 4 5 6 7 8);
impl_tuple_graph!(A BB C D E FF G H II J, 0 1 2 3 4 5 6 7 8 9);
impl_tuple_graph!(A BB C D E FF G H II J K, 0 1 2 3 4 5 6 7 8 9 10);
impl_tuple_graph!(A BB C D E FF G H II J K LL, 0 1 2 3 4 5 6 7 8 9 10 11);

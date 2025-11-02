macro_rules! impl_rc {
    ($W:ident) => {
        // Arc<T: Graph<L>> implements Graph<L>
        impl<T: Graph> Graph for $W<T> {
            type GraphDef<I: Clone + 'static, R: NonLeafRepr> = $W<T::GraphDef<I, R>>;
            type Owned = $W<T::Owned>;

            fn graph_def<I, L, V, F>(
                &self,
                viewer: V,
                map: F,
                ctx: &mut GraphContext,
            ) -> Result<Self::GraphDef<I, V::NonLeafRepr>, GraphError>
            where
                I: Clone + 'static,
                V: GraphFilter<L>,
                F: FnMut(V::Ref<'_>) -> I,
            {
                let id: GraphId = ($W::as_ptr(self) as u64).into();
                ctx_build_shared!(ctx, Self::GraphDef<I, V::NonLeafRepr>, id, {
                    let v = T::graph_def(self, viewer, map, ctx);
                    v.map($W::new)
                })
            }
            fn visit<L, V, M>(&self, viewer: V, visitor: M) -> M::Output
            where
                V: GraphFilter<L>,
                M: GraphVisitor<L, V>,
            {
                let id: GraphId = (($W::as_ptr(self) as *const T) as u64).into();
                visitor.shared::<T>(id, View::new(self, viewer))
            }
            fn map<L, V, M>(self, viewer: V, map: M) -> M::Output
            where
                V: GraphFilter<L>,
                M: GraphMap<L, V>,
            {
                let id: GraphId = (($W::as_ptr(&self) as *const T) as u64).into();
                map.shared::<T>(id, View::new(&self, viewer))
            }
        }
        impl<I: Clone + 'static, T: GraphDef<I>> GraphDef<I> for $W<T> {
            type Graph = $W<T::Graph>;

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

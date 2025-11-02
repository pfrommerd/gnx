use super::*;

impl<T: Graph> Graph for &T {
    type GraphDef<I: Clone + 'static, R: NonLeafRepr> = Arc<T::GraphDef<I, R>>;
    type Owned = Arc<T::Owned>;

    fn graph_def<I, L, F, M>(
        &self,
        viewer: F,
        map: M,
        ctx: &mut GraphContext,
    ) -> Result<Self::GraphDef<I, F::NonLeafRepr>, GraphError>
    where
        I: Clone + 'static,
        F: GraphFilter<L>,
        M: FnMut(F::Ref<'_>) -> I,
    {
        let id: GraphId = ((*self) as *const T as u64).into();
        ctx_build_shared!(ctx, Self::GraphDef<I, V::NonLeafRepr>, id, {
            let g = T::graph_def(self, viewer, map, ctx);
            g.map(Arc::new)
        })
    }

    fn visit<L, V, M>(&self, view: V, visitor: M) -> M::Output
    where
        V: GraphFilter<L>,
        M: GraphVisitor<L, V>,
    {
        let id: GraphId = ((*self) as *const T as u64).into();
        visitor.shared::<T>(id, View::new(self, view))
    }

    fn into_visit<L, V, M>(self, view: V, map: M) -> M::Output
    where
        V: GraphFilter<L>,
        M: GraphMap<L, V>,
    {
        let id: GraphId = (self as *const T as u64).into();
        map.shared::<T>(id, View::new(self, view))
    }
}

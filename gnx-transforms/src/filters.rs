//! Conversions at transform setters. Builders store the owned graph filter itself.
use crate::{TransformError, TransformResult};
use gnx_graph::{
    All, Filter, Graph, Key, KeyRef, Leaf,
    util::{Dag, DagNode, GraphExt},
};
use std::{marker::PhantomData, sync::Arc};

/// Assign a fixed axis to every match of a graph filter.
#[derive(Clone)]
pub struct OnAxis<F>(usize, F);

pub fn on_axis<F>(axis: usize, filter: F) -> OnAxis<F> {
    OnAxis(axis, filter)
}

impl<L: Leaf, F: Filter<L>> Filter<L> for OnAxis<F> {
    type Annotation<'g> = usize;
    type Owned = OnAxis<F::Owned>;
    type ChildFilter<'s>
        = OnAxis<F::ChildFilter<'s>>
    where
        Self: 's;
    fn owned(&self) -> Self::Owned {
        on_axis(self.0, self.1.owned())
    }
    fn annotate<'g>(&self, _: L::Ref<'g>) -> Self::Annotation<'g> {
        self.0
    }
    fn matches_ref<'g, G: Graph>(&self, graph: &'g G) -> Result<L::Ref<'g>, &'g G> {
        self.1.matches_ref(graph)
    }
    fn matches_value<G: Graph>(&self, graph: G) -> Result<L, G> {
        self.1.matches_value(graph)
    }
    fn child<'s>(&'s self, key: KeyRef<'s>) -> Self::ChildFilter<'s> {
        on_axis(self.0, self.1.child(key))
    }
}

/// Type witness: a filter may implement `Filter<L>` for more than one `L`.
#[doc(hidden)]
pub trait FilterType {
    type Leaf: Leaf;
    type Filter: Filter<Self::Leaf>;
    fn validate<G: Graph>(_: &Self::Filter, _: &G, _: &[Vec<Key>], _: &str) -> TransformResult<()> {
        Ok(())
    }
}
#[doc(hidden)]
pub trait AxisType: FilterType<Filter: for<'g> Filter<Self::Leaf, Annotation<'g> = usize>> {}
impl<B: FilterType> AxisType for B where B::Filter: for<'g> Filter<B::Leaf, Annotation<'g> = usize> {}

/// Select the leaf type explicitly for a filter that implements `Filter` more than once.
pub struct ByFilter<L, F = ()>(PhantomData<fn() -> (L, F)>);
impl<L: Leaf, F: Filter<L>> FilterType for ByFilter<L, F> {
    type Leaf = L;
    type Filter = F;
}
pub type DefaultAxes = ByFilter<(), OnAxis<All>>;
pub type DefaultWrt = ByFilter<(), All>;

/// Convert an axis number, positional axis DAG, or annotated filter into a graph filter.
pub trait IntoAxisFilter<K> {
    type Binding: AxisType;
    fn into_axis_filter(self) -> <Self::Binding as FilterType>::Filter;
}
/// Convert argument indices, a boolean mask, or a graph filter into a differentiation filter.
pub trait IntoWrtFilter<K> {
    type Binding: FilterType;
    fn into_wrt_filter(self) -> <Self::Binding as FilterType>::Filter;
}
impl<L: Leaf, F: for<'g> Filter<L, Annotation<'g> = usize>> IntoAxisFilter<ByFilter<L>> for F {
    type Binding = ByFilter<L, F::Owned>;
    fn into_axis_filter(self) -> F::Owned {
        self.owned()
    }
}
impl<L: Leaf, F: Filter<L>> IntoWrtFilter<ByFilter<L>> for F {
    type Binding = ByFilter<L, F::Owned>;
    fn into_wrt_filter(self) -> F::Owned {
        self.owned()
    }
}
#[doc(hidden)]
pub struct Positional;
impl IntoAxisFilter<Positional> for usize {
    type Binding = DefaultAxes;
    fn into_axis_filter(self) -> OnAxis<All> {
        on_axis(self, All)
    }
}
/// A positional mask payload: `None` never matches, `Some` selects the subtree.
#[doc(hidden)]
#[derive(Clone)]
pub struct Gate<T>(Option<T>);
impl<T: Copy + 'static> Filter<()> for Gate<T> {
    type Owned = Self;
    type ChildFilter<'s> = Self;
    type Annotation<'g> = T;
    fn owned(&self) -> Self {
        self.clone()
    }
    fn child<'s>(&'s self, _: KeyRef<'s>) -> Self {
        self.clone()
    }
    fn matches_ref<'g, G: Graph>(&self, g: &'g G) -> Result<&'g (), &'g G> {
        self.0.map(|_| &()).ok_or(g)
    }
    fn matches_value<G: Graph>(&self, g: G) -> Result<(), G> {
        self.0.map(|_| ()).ok_or(g)
    }
    fn annotate<'g>(&self, _: <() as Leaf>::Ref<'g>) -> Self::Annotation<'g> {
        self.0.expect("only matched leaves are annotated")
    }
}
/// Preserve strict shape checking for positional DAGs, including explicit empty leaves.
pub struct PositionalDag<T>(PhantomData<T>);
impl<T: Copy + 'static> FilterType for PositionalDag<T> {
    type Leaf = ();
    type Filter = Dag<Gate<T>>;
    fn validate<G: Graph>(
        f: &Self::Filter,
        _: &G,
        paths: &[Vec<Key>],
        name: &str,
    ) -> TransformResult<()> {
        crate::spec::validate_dag(f, paths, name)
    }
}
impl IntoAxisFilter<Positional> for Dag<usize> {
    type Binding = PositionalDag<usize>;
    fn into_axis_filter(self) -> Dag<Gate<usize>> {
        self.map_leaves(|&axis| Gate(Some(axis)))
    }
}
impl IntoWrtFilter<Positional> for bool {
    type Binding = ByFilter<(), Gate<()>>;
    fn into_wrt_filter(self) -> Gate<()> {
        Gate(self.then_some(()))
    }
}
impl IntoWrtFilter<Positional> for Dag<bool> {
    type Binding = PositionalDag<()>;
    fn into_wrt_filter(self) -> Dag<Gate<()>> {
        self.map_leaves(|&yes| Gate(yes.then_some(())))
    }
}

/// Select complete top-level positional arguments. A child gate never inspects nested indices.
#[derive(Clone)]
pub struct Arguments(Option<Arc<[usize]>>);
impl Arguments {
    fn new(mut indices: Vec<usize>) -> Self {
        indices.sort_unstable();
        indices.dedup();
        Self(Some(indices.into()))
    }
}
impl Filter<()> for Arguments {
    type Owned = Self;
    type ChildFilter<'s> = Gate<()>;
    type Annotation<'g> = ();
    fn owned(&self) -> Self {
        self.clone()
    }
    fn matches_ref<'g, G: Graph>(&self, g: &'g G) -> Result<&'g (), &'g G> {
        Err(g)
    }
    fn matches_value<G: Graph>(&self, g: G) -> Result<(), G> {
        Err(g)
    }
    fn annotate<'g>(&self, _: <() as Leaf>::Ref<'g>) -> Self::Annotation<'g> {}
    fn child<'s>(&'s self, key: KeyRef<'s>) -> Gate<()> {
        Gate(match (&self.0, key) {
            (None, _) => Some(()),
            (Some(indices), KeyRef::Index(i)) => usize::try_from(i)
                .ok()
                .filter(|n| indices.binary_search(n).is_ok())
                .map(|_| ()),
            _ => None,
        })
    }
}
impl FilterType for Arguments {
    type Leaf = ();
    type Filter = Self;
    fn validate<G: Graph>(f: &Self, graph: &G, _: &[Vec<Key>], _: &str) -> TransformResult<()> {
        let Some(indices) = &f.0 else { return Ok(()) };
        if indices.is_empty() {
            return Ok(());
        }
        // Inspect only the argument keys, including static or leafless arguments.
        let root = graph.to_dag(Arguments(None), |_| ());
        for &i in indices.iter() {
            let exists = match &root {
                Dag::Node(DagNode::List(items)) => i < items.len(),
                Dag::Node(DagNode::Map(m)) => {
                    isize::try_from(i).is_ok_and(|k| m.contains_key(&Key::Index(k)))
                }
                _ => false,
            };
            if !exists {
                return Err(TransformError::invalid(format!(
                    "wrt: argument index {i} is out of range"
                )));
            }
        }
        Ok(())
    }
}
impl IntoWrtFilter<Positional> for usize {
    type Binding = Arguments;
    fn into_wrt_filter(self) -> Arguments {
        Arguments::new(vec![self])
    }
}
#[doc(hidden)]
pub struct ArgumentList;
impl<I: AsRef<[usize]>> IntoWrtFilter<ArgumentList> for I {
    type Binding = Arguments;
    fn into_wrt_filter(self) -> Arguments {
        Arguments::new(self.as_ref().to_vec())
    }
}

// Flat bracket options in transform macros can also be legacy boolean/filter masks.
#[doc(hidden)]
pub struct MaskSlice<K>(PhantomData<K>);
impl<T: Clone, K> IntoWrtFilter<MaskSlice<K>> for &[T]
where
    Dag<T>: IntoWrtFilter<K>,
{
    type Binding = <Dag<T> as IntoWrtFilter<K>>::Binding;
    fn into_wrt_filter(self) -> <Self::Binding as FilterType>::Filter {
        Dag::Node(DagNode::List(
            self.iter()
                .map(|v| gnx_graph::util::DagChild::Owned(Dag::from(v.clone())))
                .collect(),
        ))
        .into_wrt_filter()
    }
}

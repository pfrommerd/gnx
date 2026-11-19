use std::borrow::Cow;
use std::convert::Infallible;
use std::sync::Arc;

use gnx_expr::{Kind, ObjectTracer};
use gnx_graph::util::Dag;
use gnx_graph::{Graph, GraphEq, GraphHash};

use crate::cache::{TraceEntry, TracerBuilder};
use crate::error::ApplyError;
use crate::spec::resolve;
use crate::traced::{Leaves, body, value};
use crate::{Callable, Function, Jit, Jitable, TransformError, TryCallable, TryJitable, jit, util};

/// A function vectorized over mapped axes of its arguments; see [`map`].
pub struct Map<I: Graph, F: Function<I>> {
    jit: Jit<I, F>,
    in_axes: Dag<usize>,
    out_axes: Dag<usize>,
    axis_size: Option<usize>,
}

/// Vectorize `func` (JAX's `vmap`). By default every argument leaf is mapped
/// along axis 0 and every result is stacked along axis 0:
///
/// ```ignore
/// let rows = map(|x: Array, w: Array| f(x, w)).in_axes(dag!(0, *));
/// ```
///
/// The function is traced once per argument signature, at the shape of one
/// instance.
pub fn map<I: Graph, F: Function<I>>(func: F) -> Map<I, F> {
    Map {
        jit: jit(func),
        in_axes: Dag::from(0),
        out_axes: Dag::from(0),
        axis_size: None,
    }
}

impl<I: Graph, F: Function<I>> Map<I, F> {
    /// The mapped axis of each argument leaf, as a prefix of the arguments.
    /// `*` leaves an argument unmapped (shared by every instance).
    pub fn in_axes(mut self, axes: impl Into<Dag<usize>>) -> Self {
        self.in_axes = axes.into();
        self
    }

    /// The axis along which each result leaf is stacked.
    pub fn out_axes(mut self, axes: impl Into<Dag<usize>>) -> Self {
        self.out_axes = axes.into();
        self
    }

    /// The number of instances; required when no argument is mapped.
    pub fn axis_size(mut self, size: usize) -> Self {
        self.axis_size = Some(size);
        self
    }

    /// Label the traced body for display.
    pub fn name(mut self, name: impl Into<Cow<'static, str>>) -> Self {
        self.jit.set_name_hint(name.into());
        self
    }
}

impl<I, F> Map<I, F>
where
    I: Graph,
    I::Owned: GraphHash + GraphEq<Builder<ObjectTracer> = TracerBuilder<I>>,
    F: Function<I>,
    F::Output: Graph<Owned = F::Output>,
{
    fn apply<A, E>(
        &self,
        input: A,
        trace: impl FnOnce(TracerBuilder<I>, Vec<Kind>) -> Result<Arc<TraceEntry<F::Output>>, E>,
    ) -> Result<F::Output, ApplyError<E>>
    where
        A: Graph<Builder<ObjectTracer> = TracerBuilder<I>>,
    {
        let leaves = Leaves::<I>::of(&input);
        let axes = resolve(&self.in_axes, &leaves.paths, "in_axes")?;
        let mut kinds = Vec::new();
        let mut captured = Vec::new();
        let mut mapped = Vec::new();
        let mut in_axes = Vec::new();
        for (object, axis) in leaves.objects.iter().zip(&axes) {
            match axis.value {
                None => {
                    kinds.push(crate::cache::kind(object));
                    captured.push(true);
                }
                Some(axis) => {
                    let trace = value(object, "a mapped argument")?;
                    let (_, instance) = util::without_axis(trace.info(), axis)?;
                    kinds.push(Kind::Value(instance));
                    captured.push(false);
                    mapped.push(trace);
                    in_axes.push(axis);
                }
            }
        }
        let builder = leaves.builder.clone();
        let entry = trace(builder, kinds).map_err(ApplyError::Body)?;
        let out_axes = resolve(&self.out_axes, &entry.output_paths, "out_axes")?
            .into_iter()
            .map(|axis| {
                axis.value.ok_or_else(|| {
                    TransformError::invalid("out_axes must give an axis for every result")
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let body = body(&entry, &leaves.objects, &captured);
        let results = super::region::map(body, &mapped, &in_axes, &out_axes, self.axis_size)
            .map_err(TransformError::from)?;
        Ok(crate::cache::rebuild_graph(
            &entry.output_builder,
            results.into_iter().map(ObjectTracer::from),
        ))
    }
}

impl<I: Graph, F: Function<I>> Function<I> for Map<I, F> {
    type Output = F::Output;
}

impl<I, F> Callable<I> for Map<I, F>
where
    I: Graph,
    I::Owned: GraphHash + GraphEq<Builder<ObjectTracer> = TracerBuilder<I>>,
    F: Jitable<I>,
{
    fn invoke(&self, input: I) -> Self::Output {
        self.apply(input, |builder, kinds| {
            Ok::<_, Infallible>(self.jit.traced(builder, kinds))
        })
        .unwrap_or_else(|error| panic!("map: {error}"))
    }
}

impl<I, F> Jitable<I> for Map<I, F>
where
    I: Graph,
    I::Owned: GraphHash + GraphEq<Builder<ObjectTracer> = TracerBuilder<I>>,
    F: Jitable<I>,
{
    fn invoke_owned(&self, input: I::Owned) -> Self::Output {
        self.apply(input, |builder, kinds| {
            Ok::<_, Infallible>(self.jit.traced(builder, kinds))
        })
        .unwrap_or_else(|error| panic!("map: {error}"))
    }
}

impl<I, F> TryCallable<I> for Map<I, F>
where
    I: Graph,
    I::Owned: GraphHash + GraphEq<Builder<ObjectTracer> = TracerBuilder<I>>,
    F: TryJitable<I>,
{
    type Error = ApplyError<F::Error>;

    fn try_invoke(&self, input: I) -> Result<Self::Output, Self::Error> {
        self.apply(input, |builder, kinds| self.jit.try_traced(builder, kinds))
    }
}

impl<I, F> TryJitable<I> for Map<I, F>
where
    I: Graph,
    I::Owned: GraphHash + GraphEq<Builder<ObjectTracer> = TracerBuilder<I>>,
    F: TryJitable<I>,
{
    fn try_invoke_owned(&self, input: I::Owned) -> Result<Self::Output, Self::Error> {
        self.apply(input, |builder, kinds| self.jit.try_traced(builder, kinds))
    }
}

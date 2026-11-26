use std::borrow::Cow;
use std::sync::Arc;

use crate::on_axis;
use gnx_expr::{Kind, ObjectTracer, VarInfo};
use gnx_graph::All;
use gnx_graph::{Graph, GraphEq, GraphHash};

use crate::cache::{TraceEntry, TracerBuilder};
use crate::error::{ApplyError, TransformError};
use crate::filters::{AxisType, DefaultAxes, IntoAxisFilter};
use crate::traced::{Leaves, body, value};
use crate::{Callable, Function, Jit, Jitable, TryCallable, TryJitable, jit, util};

/// A function vectorized over mapped axes of its arguments; see [`map`].
pub struct Map<I: Graph, F: Function<I>, X: AxisType = DefaultAxes, Y: AxisType = DefaultAxes> {
    jit: Jit<I, F>,
    in_axes: X::Filter,
    out_axes: Y::Filter,
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
        in_axes: on_axis(0, All),
        out_axes: on_axis(0, All),
        axis_size: None,
    }
}

impl<I: Graph, F: Function<I>, X: AxisType, Y: AxisType> Map<I, F, X, Y> {
    /// The mapped axis of each argument leaf, as a prefix of the arguments.
    /// `*` leaves an argument unmapped (shared by every instance).
    pub fn in_axes<K, S: IntoAxisFilter<K>>(self, axes: S) -> Map<I, F, S::Binding, Y> {
        Map {
            jit: self.jit,
            in_axes: axes.into_axis_filter(),
            out_axes: self.out_axes,
            axis_size: self.axis_size,
        }
    }

    /// The axis along which each result leaf is stacked.
    pub fn out_axes<K, S: IntoAxisFilter<K>>(mut self, axes: S) -> Map<I, F, X, S::Binding> {
        self.jit.clear_traces();
        Map {
            jit: self.jit,
            in_axes: self.in_axes,
            out_axes: axes.into_axis_filter(),
            axis_size: self.axis_size,
        }
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

impl<I, F, X: AxisType, Y: AxisType> Map<I, F, X, Y>
where
    I: Graph,
    I::Owned: GraphHash + GraphEq<Builder<ObjectTracer> = TracerBuilder<I>>,
    F: Function<I>,
    F::Output: Graph<Owned = F::Output>,
{
    fn apply<A, E>(
        &self,
        input: A,
        trace: impl FnOnce(
            TracerBuilder<I>,
            Vec<(Kind, VarInfo)>,
        ) -> Result<Arc<TraceEntry<F::Output>>, ApplyError<E>>,
    ) -> Result<F::Output, ApplyError<E>>
    where
        A: Graph<Builder<ObjectTracer> = TracerBuilder<I>>,
    {
        let leaves = Leaves::<I>::of(&input);
        let axes = crate::spec::axes::<X>(&self.in_axes, &input, &leaves.paths, "in_axes")?;
        let mut kinds = Vec::new();
        let mut captured = Vec::new();
        let mut mapped = Vec::new();
        let mut in_axes = Vec::new();
        for ((object, axis), path) in leaves.objects.iter().zip(&axes).zip(&leaves.paths) {
            match axis.value {
                None => {
                    kinds.push(crate::cache::kind(object));
                    captured.push(true);
                }
                Some(axis) => {
                    let trace = value(object, "a mapped argument")
                        .map_err(|e| crate::spec::at_path("in_axes", path, e))?;
                    let (_, instance) = util::without_axis(trace.info(), axis)
                        .map_err(|e| crate::spec::at_path("in_axes", path, e))?;
                    kinds.push((Kind::Value, instance));
                    captured.push(false);
                    mapped.push(trace);
                    in_axes.push(axis);
                }
            }
        }
        let builder = leaves.builder.clone();
        let entry = trace(builder, kinds)?;
        let out_axes = entry
            .output_axes
            .as_deref()
            .expect("map trace contains output axes");
        let body = body(&entry, &leaves.objects, &captured)?;
        let results = super::region::map(body, &mapped, &in_axes, out_axes, self.axis_size)
            .map_err(TransformError::from)?;
        Ok(crate::cache::rebuild_graph(
            &entry.output_builder,
            results.into_iter().map(ObjectTracer::from),
        ))
    }
}

impl<I: Graph, F: Function<I>, X: AxisType, Y: AxisType> Function<I> for Map<I, F, X, Y> {
    type Output = F::Output;
}

impl<I, F, X: AxisType, Y: AxisType> Callable<I> for Map<I, F, X, Y>
where
    I: Graph,
    I::Owned: GraphHash + GraphEq<Builder<ObjectTracer> = TracerBuilder<I>>,
    F: Jitable<I>,
{
    fn invoke(&self, input: I) -> Self::Output {
        self.apply(input, |builder, kinds| {
            self.jit.mapped::<Y>(builder, kinds, &self.out_axes)
        })
        .unwrap_or_else(|error| panic!("map: {error}"))
    }
}

impl<I, F, X: AxisType, Y: AxisType> Jitable<I> for Map<I, F, X, Y>
where
    I: Graph,
    I::Owned: GraphHash + GraphEq<Builder<ObjectTracer> = TracerBuilder<I>>,
    F: Jitable<I>,
{
    fn invoke_owned(&self, input: I::Owned) -> Self::Output {
        self.apply(input, |builder, kinds| {
            self.jit.mapped::<Y>(builder, kinds, &self.out_axes)
        })
        .unwrap_or_else(|error| panic!("map: {error}"))
    }
}

impl<I, F, X: AxisType, Y: AxisType> TryCallable<I> for Map<I, F, X, Y>
where
    I: Graph,
    I::Owned: GraphHash + GraphEq<Builder<ObjectTracer> = TracerBuilder<I>>,
    F: TryJitable<I>,
{
    type Error = ApplyError<F::Error>;

    fn try_invoke(&self, input: I) -> Result<Self::Output, Self::Error> {
        self.apply(input, |builder, kinds| {
            self.jit.try_mapped::<Y>(builder, kinds, &self.out_axes)
        })
    }
}

impl<I, F, X: AxisType, Y: AxisType> TryJitable<I> for Map<I, F, X, Y>
where
    I: Graph,
    I::Owned: GraphHash + GraphEq<Builder<ObjectTracer> = TracerBuilder<I>>,
    F: TryJitable<I>,
{
    fn try_invoke_owned(&self, input: I::Owned) -> Result<Self::Output, Self::Error> {
        self.apply(input, |builder, kinds| {
            self.jit.try_mapped::<Y>(builder, kinds, &self.out_axes)
        })
    }
}

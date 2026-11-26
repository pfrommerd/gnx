//! Jit-based autodiff builders: [`jvp`], [`vjp`], [`grad`],
//! [`value_and_grad`], and [`hessian`].

use std::borrow::Cow;
use std::convert::Infallible;
use std::sync::Arc;

use gnx_common::Array;
use gnx_common::ops::{eye_op, reshape_op};
use gnx_expr::{Kind, ObjectTracer, Trace, VarInfo};
use gnx_graph::{All, Filter};
use gnx_graph::{Graph, GraphEq, GraphHash, Leaf, OwnedGraph};
use std::marker::PhantomData;

use super::region;
use crate::cache::{TraceEntry, TracerBuilder, rebuild_graph};
use crate::error::ApplyError;
use crate::filters::Arguments;

struct Wrt<L, F> {
    filter: F,
    positional: Option<Arguments>,
    leaf: PhantomData<fn() -> L>,
}
impl<L, F> Wrt<L, F> {
    fn new(filter: F) -> Self {
        Self {
            filter,
            positional: None,
            leaf: PhantomData,
        }
    }
}
use crate::traced::{
    Leaves, body, constant, differentiated, objects, ones, trace_region, value, zeros,
};
use crate::{
    Callable, Function, Jit, Jitable, TransformError, TransformResult, TryCallable, TryJitable,
    jit, util,
};

#[derive(Clone, Copy)]
enum Inputs {
    Values,
    ReadOnlyCells,
}

/// A traced function with its differentiated argument leaves split out.
struct Prepared<I: Graph, O: Graph> {
    leaves: Leaves<I>,
    slots: Vec<Option<usize>>,
    entry: Arc<TraceEntry<O>>,
    body: gnx_expr::ApplyRegion,
    primals: Vec<Trace>,
    outputs: Vec<VarInfo>,
}

impl<I: Graph, O: Graph<Owned = O>> Prepared<I, O> {
    fn new<L: Leaf, W: Filter<L>, A, E>(
        inputs: Inputs,
        wrt: &Wrt<L, W>,
        input: &A,
        trace: impl FnOnce(TracerBuilder<I>, Vec<(Kind, VarInfo)>) -> Result<Arc<TraceEntry<O>>, E>,
    ) -> Result<Self, ApplyError<E>>
    where
        A: Graph<Builder<ObjectTracer> = TracerBuilder<I>>,
    {
        let leaves = Leaves::<I>::of(input);
        if let Some(args) = &wrt.positional {
            args.validate(input)?;
        }
        let selected = crate::spec::wrt(&wrt.filter, input, &leaves.paths);
        let cell_grads = matches!(inputs, Inputs::ReadOnlyCells);
        let differentiate = differentiated(&leaves.objects, &leaves.paths, &selected, cell_grads)?;
        let entry = trace(leaves.builder.clone(), leaves.kinds()).map_err(ApplyError::Body)?;
        let outputs = crate::cf::value_results(entry.expr.view()).map_err(TransformError::from)?;
        if cell_grads && entry.expr.view().writes_parameters() {
            return Err(
                TransformError::invalid("grad cannot write an input or captured ArrayRef").into(),
            );
        }
        let has_cell_inputs = leaves
            .objects
            .iter()
            .any(|o| matches!(o, ObjectTracer::Cell(_)));
        let super::cell_grad::Operands {
            body,
            slots,
            primals,
        } = if cell_grads && has_cell_inputs {
            super::cell_grad::prepare(&entry, &leaves.objects, &differentiate)?
        } else {
            let captured: Vec<_> = differentiate.iter().map(|flag| !flag).collect();
            let body = body(&entry, &leaves.objects, &captured)?;
            let mut primals = Vec::new();
            let mut slots = Vec::new();
            for (object, selected) in leaves.objects.iter().zip(differentiate) {
                slots.push(selected.then_some(primals.len()));
                if selected {
                    primals.push(value(object, "a differentiated argument")?);
                }
            }
            super::cell_grad::Operands {
                body,
                slots,
                primals,
            }
        };
        Ok(Self {
            leaves,
            slots,
            entry,
            body,
            primals,
            outputs,
        })
    }

    fn wrt(&self) -> Vec<usize> {
        (0..self.primals.len()).collect()
    }

    /// Floating-point results, which get tangents and cotangents.
    fn of(&self) -> Vec<usize> {
        (0..self.outputs.len())
            .filter(|&i| util::is_inexact(&self.outputs[i]))
            .collect()
    }

    fn rebuild_output(&self, traces: Vec<Trace>) -> O {
        rebuild_graph(
            &self.entry.output_builder,
            traces.into_iter().map(ObjectTracer::from),
        )
    }

    /// Forward mode with one tangent leaf per argument leaf.
    fn jvp(&self, tangents: &[ObjectTracer]) -> TransformResult<(Vec<Trace>, Vec<Trace>)> {
        if tangents.len() != self.leaves.objects.len() {
            return Err(TransformError::invalid(
                "tangents must have the same structure as the primals",
            ));
        }
        let tangents = tangents
            .iter()
            .zip(&self.slots)
            .filter(|(_, slot)| slot.is_some())
            .map(|(object, _)| value(object, "a tangent"))
            .collect::<TransformResult<Vec<_>>>()?;
        let of = self.of();
        let (outputs, of_tangents) = region::jvp(
            self.body.clone(),
            &self.primals,
            &tangents,
            &self.wrt(),
            &of,
        )?;
        let mut of_tangents = of_tangents.into_iter();
        let tangents = self
            .outputs
            .iter()
            .enumerate()
            .map(|(i, info)| {
                if of.contains(&i) {
                    of_tangents
                        .next()
                        .expect("tangent per differentiated result")
                } else {
                    zeros(info)
                }
            })
            .collect();
        Ok((outputs, tangents))
    }

    /// Reverse mode given one cotangent per result (`None` for results that
    /// aren't differentiated). Returns the results and one cotangent leaf per
    /// argument leaf.
    fn vjp(
        &self,
        cotangents: Vec<Option<Trace>>,
    ) -> TransformResult<(Vec<Trace>, Vec<ObjectTracer>)> {
        let of = self.of();
        let cotangents: Vec<_> = of
            .iter()
            .map(|&i| {
                cotangents[i]
                    .clone()
                    .ok_or_else(|| TransformError::invalid("missing cotangent"))
            })
            .collect::<TransformResult<_>>()?;
        let (outputs, input_cts) = region::vjp(
            self.body.clone(),
            &self.primals,
            &cotangents,
            &self.wrt(),
            &of,
        )?;
        let leaves = super::cell_grad::cotangents(&self.leaves.objects, &self.slots, &input_cts);
        Ok((outputs, leaves))
    }

    /// `vjp` with a cotangent of one for the single scalar result.
    fn grad(&self) -> TransformResult<(Vec<Trace>, Vec<ObjectTracer>)> {
        match self.outputs.as_slice() {
            [info] if util::is_float(info) && util::dims(info)?.is_empty() => {
                self.vjp(vec![Some(ones(info))])
            }
            _ => Err(TransformError::invalid(
                "grad needs a function returning one floating-point scalar",
            )),
        }
    }
}

macro_rules! builder_methods {
    ($ty:ident) => {
        impl<I: Graph, F: Function<I>, L: Leaf, W: Filter<L>> $ty<I, F, L, W> {
            /// Differentiate the tracer leaves matched by a graph filter.
            pub fn wrt<T: Leaf, S: Filter<T>>(self, filter: S) -> $ty<I, F, T, S::Owned> {
                $ty {
                    jit: self.jit,
                    wrt: Wrt::new(filter.owned()),
                }
            }

            /// Differentiate one complete top-level positional argument (zero-based).
            pub fn wrt_arg(self, index: usize) -> $ty<I, F, (), Arguments> {
                self.wrt_args([index])
            }

            /// Differentiate the specified top-level arguments; duplicates are ignored.
            pub fn wrt_args(self, indices: impl AsRef<[usize]>) -> $ty<I, F, (), Arguments> {
                let args = Arguments::new(indices.as_ref().to_vec());
                let mut result = self.wrt(&args);
                result.wrt.positional = Some(args);
                result
            }

            /// Label the traced body for display.
            pub fn name(mut self, name: impl Into<Cow<'static, str>>) -> Self {
                self.jit.set_name_hint(name.into());
                self
            }
        }
    };
}

/// Implement the four call traits for a wrapper from one `apply` method.
macro_rules! call_traits {
    ($ty:ident, $input:ty, $owned:ty, $output:ty, $name:literal) => {
        impl<I: Graph, F: Function<I>, L: Leaf, W: Filter<L>> Function<$input> for $ty<I, F, L, W>
        where
            $input: Graph,
            $output: Graph,
        {
            type Output = $output;
        }

        impl<I, F, L: Leaf, W: Filter<L>> Callable<$input> for $ty<I, F, L, W>
        where
            I: Graph,
            I::Owned: GraphHash + GraphEq<Builder<ObjectTracer> = TracerBuilder<I>>,
            F: Jitable<I>,
            $input: Graph,
            $output: Graph,
        {
            fn invoke(&self, input: $input) -> Self::Output {
                self.apply(input, |builder, kinds| {
                    Ok::<_, Infallible>(self.jit.traced(builder, kinds))
                })
                .unwrap_or_else(|error: ApplyError<Infallible>| {
                    panic!(concat!($name, ": {}"), error)
                })
            }
        }

        impl<I, F, L: Leaf, W: Filter<L>> Jitable<$input> for $ty<I, F, L, W>
        where
            I: Graph,
            I::Owned: GraphHash + GraphEq<Builder<ObjectTracer> = TracerBuilder<I>>,
            F: Jitable<I>,
            $output: OwnedGraph,
        {
            fn invoke_owned(&self, input: <$input as Graph>::Owned) -> Self::Output {
                self.apply(input, |builder, kinds| {
                    Ok::<_, Infallible>(self.jit.traced(builder, kinds))
                })
                .unwrap_or_else(|error: ApplyError<Infallible>| {
                    panic!(concat!($name, ": {}"), error)
                })
            }
        }

        impl<I, F, L: Leaf, W: Filter<L>> TryCallable<$input> for $ty<I, F, L, W>
        where
            I: Graph,
            I::Owned: GraphHash + GraphEq<Builder<ObjectTracer> = TracerBuilder<I>>,
            F: TryJitable<I>,
            $input: Graph,
            $output: Graph,
        {
            type Error = ApplyError<F::Error>;

            fn try_invoke(&self, input: $input) -> Result<Self::Output, Self::Error> {
                self.apply(input, |builder, kinds| self.jit.try_traced(builder, kinds))
            }
        }

        impl<I, F, L: Leaf, W: Filter<L>> TryJitable<$input> for $ty<I, F, L, W>
        where
            I: Graph,
            I::Owned: GraphHash + GraphEq<Builder<ObjectTracer> = TracerBuilder<I>>,
            F: TryJitable<I>,
            $output: OwnedGraph,
        {
            fn try_invoke_owned(
                &self,
                input: <$input as Graph>::Owned,
            ) -> Result<Self::Output, Self::Error> {
                self.apply(input, |builder, kinds| self.jit.try_traced(builder, kinds))
            }
        }
    };
}

// ----- jvp --------------------------------------------------------------------

/// Forward mode; see [`jvp`].
pub struct Jvp<I: Graph, F: Function<I>, L: Leaf = (), W: Filter<L> = All> {
    jit: Jit<I, F>,
    wrt: Wrt<L, W>,
}

/// Forward-mode differentiation (JAX's `jvp`). Call with `(primals, tangents)`,
/// where the tangents have the primals' structure, to get `(results,
/// result_tangents)`. Non-floating-point results get zero tangents.
pub fn jvp<I: Graph, F: Function<I>>(func: F) -> Jvp<I, F> {
    Jvp {
        jit: jit(func),
        wrt: Wrt::new(All),
    }
}
builder_methods!(Jvp);

impl<I, F, L: Leaf, W: Filter<L>> Jvp<I, F, L, W>
where
    I: Graph,
    F: Function<I>,
    F::Output: Graph<Owned = F::Output>,
{
    fn apply<A, T, E>(
        &self,
        (primals, tangents): (A, T),
        trace: impl FnOnce(
            TracerBuilder<I>,
            Vec<(Kind, VarInfo)>,
        ) -> Result<Arc<TraceEntry<F::Output>>, E>,
    ) -> Result<(F::Output, F::Output), ApplyError<E>>
    where
        A: Graph<Builder<ObjectTracer> = TracerBuilder<I>>,
        T: Graph,
    {
        let prepared = Prepared::<I, F::Output>::new(Inputs::Values, &self.wrt, &primals, trace)?;
        let (outputs, tangents) = prepared.jvp(&objects(&tangents))?;
        Ok((
            prepared.rebuild_output(outputs),
            prepared.rebuild_output(tangents),
        ))
    }
}
call_traits!(
    Jvp,
    (I, I),
    (<I as Graph>::Owned, <I as Graph>::Owned),
    (F::Output, F::Output),
    "jvp"
);

// ----- vjp --------------------------------------------------------------------

/// Reverse mode; see [`vjp`].
pub struct Vjp<I: Graph, F: Function<I>, L: Leaf = (), W: Filter<L> = All> {
    jit: Jit<I, F>,
    wrt: Wrt<L, W>,
}

/// Reverse-mode differentiation (JAX's `vjp`). Call with `(primals,
/// cotangents)`, where the cotangents have the results' structure, to get
/// `(results, argument_cotangents)`; the latter has the arguments' structure,
/// with zeros for leaves that aren't differentiated. See also
/// [`Vjp::pullback`].
pub fn vjp<I: Graph, F: Function<I>>(func: F) -> Vjp<I, F> {
    Vjp {
        jit: jit(func),
        wrt: Wrt::new(All),
    }
}
builder_methods!(Vjp);

impl<I, F, L: Leaf, W: Filter<L>> Vjp<I, F, L, W>
where
    I: Graph,
    F: Function<I>,
    F::Output: Graph<Owned = F::Output>,
{
    fn apply<A, C, E>(
        &self,
        (primals, cotangents): (A, C),
        trace: impl FnOnce(
            TracerBuilder<I>,
            Vec<(Kind, VarInfo)>,
        ) -> Result<Arc<TraceEntry<F::Output>>, E>,
    ) -> Result<(F::Output, I::Owned), ApplyError<E>>
    where
        A: Graph<Builder<ObjectTracer> = TracerBuilder<I>>,
        C: Graph,
    {
        let prepared = Prepared::<I, F::Output>::new(Inputs::Values, &self.wrt, &primals, trace)?;
        let cotangents = objects(&cotangents);
        if cotangents.len() != prepared.outputs.len() {
            return Err(TransformError::invalid(
                "cotangents must have the same structure as the results",
            )
            .into());
        }
        let cotangents = cotangents
            .iter()
            .map(|object| value(object, "a cotangent").map(Some))
            .collect::<TransformResult<_>>()?;
        let (outputs, input_cts) = prepared.vjp(cotangents)?;
        Ok((
            prepared.rebuild_output(outputs),
            prepared.leaves.rebuild(input_cts),
        ))
    }
}
call_traits!(
    Vjp,
    (I, F::Output),
    (<I as Graph>::Owned, F::Output),
    (F::Output, <I as Graph>::Owned),
    "vjp"
);

impl<I, F, L: Leaf, W: Filter<L>> Vjp<I, F, L, W>
where
    I: Graph,
    I::Owned: GraphHash + GraphEq<Builder<ObjectTracer> = TracerBuilder<I>>,
    F: Jitable<I>,
{
    /// The results at `primals` and a pullback mapping result cotangents to
    /// argument cotangents. Each pullback call records one fused `ad.vjp`;
    /// the duplicated primal work is removed after expansion.
    pub fn pullback(&self, primals: I) -> (F::Output, impl Fn(F::Output) -> I::Owned + '_)
    where
        I: Clone,
    {
        let outputs = self.jit.invoke(primals.clone());
        (outputs, move |cotangents| {
            self.invoke((primals.clone(), cotangents)).1
        })
    }
}

// ----- grad / value_and_grad --------------------------------------------------

/// The gradient of a scalar function; see [`grad`].
pub struct Grad<I: Graph, F: Function<I>, L: Leaf = (), W: Filter<L> = All> {
    jit: Jit<I, F>,
    wrt: Wrt<L, W>,
}

/// The gradient of a function returning one floating-point scalar (JAX's
/// `grad`). The result has the arguments' structure, with zeros for leaves
/// that aren't differentiated.
///
/// For an `ArrayRef`, the independent variable is the cell's contents at entry.
/// The loss may read argument and captured cells, but may not write them.
/// Tied cells accumulate one derivative across all uses and return one fresh
/// gradient cell shared by the corresponding result paths. Ordinary array
/// arguments remain independent by position. When an enclosing trace leaves
/// cell aliases ambiguous, call `grad` outside it or pass array values instead.
pub fn grad<I: Graph, F: Function<I>>(func: F) -> Grad<I, F> {
    Grad {
        jit: jit(func),
        wrt: Wrt::new(All),
    }
}
builder_methods!(Grad);

impl<I, F, L: Leaf, W: Filter<L>> Grad<I, F, L, W>
where
    I: Graph,
    F: Function<I>,
    F::Output: Graph<Owned = F::Output>,
{
    fn apply<A, E>(
        &self,
        primals: A,
        trace: impl FnOnce(
            TracerBuilder<I>,
            Vec<(Kind, VarInfo)>,
        ) -> Result<Arc<TraceEntry<F::Output>>, E>,
    ) -> Result<I::Owned, ApplyError<E>>
    where
        A: Graph<Builder<ObjectTracer> = TracerBuilder<I>>,
    {
        let prepared =
            Prepared::<I, F::Output>::new(Inputs::ReadOnlyCells, &self.wrt, &primals, trace)?;
        let (_, input_cts) = prepared.grad()?;
        Ok(prepared.leaves.rebuild(input_cts))
    }
}
call_traits!(Grad, I, <I as Graph>::Owned, <I as Graph>::Owned, "grad");

/// The value and gradient of a scalar function; see [`value_and_grad`].
pub struct ValueAndGrad<I: Graph, F: Function<I>, L: Leaf = (), W: Filter<L> = All> {
    jit: Jit<I, F>,
    wrt: Wrt<L, W>,
}

/// Like [`grad`], also returning the function's result.
pub fn value_and_grad<I: Graph, F: Function<I>>(func: F) -> ValueAndGrad<I, F> {
    ValueAndGrad {
        jit: jit(func),
        wrt: Wrt::new(All),
    }
}
builder_methods!(ValueAndGrad);

impl<I, F, L: Leaf, W: Filter<L>> ValueAndGrad<I, F, L, W>
where
    I: Graph,
    F: Function<I>,
    F::Output: Graph<Owned = F::Output>,
{
    fn apply<A, E>(
        &self,
        primals: A,
        trace: impl FnOnce(
            TracerBuilder<I>,
            Vec<(Kind, VarInfo)>,
        ) -> Result<Arc<TraceEntry<F::Output>>, E>,
    ) -> Result<(F::Output, I::Owned), ApplyError<E>>
    where
        A: Graph<Builder<ObjectTracer> = TracerBuilder<I>>,
    {
        let prepared =
            Prepared::<I, F::Output>::new(Inputs::ReadOnlyCells, &self.wrt, &primals, trace)?;
        let (outputs, input_cts) = prepared.grad()?;
        Ok((
            prepared.rebuild_output(outputs),
            prepared.leaves.rebuild(input_cts),
        ))
    }
}
call_traits!(
    ValueAndGrad,
    I,
    <I as Graph>::Owned,
    (F::Output, <I as Graph>::Owned),
    "value_and_grad"
);

// ----- hessian ----------------------------------------------------------------

/// The Hessian of a scalar function; see [`hessian`].
pub struct Hessian<I: Graph, F: Function<I>, L: Leaf = (), W: Filter<L> = All> {
    jit: Jit<I, F>,
    wrt: Wrt<L, W>,
}

/// The Hessian of a function returning one floating-point scalar, with respect
/// to exactly one array argument leaf of shape `S` (select it with `wrt` if
/// there are several). The result has shape `S ++ S`.
///
/// Computed as forward-over-reverse: the gradient is differentiated with a
/// `jvp` mapped over the standard basis.
pub fn hessian<I: Graph, F: Function<I>>(func: F) -> Hessian<I, F> {
    Hessian {
        jit: jit(func),
        wrt: Wrt::new(All),
    }
}
builder_methods!(Hessian);

impl<I, F, L: Leaf, W: Filter<L>> Hessian<I, F, L, W>
where
    I: Graph,
    F: Function<I>,
    F::Output: Graph<Owned = F::Output>,
{
    fn apply<A, E>(
        &self,
        primals: A,
        trace: impl FnOnce(
            TracerBuilder<I>,
            Vec<(Kind, VarInfo)>,
        ) -> Result<Arc<TraceEntry<F::Output>>, E>,
    ) -> Result<Array, ApplyError<E>>
    where
        A: Graph<Builder<ObjectTracer> = TracerBuilder<I>>,
    {
        let prepared = Prepared::<I, F::Output>::new(Inputs::Values, &self.wrt, &primals, trace)?;
        let [x] = prepared.primals.as_slice() else {
            return Err(TransformError::invalid(
                "hessian differentiates exactly one array argument; select it with wrt",
            )
            .into());
        };
        let output = match prepared.outputs.as_slice() {
            [info] if util::is_float(info) && util::dims(info)?.is_empty() => info.clone(),
            _ => {
                return Err(TransformError::invalid(
                    "hessian needs a function returning one floating-point scalar",
                )
                .into());
            }
        };
        let info = x.info().clone();
        let shape = util::dims(&info)?;
        let dtype = util::dtype(&info)?;
        let size: usize = shape.iter().product();
        // g(x) = vjp(f)(x, 1)
        let body = prepared.body.clone();
        let gradient = trace_region(std::slice::from_ref(&info), |args| {
            Ok(region::vjp(body, &args[..1], &[ones(&output)], &[0], &[0])?.1)
        })?;
        // hvp(v) = jvp(g)(x; v), mapped over the standard basis.
        let hvp = trace_region(std::slice::from_ref(&info), |v| {
            Ok(region::jvp(&gradient, std::slice::from_ref(x), &v[..1], &[0], &[0])?.1)
        })?;
        let eye = constant(eye_op(), util::info(&[size, size], dtype));
        let basis_shape: Vec<_> = std::iter::once(size).chain(shape.iter().copied()).collect();
        let basis = reshape(&eye, util::info(&basis_shape, dtype));
        let rows = crate::map::region::map(&hvp, &[basis], &[0], &[0], None)
            .map_err(TransformError::from)?
            .pop()
            .ok_or_else(|| TransformError::invalid("hessian map arity"))?;
        let square: Vec<_> = shape.iter().chain(&shape).copied().collect();
        let hessian = reshape(&rows, util::info(&square, dtype));
        Ok(Array::try_from_value(hessian)
            .map_err(|_| TransformError::invalid("hessian result is not an array"))?)
    }
}

fn reshape(value: &Trace, info: VarInfo) -> Trace {
    gnx_expr::Invocation::invoke(reshape_op(), vec![], vec![value.clone()], vec![info])
        .pop()
        .expect("one result")
        .into_trace_ref()
}

call_traits!(Hessian, I, <I as Graph>::Owned, Array, "hessian");

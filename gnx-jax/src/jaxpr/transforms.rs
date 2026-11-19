//! Native JAX lowerings for the `ad` dialect: `jax.vmap`, `jax.jvp`, `jax.vjp`.
//! JAX keeps these transformations rather than expanding them in GNX.

use super::control_flow::{RegionAbi, tuple, unpack};
use super::*;
use crate::register_jax_lowering;
use gnx_transforms::ad::{DiffParameters, JvpOperation, VjpOperation};
use gnx_transforms::map::MapOperation;

impl JaxLowering for MapOperation {
    fn lower_jax<'py>(
        &self,
        ctx: &mut TraceContext<'py>,
        _: &GnxEqn,
        operands: &[TraceValue<'py>],
    ) -> Result<Vec<TraceValue<'py>>> {
        let parameters = self.parameters();
        let py = ctx.py;
        let body = ctx.region_callable(0, RegionAbi::Branch)?;
        let options = PyDict::new(py);
        options.set_item("in_axes", PyTuple::new(py, &parameters.in_axes)?)?;
        options.set_item("out_axes", PyTuple::new(py, &parameters.out_axes)?)?;
        options.set_item("axis_size", parameters.size)?;
        let mapped = py
            .import("jax")?
            .getattr("vmap")?
            .call((body,), Some(&options))?;
        unpack(mapped.call1(tuple(py, operands)?)?)
    }
}

/// `f` restricted to the inputs `wrt` (the others fixed to `primals`) and the
/// results `of`.
fn restrict<'py>(
    py: Python<'py>,
    f: Bound<'py, PyCFunction>,
    primals: &[TraceValue<'py>],
    wrt: Vec<usize>,
    of: Vec<usize>,
) -> PyResult<Bound<'py, PyCFunction>> {
    let f = f.unbind();
    let primals: Vec<Py<PyAny>> = primals
        .iter()
        .map(|value| value.as_native().clone().unbind())
        .collect();
    PyCFunction::new_closure(
        py,
        Some(c"gnx_restricted"),
        None,
        move |args: &Bound<'_, PyTuple>, _: Option<&Bound<'_, PyDict>>| -> PyResult<Py<PyAny>> {
            let py = args.py();
            let mut inputs: Vec<_> = primals.iter().map(|value| value.bind(py).clone()).collect();
            for (k, &index) in wrt.iter().enumerate() {
                inputs[index] = args.get_item(k)?;
            }
            let outputs = f
                .bind(py)
                .call1(PyTuple::new(py, inputs)?)?
                .cast_into::<PyTuple>()?;
            let selected = of
                .iter()
                .map(|&index| outputs.get_item(index))
                .collect::<PyResult<Vec<_>>>()?;
            Ok(PyTuple::new(py, selected)?.into_any().unbind())
        },
    )
}

/// The body applied to all primals, plus the body restricted to `wrt`/`of`
/// and the primals it is differentiated at.
fn split_body<'a, 'py>(
    ctx: &TraceContext<'py>,
    eqn: &GnxEqn,
    operands: &'a [TraceValue<'py>],
    parameters: &DiffParameters,
) -> Result<(
    Vec<TraceValue<'py>>,
    Bound<'py, PyCFunction>,
    Bound<'py, PyTuple>,
    &'a [TraceValue<'py>],
)> {
    let inputs = eqn
        .subexprs()
        .first()
        .ok_or_else(|| invalid("missing transformation body"))?
        .expr()
        .inputs()
        .len();
    let (primals, extra) = operands.split_at(inputs);
    let body = ctx.region_callable(0, RegionAbi::Branch)?;
    let results = unpack(body.call1(tuple(ctx.py, primals)?)?)?;
    let wrt_primals: Vec<_> = parameters.wrt.iter().map(|&i| primals[i].clone()).collect();
    let restricted = restrict(
        ctx.py,
        body,
        primals,
        parameters.wrt.clone(),
        parameters.of.clone(),
    )?;
    Ok((results, restricted, tuple(ctx.py, &wrt_primals)?, extra))
}

impl JaxLowering for JvpOperation {
    fn lower_jax<'py>(
        &self,
        ctx: &mut TraceContext<'py>,
        eqn: &GnxEqn,
        operands: &[TraceValue<'py>],
    ) -> Result<Vec<TraceValue<'py>>> {
        let (mut results, restricted, primals, tangents) =
            split_body(ctx, eqn, operands, self.parameters())?;
        let output = ctx
            .py
            .import("jax")?
            .call_method1("jvp", (restricted, primals, tuple(ctx.py, tangents)?))?;
        results.extend(unpack(output.get_item(1)?)?);
        Ok(results)
    }
}

impl JaxLowering for VjpOperation {
    fn lower_jax<'py>(
        &self,
        ctx: &mut TraceContext<'py>,
        eqn: &GnxEqn,
        operands: &[TraceValue<'py>],
    ) -> Result<Vec<TraceValue<'py>>> {
        let (mut results, restricted, primals, cotangents) =
            split_body(ctx, eqn, operands, self.parameters())?;
        let mut args = vec![restricted.into_any()];
        args.extend(primals.iter());
        let output = ctx
            .py
            .import("jax")?
            .call_method1("vjp", PyTuple::new(ctx.py, args)?)?;
        let pullback = output.get_item(1)?;
        results.extend(unpack(pullback.call1((tuple(ctx.py, cotangents)?,))?)?);
        Ok(results)
    }
}

register_jax_lowering!(MapOperation, JvpOperation, VjpOperation);

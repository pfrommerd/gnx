use super::*;
use crate::register_jax_lowering;
use gnx_transforms::cf::{CondOperation, ScanOperation, SwitchOperation, WhileOperation};

#[derive(Clone, Copy)]
pub(super) enum RegionAbi {
    Branch,
    Predicate,
    Carry,
    Scan(usize),
}

pub(super) fn tuple<'py>(
    py: Python<'py>,
    values: &[TraceValue<'py>],
) -> PyResult<Bound<'py, PyTuple>> {
    PyTuple::new(py, values.iter().map(TraceValue::as_native))
}

pub(super) fn unpack(value: Bound<'_, PyAny>) -> Result<Vec<TraceValue<'_>>> {
    Ok(value
        .cast_into::<PyTuple>()
        .map_err(PyErr::from)?
        .iter()
        .map(TraceValue)
        .collect())
}

impl<'py> TraceContext<'py> {
    pub(super) fn region_callable(
        &self,
        index: usize,
        abi: RegionAbi,
    ) -> Result<Bound<'py, PyCFunction>> {
        let child = self
            .children
            .get(index)
            .ok_or_else(|| invalid("missing control-flow region"))?;
        let expr = child.expr.clone();
        let captures: Vec<_> = child
            .captures
            .iter()
            .map(|value| value.as_native().clone().unbind())
            .collect();
        Ok(PyCFunction::new_closure(
            self.py,
            Some(c"gnx_region"),
            None,
            move |args: &Bound<'_, PyTuple>,
                  _kwargs: Option<&Bound<'_, PyDict>>|
                  -> PyResult<Py<PyAny>> {
                let py = args.py();
                let mut operands: Vec<_> = captures
                    .iter()
                    .map(|value| TraceValue(value.bind(py).clone()))
                    .collect();
                match abi {
                    RegionAbi::Branch => operands.extend(args.iter().map(TraceValue)),
                    RegionAbi::Predicate | RegionAbi::Carry => {
                        operands
                            .extend(unpack(args.get_item(0)?).map_err(|err| err.into_pyerr(py))?);
                    }
                    RegionAbi::Scan(_) => {
                        for position in 0..2 {
                            operands.extend(
                                unpack(args.get_item(position)?)
                                    .map_err(|err| err.into_pyerr(py))?,
                            );
                        }
                    }
                }
                let mut outputs =
                    trace_body(py, &expr, operands).map_err(|err| err.into_pyerr(py))?;
                match abi {
                    RegionAbi::Predicate => Ok(outputs
                        .pop()
                        .ok_or_else(|| PyValueError::new_err("missing predicate result"))?
                        .into_native()
                        .unbind()),
                    RegionAbi::Branch | RegionAbi::Carry => {
                        Ok(tuple(py, &outputs)?.into_any().unbind())
                    }
                    RegionAbi::Scan(carry_len) => {
                        let ys = outputs.split_off(carry_len);
                        Ok(PyTuple::new(py, [tuple(py, &outputs)?, tuple(py, &ys)?])?
                            .into_any()
                            .unbind())
                    }
                }
            },
        )?)
    }
}

impl JaxLowering for CondOperation {
    fn lower_jax<'py>(
        &self,
        ctx: &mut TraceContext<'py>,
        _: &GnxEqn,
        operands: &[TraceValue<'py>],
    ) -> Result<Vec<TraceValue<'py>>> {
        let [selector, operands @ ..] = operands else {
            return Err(invalid("missing cond predicate"));
        };
        let t = ctx.region_callable(0, RegionAbi::Branch)?.into_any();
        let f = ctx.region_callable(1, RegionAbi::Branch)?.into_any();
        let mut args = vec![selector.as_native().clone(), t, f];
        args.extend(operands.iter().map(|value| value.as_native().clone()));
        unpack(
            ctx.lax()?
                .0
                .call_method1("cond", PyTuple::new(ctx.py, args)?)?,
        )
    }
}
register_jax_lowering!(CondOperation);

impl JaxLowering for SwitchOperation {
    fn lower_jax<'py>(
        &self,
        ctx: &mut TraceContext<'py>,
        eqn: &GnxEqn,
        operands: &[TraceValue<'py>],
    ) -> Result<Vec<TraceValue<'py>>> {
        let [selector, operands @ ..] = operands else {
            return Err(invalid("missing switch index"));
        };
        let branches = (0..eqn.subexprs().len())
            .map(|index| ctx.region_callable(index, RegionAbi::Branch))
            .collect::<Result<Vec<_>>>()?;
        let mut args = vec![
            selector.as_native().clone(),
            PyTuple::new(ctx.py, branches)?.into_any(),
        ];
        args.extend(operands.iter().map(|value| value.as_native().clone()));
        unpack(
            ctx.lax()?
                .0
                .call_method1("switch", PyTuple::new(ctx.py, args)?)?,
        )
    }
}
register_jax_lowering!(SwitchOperation);

impl JaxLowering for WhileOperation {
    fn lower_jax<'py>(
        &self,
        ctx: &mut TraceContext<'py>,
        _: &GnxEqn,
        operands: &[TraceValue<'py>],
    ) -> Result<Vec<TraceValue<'py>>> {
        let predicate = ctx.region_callable(0, RegionAbi::Predicate)?;
        let body = ctx.region_callable(1, RegionAbi::Carry)?;
        unpack(
            ctx.lax()?
                .0
                .call_method1("while_loop", (predicate, body, tuple(ctx.py, operands)?))?,
        )
    }
}
register_jax_lowering!(WhileOperation);

impl JaxLowering for ScanOperation {
    fn lower_jax<'py>(
        &self,
        ctx: &mut TraceContext<'py>,
        _: &GnxEqn,
        operands: &[TraceValue<'py>],
    ) -> Result<Vec<TraceValue<'py>>> {
        let parameters = self.parameters();
        let length = parameters.length;
        let carry_len = parameters.physical_carry_len()?;
        let (carry, xs) = operands.split_at(carry_len);
        let body = ctx.region_callable(0, RegionAbi::Scan(carry_len))?;
        let outputs = ctx.lax()?.0.call_method1(
            "scan",
            (body, tuple(ctx.py, carry)?, tuple(ctx.py, xs)?, length),
        )?;
        let outputs = outputs.cast_into::<PyTuple>().map_err(PyErr::from)?;
        let mut carry = unpack(outputs.get_item(0)?)?;
        carry.extend(unpack(outputs.get_item(1)?)?);
        Ok(carry)
    }
}
register_jax_lowering!(ScanOperation);

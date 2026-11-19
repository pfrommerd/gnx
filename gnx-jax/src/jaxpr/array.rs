//! JAX lowerings for the `array` dialect operations beyond `add`.

use super::*;
use crate::register_jax_lowering;
use gnx_math::ops::{
    BroadcastInDimOperation, EyeOperation, MulOperation, OnesOperation, ReduceSumOperation,
    ReshapeOperation, RevOperation, SelectOperation, TransposeOperation, ZerosOperation,
};

fn lax_call<'py>(
    ctx: &TraceContext<'py>,
    name: &str,
    args: Vec<Bound<'py, PyAny>>,
) -> Result<Vec<TraceValue<'py>>> {
    let lax = ctx.lax()?;
    Ok(vec![TraceValue(
        lax.0.call_method1(name, PyTuple::new(ctx.py, args)?)?,
    )])
}

fn result_info(eqn: &GnxEqn) -> Result<&ArrayInfo> {
    match eqn.outputs() {
        [(_, info)] => expected(info),
        _ => Err(invalid("array operations have one result")),
    }
}

fn indices<'py>(ctx: &TraceContext<'py>, values: &[usize]) -> Result<Bound<'py, PyAny>> {
    Ok(PyTuple::new(ctx.py, values)?.into_any())
}

fn operand<'a, 'py>(
    operands: &'a [TraceValue<'py>],
    count: usize,
) -> Result<&'a [TraceValue<'py>]> {
    if operands.len() != count {
        return Err(invalid(format!("expected {count} operands")));
    }
    Ok(operands)
}

fn native<'py>(values: &[TraceValue<'py>]) -> Vec<Bound<'py, PyAny>> {
    values
        .iter()
        .map(|value| value.as_native().clone())
        .collect()
}

impl JaxLowering for MulOperation {
    fn lower_jax<'py>(
        &self,
        ctx: &mut TraceContext<'py>,
        eqn: &GnxEqn,
        operands: &[TraceValue<'py>],
    ) -> Result<Vec<TraceValue<'py>>> {
        let name = if result_info(eqn)?.dtype() == DType::Bool {
            "bitwise_and"
        } else {
            "mul"
        };
        lax_call(ctx, name, native(operand(operands, 2)?))
    }
}

impl JaxLowering for SelectOperation {
    fn lower_jax<'py>(
        &self,
        ctx: &mut TraceContext<'py>,
        _: &GnxEqn,
        operands: &[TraceValue<'py>],
    ) -> Result<Vec<TraceValue<'py>>> {
        lax_call(ctx, "select", native(operand(operands, 3)?))
    }
}

impl JaxLowering for ZerosOperation {
    fn lower_jax<'py>(
        &self,
        ctx: &mut TraceContext<'py>,
        eqn: &GnxEqn,
        operands: &[TraceValue<'py>],
    ) -> Result<Vec<TraceValue<'py>>> {
        operand(operands, 0)?;
        let info = result_info(eqn)?;
        let dtype: &'static str = info.dtype().into();
        let zeros = ctx
            .py
            .import("jax.numpy")?
            .call_method1("zeros", (PyTuple::new(ctx.py, dimensions(info)?)?, dtype))?;
        Ok(vec![TraceValue(zeros)])
    }
}

impl JaxLowering for OnesOperation {
    fn lower_jax<'py>(
        &self,
        ctx: &mut TraceContext<'py>,
        eqn: &GnxEqn,
        operands: &[TraceValue<'py>],
    ) -> Result<Vec<TraceValue<'py>>> {
        operand(operands, 0)?;
        let info = result_info(eqn)?;
        let dtype: &'static str = info.dtype().into();
        let ones = ctx
            .py
            .import("jax.numpy")?
            .call_method1("ones", (PyTuple::new(ctx.py, dimensions(info)?)?, dtype))?;
        Ok(vec![TraceValue(ones)])
    }
}

impl JaxLowering for EyeOperation {
    fn lower_jax<'py>(
        &self,
        ctx: &mut TraceContext<'py>,
        eqn: &GnxEqn,
        operands: &[TraceValue<'py>],
    ) -> Result<Vec<TraceValue<'py>>> {
        operand(operands, 0)?;
        let info = result_info(eqn)?;
        let [rows, columns] = dimensions(info)?[..] else {
            return Err(invalid("eye produces a matrix"));
        };
        let dtype: &'static str = info.dtype().into();
        let eye = ctx
            .py
            .import("jax.numpy")?
            .call_method1("eye", (rows, columns, 0, dtype))?;
        Ok(vec![TraceValue(eye)])
    }
}

impl JaxLowering for ReshapeOperation {
    fn lower_jax<'py>(
        &self,
        ctx: &mut TraceContext<'py>,
        eqn: &GnxEqn,
        operands: &[TraceValue<'py>],
    ) -> Result<Vec<TraceValue<'py>>> {
        let [input] = operand(operands, 1)? else {
            unreachable!()
        };
        let shape = PyTuple::new(ctx.py, dimensions(result_info(eqn)?)?)?.into_any();
        lax_call(ctx, "reshape", vec![input.as_native().clone(), shape])
    }
}

impl JaxLowering for BroadcastInDimOperation {
    fn lower_jax<'py>(
        &self,
        ctx: &mut TraceContext<'py>,
        eqn: &GnxEqn,
        operands: &[TraceValue<'py>],
    ) -> Result<Vec<TraceValue<'py>>> {
        let [input] = operand(operands, 1)? else {
            unreachable!()
        };
        let shape = PyTuple::new(ctx.py, dimensions(result_info(eqn)?)?)?.into_any();
        let dims = indices(ctx, self.dims())?;
        lax_call(
            ctx,
            "broadcast_in_dim",
            vec![input.as_native().clone(), shape, dims],
        )
    }
}

impl JaxLowering for ReduceSumOperation {
    fn lower_jax<'py>(
        &self,
        ctx: &mut TraceContext<'py>,
        eqn: &GnxEqn,
        operands: &[TraceValue<'py>],
    ) -> Result<Vec<TraceValue<'py>>> {
        let [input] = operand(operands, 1)? else {
            unreachable!()
        };
        let name = if result_info(eqn)?.dtype() == DType::Bool {
            "reduce_or"
        } else {
            "reduce_sum"
        };
        let axes = indices(ctx, self.axes())?;
        lax_call(ctx, name, vec![input.as_native().clone(), axes])
    }
}

impl JaxLowering for TransposeOperation {
    fn lower_jax<'py>(
        &self,
        ctx: &mut TraceContext<'py>,
        _: &GnxEqn,
        operands: &[TraceValue<'py>],
    ) -> Result<Vec<TraceValue<'py>>> {
        let [input] = operand(operands, 1)? else {
            unreachable!()
        };
        let perm = indices(ctx, self.perm())?;
        lax_call(ctx, "transpose", vec![input.as_native().clone(), perm])
    }
}

impl JaxLowering for RevOperation {
    fn lower_jax<'py>(
        &self,
        ctx: &mut TraceContext<'py>,
        _: &GnxEqn,
        operands: &[TraceValue<'py>],
    ) -> Result<Vec<TraceValue<'py>>> {
        let [input] = operand(operands, 1)? else {
            unreachable!()
        };
        let axes = indices(ctx, self.axes())?;
        lax_call(ctx, "rev", vec![input.as_native().clone(), axes])
    }
}

register_jax_lowering!(
    MulOperation,
    SelectOperation,
    ZerosOperation,
    OnesOperation,
    EyeOperation,
    ReshapeOperation,
    BroadcastInDimOperation,
    ReduceSumOperation,
    TransposeOperation,
    RevOperation,
);

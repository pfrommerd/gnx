use super::{invalid, unsupported};
use gnx_common::ops::reduction::ReduceAddOperation;
use gnx_expr::{Eqn, Expr, VarInfo};
use gnx_math::{
    ArrayInfo, DType, Dim,
    ops::{
        BroadcastInDimOperation, OnesOperation, ReshapeOperation, TransposeOperation,
        ZerosOperation,
    },
};
use std::{collections::HashMap, io};
use tch::{CModule, Kind, Tensor};

thread_local! { static TRACE_DEVICE: std::cell::Cell<tch::Device> = const { std::cell::Cell::new(tch::Device::Cpu) }; }

/// Device selected for tensors created by a lowering, including nullary ops.
/// Like LibTorch's tracer, the scope is synchronous and thread-local.
pub fn tracing_device() -> tch::Device {
    TRACE_DEVICE.get()
}

struct TraceScope(tch::Device);
impl Drop for TraceScope {
    fn drop(&mut self) {
        TRACE_DEVICE.set(self.0);
    }
}

/// Emit traceable, pure tensor operations. A rule may inspect static metadata,
/// but must not read operand data or mutate its inputs.
pub trait TorchLowering {
    fn lower_torch(&self, eqn: &Eqn, inputs: &[Tensor]) -> io::Result<Vec<Tensor>>;

    /// Lower an equation with child regions. Each slice holds one region's
    /// captures in closure order; explicit operands are supplied separately.
    fn lower_torch_regions(
        &self,
        eqn: &Eqn,
        inputs: &[Tensor],
        _captures: &[Vec<Tensor>],
    ) -> io::Result<Vec<Tensor>> {
        self.lower_torch(eqn, inputs)
    }
}

#[macro_export]
macro_rules! register_torch_lowering {
    ($($op:ty),+ $(,)?) => {
        $crate::__gnx_expr::register_op_interface!(dyn $crate::TorchLowering: $($op),+);
    };
}

fn metadata(info: &VarInfo) -> io::Result<&ArrayInfo> {
    info.downcast()
        .map_err(|_| unsupported("Torch requires array values"))
}

pub(super) fn layout(info: &ArrayInfo) -> io::Result<(Vec<i64>, Kind)> {
    let dims = info
        .shape()
        .dims()
        .iter()
        .map(|d| match d {
            Dim::Fixed(n) => i64::try_from(*n).map_err(invalid),
            _ => Err(unsupported("Torch requires fixed dimensions")),
        })
        .collect::<io::Result<_>>()?;
    let kind = match info.dtype() {
        DType::Bool => Kind::Bool,
        DType::F32 => Kind::Float,
        DType::F64 => Kind::Double,
        DType::F16 => Kind::Half,
        DType::BF16 => Kind::BFloat16,
        DType::U8 => Kind::Uint8,
        DType::I8 => Kind::Int8,
        DType::I16 => Kind::Int16,
        DType::I32 => Kind::Int,
        DType::I64 => Kind::Int64,
        DType::C64 => Kind::ComplexFloat,
        DType::C128 => Kind::ComplexDouble,
        other => return Err(unsupported(format!("unsupported Torch dtype {other:?}"))),
    };
    Ok((dims, kind))
}

pub(super) fn check(t: &Tensor, info: &ArrayInfo, device: tch::Device) -> io::Result<()> {
    let (dims, kind) = layout(info)?;
    if t.device() != device || t.size() != dims || t.f_kind().map_err(io::Error::other)? != kind {
        return Err(invalid(format!(
            "Torch tensor does not match {device:?} array {info:?}"
        )));
    }
    Ok(())
}

fn emit(body: &Expr, inputs: &[Tensor]) -> io::Result<Tensor> {
    let [output] = emit_results(body, inputs)?
        .try_into()
        .map_err(|_| invalid("expected one region result"))?;
    Ok(output)
}

fn emit_results(body: &Expr, inputs: &[Tensor]) -> io::Result<Vec<Tensor>> {
    if inputs.len() != body.closure().len() + body.inputs().len() {
        return Err(invalid("region input arity mismatch"));
    }
    let mut values: HashMap<_, _> = body
        .closure()
        .iter()
        .chain(body.inputs())
        .zip(inputs)
        .map(|((v, _), t)| (*v, t.shallow_clone()))
        .collect();
    for (i, eqn) in body.eqns().iter().enumerate() {
        let op = eqn.op();
        let rule = op
            .interface::<dyn TorchLowering>()
            .map_err(|e| unsupported(e.to_string()))?;
        let operands: Vec<_> = eqn
            .inputs()
            .iter()
            .map(|v| values[v].shallow_clone())
            .collect();
        let captures = eqn
            .subexprs()
            .iter()
            .map(|region| {
                region
                    .closure()
                    .iter()
                    .map(|v| values[v].shallow_clone())
                    .collect()
            })
            .collect::<Vec<_>>();
        let results = rule
            .lower_torch_regions(eqn, &operands, &captures)
            .map_err(|e| io::Error::new(e.kind(), format!("equation {i} ({op}): {e}")))?;
        if results.len() != eqn.outputs().len() {
            return Err(invalid(format!("{op}: wrong result count")));
        }
        for ((v, info), t) in eqn.outputs().iter().zip(results) {
            check(&t, metadata(info)?, tracing_device())?;
            values.insert(*v, t);
        }
    }
    Ok(body
        .outputs()
        .iter()
        .map(|var| values[var].shallow_clone())
        .collect())
}

pub(super) fn trace(expr: &Expr, device: tch::Device) -> io::Result<Vec<(CModule, ArrayInfo)>> {
    let _scope = TraceScope(TRACE_DEVICE.replace(device));
    let examples = expr
        .closure()
        .iter()
        .chain(expr.inputs())
        .map(|(_, info)| {
            let (dims, kind) = layout(metadata(info)?)?;
            Tensor::f_zeros(&dims, (kind, tracing_device())).map_err(io::Error::other)
        })
        .collect::<io::Result<Vec<_>>>()?;
    // tch registers raw graph outputs, not a tuple. A callable method needs
    // one return value, so trace only the live equations for each result.
    expr.outputs()
        .iter()
        .map(|v| {
            let body = gnx_expr::pass::dce(&Expr::new(
                expr.closure().to_vec(),
                expr.inputs().to_vec(),
                expr.eqns().to_vec(),
                vec![*v],
            ));
            let mut error = None;
            let module = CModule::create_by_tracing("Gnx", "forward", &examples, &mut |inputs| {
                match emit(&body, inputs) {
                    Ok(t) => vec![t],
                    // Always return to tch so it can close the active trace.
                    Err(e) => {
                        error = Some(e);
                        vec![]
                    }
                }
            });
            if let Some(e) = error {
                return Err(e);
            }
            Ok((
                module.map_err(io::Error::other)?,
                metadata(expr.value_info(v).expect("checked output"))?.clone(),
            ))
        })
        .collect()
}

fn result_layout(eqn: &Eqn) -> io::Result<(Vec<i64>, Kind)> {
    let [(_, info)] = eqn.outputs() else {
        return Err(invalid("expected one result"));
    };
    layout(metadata(info)?)
}
fn axes(a: &[usize]) -> io::Result<Vec<i64>> {
    a.iter()
        .map(|&n| i64::try_from(n).map_err(invalid))
        .collect()
}

// Built-in single-result rules share arity checking and native error handling.
macro_rules! rule {
    ($ty:ty, |$op:ident, $eqn:ident, [$($arg:ident),*]| $body:block) => {
        impl TorchLowering for $ty {
            fn lower_torch(&self, $eqn: &Eqn, inputs: &[Tensor]) -> io::Result<Vec<Tensor>> {
                let $op = self;
                let [$($arg),*] = inputs else { return Err(invalid("incorrect Torch operand count")) };
                Ok(vec![$body.map_err(io::Error::other)?])
            }
        }
        crate::register_torch_lowering!($ty);
    }
}
mod beta;
mod bitwise;
mod contraction;
mod control_flow;
pub use control_flow::{trace_if, trace_while};
mod convolution;
mod elementwise;
mod igamma;
mod indexing;
mod linalg;
mod math;
mod ordering;
mod precision;
mod quadrature;
mod reduction;
mod schur;
mod structural;
mod tridiagonal_solve;
rule!(ZerosOperation, |_op, eqn, []| {
    let (dims, k) = result_layout(eqn)?;
    Tensor::f_zeros(dims, (k, tracing_device()))
});
rule!(OnesOperation, |_op, eqn, []| {
    let (dims, k) = result_layout(eqn)?;
    Tensor::f_ones(dims, (k, tracing_device()))
});
rule!(ReshapeOperation, |_op, eqn, [x]| {
    x.f_reshape(result_layout(eqn)?.0)
});
rule!(TransposeOperation, |op, _eqn, [x]| {
    x.f_permute(axes(op.perm())?)
});
rule!(BroadcastInDimOperation, |op, eqn, [x]| {
    let (target, _) = result_layout(eqn)?;
    let mut shape = vec![1; target.len()];
    let source = x.size();
    let dims = op.dims();
    if dims.len() != source.len()
        || dims.iter().any(|&d| d >= shape.len())
        || dims.windows(2).any(|w| w[0] >= w[1])
    {
        return Err(invalid("invalid broadcast dimensions"));
    }
    for (&d, s) in dims.iter().zip(source) {
        shape[d] = s;
    }
    x.f_reshape(shape).and_then(|t| t.f_expand(target, false))
});
rule!(ReduceAddOperation, |op, _eqn, [x]| {
    let dims = axes(op.axes())?;
    if dims.is_empty() {
        return Ok(vec![x.shallow_clone()]);
    }
    x.f_sum_dim_intlist(dims.as_slice(), false, x.kind())
});

//! Array metadata helpers and typed emitters shared by transformation rules.

use gnx_common::ops::{
    add_op, broadcast_in_dim_op, mul_op, rev_op, select_op, transpose_op, zeros_op,
};
use gnx_common::pass::Builder;
use gnx_common::{ArrayInfo, DType, IntoShape, fixed_dims};
use gnx_expr::{Eqn, ExprHandle, Ref, SubExprBuilder, Var, VarInfo};

use crate::{TransformError, TransformResult};

pub fn array(info: &VarInfo) -> TransformResult<ArrayInfo> {
    ArrayInfo::from_info(info).ok_or_else(|| {
        TransformError::invalid(format!("transformations only support arrays, found {info}"))
    })
}

pub fn dims(info: &VarInfo) -> TransformResult<Vec<usize>> {
    Ok(fixed_dims(&array(info)?)?)
}

pub fn dtype(info: &VarInfo) -> TransformResult<DType> {
    Ok(array(info)?.dtype())
}

pub fn info(dims: impl IntoShape, dtype: DType) -> VarInfo {
    VarInfo::new(ArrayInfo::new(dims, dtype))
}

/// Insert an axis of `size` at `axis`.
pub fn with_axis(value: &VarInfo, axis: usize, size: usize) -> TransformResult<VarInfo> {
    let mut shape = dims(value)?;
    if axis > shape.len() {
        return Err(TransformError::invalid("axis out of range"));
    }
    shape.insert(axis, size);
    Ok(info(&shape, dtype(value)?))
}

/// Remove `axis`, returning its size and the remaining metadata.
pub fn without_axis(value: &VarInfo, axis: usize) -> TransformResult<(usize, VarInfo)> {
    let mut shape = dims(value)?;
    if axis >= shape.len() {
        return Err(TransformError::invalid("axis out of range"));
    }
    let size = shape.remove(axis);
    Ok((size, info(&shape, dtype(value)?)))
}

pub fn is_float(value: &VarInfo) -> bool {
    matches!(
        dtype(value),
        Ok(DType::F16 | DType::BF16 | DType::F32 | DType::F64)
    )
}

pub fn zeros(b: &mut Builder, value: &VarInfo) -> Var {
    b.emit1(zeros_op(), &[], value.clone())
}

pub fn add(b: &mut Builder, x: Var, y: Var) -> Var {
    let value = b.info(x).clone();
    b.emit1(add_op(), &[x, y], value)
}

pub fn mul(b: &mut Builder, x: Var, y: Var) -> Var {
    let value = b.info(x).clone();
    b.emit1(mul_op(), &[x, y], value)
}

pub fn select(b: &mut Builder, predicate: Var, on_true: Var, on_false: Var) -> Var {
    let value = b.info(on_true).clone();
    b.emit1(select_op(), &[predicate, on_true, on_false], value)
}

pub fn broadcast_in_dim(
    b: &mut Builder,
    x: Var,
    shape: impl IntoShape,
    mapping: &[usize],
) -> TransformResult<Var> {
    let result = info(shape, dtype(b.info(x))?);
    Ok(b.emit1(broadcast_in_dim_op(mapping), &[x], result))
}

pub fn reduce_sum(b: &mut Builder, x: Var, axes: &[usize]) -> TransformResult<Var> {
    Ok(gnx_common::ops::reduction::reduce_add(b, x, axes)?)
}

/// Result axis `i` is operand axis `perm[i]`.
pub fn transpose(b: &mut Builder, x: Var, perm: &[usize]) -> TransformResult<Var> {
    if perm.iter().copied().eq(0..perm.len()) {
        return Ok(x);
    }
    let value = b.info(x).clone();
    let shape = dims(&value)?;
    let result: Vec<_> = perm.iter().map(|&axis| shape[axis]).collect();
    Ok(b.emit1(transpose_op(perm), &[x], info(&result, dtype(&value)?)))
}

pub fn inverse(perm: &[usize]) -> Vec<usize> {
    let mut inverse = vec![0; perm.len()];
    for (i, &axis) in perm.iter().enumerate() {
        inverse[axis] = i;
    }
    inverse
}

pub fn rev(b: &mut Builder, x: Var, axes: &[usize]) -> Var {
    let value = b.info(x).clone();
    b.emit1(rev_op(axes), &[x], value)
}

pub fn move_axis(b: &mut Builder, x: Var, from: usize, to: usize) -> TransformResult<Var> {
    if from == to {
        return Ok(x);
    }
    let rank = dims(b.info(x))?.len();
    let mut perm: Vec<_> = (0..rank).filter(|&axis| axis != from).collect();
    perm.insert(to, from);
    transpose(b, x, &perm)
}

/// Insert a new axis of `size` at `axis` by broadcasting.
pub fn broadcast_axis(b: &mut Builder, x: Var, axis: usize, size: usize) -> TransformResult<Var> {
    let shape = dims(&with_axis(b.info(x), axis, size)?)?;
    let mapping: Vec<_> = (0..shape.len()).filter(|&i| i != axis).collect();
    broadcast_in_dim(b, x, &shape, &mapping)
}

/// Re-emit `eqn` unchanged except for its operands and region captures.
pub fn copy_eqn(b: &mut Builder, eqn: Eqn<'_>, inputs: &[Var], captures: &[Vec<Var>]) -> Vec<Var> {
    let subexprs = eqn
        .subexprs()
        .zip(captures)
        .map(|(sub, captures)| {
            SubExprBuilder::new(sub.to_ref())
                .with_vars(captures.iter().copied())
                .with_symbols(sub.provided_symbols().iter().copied())
        })
        .collect();
    b.emit(eqn.op().to_owned(), inputs, output_infos(eqn), subexprs)
}

pub fn output_infos(eqn: Eqn<'_>) -> Vec<VarInfo> {
    eqn.outputs()
        .map(|param| param.info_ref().to_owned())
        .collect()
}

/// The only region of an equation, if it has exactly one.
pub fn single_region(eqn: Eqn<'_>) -> Option<gnx_expr::SubExpr<'_>> {
    let mut regions = eqn.subexprs();
    match (regions.next(), regions.next()) {
        (Some(region), None) => Some(region),
        _ => None,
    }
}

/// The metadata of the explicit inputs of an expression.
pub fn input_infos(expr: gnx_expr::Expr<'_>) -> Vec<VarInfo> {
    expr.inputs()
        .map(|param| param.info_ref().to_owned())
        .collect()
}

/// A readable handle to the body of a region of an equation under
/// construction. A body that is itself still under construction is built
/// into a root of its own.
pub fn region_body(sub: &SubExprBuilder) -> TransformResult<ExprHandle> {
    Ok(match sub.expr() {
        Ref::Internal(expr) => gnx_expr::ExprBuilder::clone(expr).build()?.handle(),
        Ref::External(handle) => handle.clone(),
    })
}

/// The metadata of variables bound in `b`.
pub fn infos(b: &Builder, vars: &[Var]) -> Vec<VarInfo> {
    vars.iter().map(|&var| b.info(var).clone()).collect()
}

/// Set every flag in `lhs` that is set in `rhs`; returns whether any changed.
/// Used to reach fixed points over carried-value flags (nonzero tangents,
/// batched carries).
pub fn union(lhs: &mut [bool], rhs: &[bool]) -> bool {
    let mut changed = false;
    for (lhs, &rhs) in lhs.iter_mut().zip(rhs) {
        if rhs && !*lhs {
            *lhs = true;
            changed = true;
        }
    }
    changed
}

pub fn is_inexact(value: &VarInfo) -> bool {
    is_float(value) || matches!(dtype(value), Ok(DType::C64 | DType::C128))
}

/// The value metadata of the inputs and results of a traced body.
pub(crate) fn value_signature(
    body: gnx_expr::HExpr<'_>,
) -> Result<(Vec<VarInfo>, Vec<VarInfo>), gnx_expr::LoweringError> {
    Ok((
        crate::cf::value_infos(body.inputs())?,
        crate::cf::value_results(body)?,
    ))
}

/// The metadata of the inputs and results of the one region of an equation.
pub(crate) fn region_signature(
    eqn: Eqn<'_>,
) -> Result<(Vec<VarInfo>, Vec<VarInfo>), gnx_expr::LoweringError> {
    use gnx_expr::LoweringError;
    let body = single_region(eqn)
        .ok_or(LoweringError::RegionSignature(
            "transformations take exactly one body",
        ))?
        .expr();
    let outputs = (body.outputs().iter())
        .map(|&var| {
            (body.value_info(var).map(|info| info.to_owned()))
                .ok_or(LoweringError::UndefinedVariable(var))
        })
        .collect::<Result<_, _>>()?;
    Ok((input_infos(body), outputs))
}

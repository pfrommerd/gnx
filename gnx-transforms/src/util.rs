//! Array metadata helpers and typed emitters shared by transformation rules.

use gnx_array::{ArrayInfo, DType, Shape, fixed_dims};
use gnx_expr::pass::Builder;
use gnx_expr::{Eqn, SubExpr, ValueInfo, Var};
use gnx_math_base::ops::{
    add_op, broadcast_in_dim_op, mul_op, rev_op, select_op, transpose_op, zeros_op,
};

use crate::{TransformError, TransformResult};

pub fn array(info: &ValueInfo) -> TransformResult<&ArrayInfo> {
    info.downcast::<ArrayInfo>().map_err(|_| {
        TransformError::invalid(format!("transformations only support arrays, found {info}"))
    })
}

pub fn dims(info: &ValueInfo) -> TransformResult<Vec<usize>> {
    Ok(fixed_dims(array(info)?)?)
}

pub fn dtype(info: &ValueInfo) -> TransformResult<DType> {
    Ok(array(info)?.dtype())
}

pub fn info(dims: &[usize], dtype: DType) -> ValueInfo {
    ValueInfo::new(ArrayInfo::new(Shape::fixed(dims.iter().copied()), dtype))
}

/// Insert an axis of `size` at `axis`.
pub fn with_axis(value: &ValueInfo, axis: usize, size: usize) -> TransformResult<ValueInfo> {
    let mut shape = dims(value)?;
    if axis > shape.len() {
        return Err(TransformError::invalid("axis out of range"));
    }
    shape.insert(axis, size);
    Ok(info(&shape, dtype(value)?))
}

/// Remove `axis`, returning its size and the remaining metadata.
pub fn without_axis(value: &ValueInfo, axis: usize) -> TransformResult<(usize, ValueInfo)> {
    let mut shape = dims(value)?;
    if axis >= shape.len() {
        return Err(TransformError::invalid("axis out of range"));
    }
    let size = shape.remove(axis);
    Ok((size, info(&shape, dtype(value)?)))
}

pub fn is_float(value: &ValueInfo) -> bool {
    matches!(
        dtype(value),
        Ok(DType::F16 | DType::BF16 | DType::F32 | DType::F64)
    )
}

pub fn zeros(b: &mut Builder, value: &ValueInfo) -> Var {
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
    shape: &[usize],
    mapping: &[usize],
) -> TransformResult<Var> {
    let result = info(shape, dtype(b.info(x))?);
    Ok(b.emit1(broadcast_in_dim_op(mapping), &[x], result))
}

pub fn reduce_sum(b: &mut Builder, x: Var, axes: &[usize]) -> TransformResult<Var> {
    Ok(gnx_math_base::ops::reduction::reduce_add(b, x, axes)?)
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
pub fn copy_eqn(b: &mut Builder, eqn: &Eqn, inputs: &[Var], captures: &[Vec<Var>]) -> Vec<Var> {
    let subexprs = eqn
        .subexprs()
        .iter()
        .zip(captures)
        .map(|(sub, captures)| SubExpr::new(sub.expr().clone(), captures.clone()))
        .collect();
    b.emit(
        eqn.op().clone(),
        inputs,
        eqn.outputs().iter().map(|(_, value)| value.clone()),
        subexprs,
    )
}

pub fn output_infos(eqn: &Eqn) -> Vec<ValueInfo> {
    eqn.outputs()
        .iter()
        .map(|(_, value)| value.clone())
        .collect()
}

/// The metadata of variables bound in `b`.
pub fn infos(b: &Builder, vars: &[Var]) -> Vec<ValueInfo> {
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

pub fn is_inexact(value: &ValueInfo) -> bool {
    is_float(value) || matches!(dtype(value), Ok(DType::C64 | DType::C128))
}

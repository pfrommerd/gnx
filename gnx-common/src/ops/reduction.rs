//! Axis reductions used by array operations and transform rules.

use std::sync::Arc;

use crate::{ArrayInfo, Shape, fixed_dims};
use gnx_expr::pass::Builder;
use gnx_expr::{Expr, SubExpr, ValueInfo, Var};

use super::{
    Dialect, Eqn, Op, Operation, Result, arity, fmt, result_rank, signature, strictly_increasing,
};

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct ReduceOperation {
    axes: Vec<usize>,
}

impl ReduceOperation {
    pub fn axes(&self) -> &[usize] {
        &self.axes
    }
}

array_operation!(ReduceOperation, "reduce", |op, eqn| {
    let rank = check_region(eqn)?;
    let rank = rank.checked_add(op.axes.len()).ok_or_else(|| signature("reduce rank overflow"))?;
    strictly_increasing(&op.axes, rank)
}, axes;
    fn lower(&self, eqn: &gnx_expr::HEqn, ctx: &mut gnx_expr::lower::LoweringContext<'_>) -> Result<()> {
        ctx.lower_closed_regions_apply(eqn)
    }
);

/// Sum along selected axes, using logical OR for boolean arrays.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct ReduceAddOperation {
    axes: Vec<usize>,
}

impl ReduceAddOperation {
    pub fn axes(&self) -> &[usize] {
        &self.axes
    }
}

array_operation!(
    ReduceAddOperation,
    "reduce_add",
    |op, eqn| {
        arity(eqn, 1)?;
        let input_rank = result_rank(eqn)?
            .checked_add(op.axes.len())
            .ok_or_else(|| signature("reduce_add rank overflow"))?;
        strictly_increasing(&op.axes, input_rank)
    },
    axes
);

#[doc(hidden)]
pub fn array(info: &ValueInfo) -> Result<&ArrayInfo> {
    info.downcast::<ArrayInfo>()
        .map_err(|_| signature("reduction values must be arrays"))
}

#[doc(hidden)]
pub fn check_region(eqn: &Eqn) -> Result<usize> {
    let count = eqn.outputs().len();
    let [region] = eqn.subexprs() else {
        return Err(signature("reduction needs one reducer region"));
    };
    if count == 0 || eqn.inputs().len() != 2 * count || !region.closure().is_empty() {
        return Err(signature("invalid reducer arity or captures"));
    }
    let output = array(&eqn.outputs()[0].1)?;
    let mut dtypes = Vec::with_capacity(count);
    for (_, result) in eqn.outputs() {
        let result = array(result)?;
        if result.shape() != output.shape() {
            return Err(signature("reduction outputs must share one shape"));
        }
        dtypes.push(result.dtype());
    }
    check_reducer(region.expr(), &dtypes)?;
    Ok(output.shape().dims().len())
}

fn check_reducer(reducer: &Expr, dtypes: &[crate::DType]) -> Result<()> {
    let count = dtypes.len();
    if !reducer.closure().is_empty()
        || reducer.inputs().len() != 2 * count
        || reducer.outputs().len() != count
    {
        return Err(signature("invalid reducer arity or captures"));
    }
    for (i, &dtype) in dtypes.iter().enumerate() {
        let scalar = ValueInfo::new(ArrayInfo::new(Shape::fixed([]), dtype));
        if reducer.inputs()[i].1 != scalar
            || reducer.inputs()[i + count].1 != scalar
            || reducer.value_info(&reducer.outputs()[i]) != Some(&scalar)
        {
            return Err(signature(
                "reducer argument and result metadata must match corresponding scalars",
            ));
        }
    }
    Ok(())
}

pub fn reduce_op(axes: &[usize]) -> Op {
    Op::new(ReduceOperation {
        axes: axes.to_vec(),
    })
}

fn reduced_shape(shape: &[usize], axes: &[usize]) -> Result<Vec<usize>> {
    strictly_increasing(axes, shape.len())?;
    Ok(shape
        .iter()
        .enumerate()
        .filter_map(|(axis, &size)| (!axes.contains(&axis)).then_some(size))
        .collect())
}

/// Emit a variadic reduction. The caller supplies an associative reducer and its identities.
pub fn reduce(
    builder: &mut Builder,
    operands: &[Var],
    initial: &[Var],
    axes: &[usize],
    reducer: Arc<Expr>,
) -> Result<Vec<Var>> {
    emit(
        builder,
        operands,
        initial,
        reducer,
        reduce_op(axes),
        |shape| reduced_shape(shape, axes),
    )
}

#[doc(hidden)]
pub fn emit(
    builder: &mut Builder,
    operands: &[Var],
    initial: &[Var],
    reducer: Arc<Expr>,
    op: Op,
    result_shape: impl FnOnce(&[usize]) -> Result<Vec<usize>>,
) -> Result<Vec<Var>> {
    let Some(first) = operands.first() else {
        return Err(signature("reduction needs at least one operand"));
    };
    if operands.len() != initial.len() {
        return Err(signature("reduction needs one identity per operand"));
    }
    let input = array(builder.info(*first))?;
    let dims = fixed_dims(input).map_err(|_| signature("reduction requires fixed dimensions"))?;
    let shape = Shape::fixed(result_shape(&dims)?);
    let mut results = Vec::with_capacity(operands.len());
    let mut dtypes = Vec::with_capacity(operands.len());
    for (&operand, &init) in operands.iter().zip(initial) {
        let operand = array(builder.info(operand))?;
        let init = array(builder.info(init))?;
        if operand.shape() != input.shape()
            || !init.shape().dims().is_empty()
            || init.dtype() != operand.dtype()
        {
            return Err(signature(
                "reduction operands must share a shape and match their scalar identities",
            ));
        }
        dtypes.push(operand.dtype());
        results.push(ValueInfo::new(ArrayInfo::new(
            shape.clone(),
            operand.dtype(),
        )));
    }
    let inputs: Vec<_> = operands.iter().chain(initial).copied().collect();
    check_reducer(&reducer, &dtypes)?;
    reducer.check()?;
    Ok(builder.emit(op, &inputs, results, vec![SubExpr::new(reducer, vec![])]))
}

pub fn reduce_add_op(axes: &[usize]) -> Op {
    Op::new(ReduceAddOperation {
        axes: axes.to_vec(),
    })
}

/// Emit a sum for transform rules.
pub fn reduce_add(builder: &mut Builder, operand: Var, axes: &[usize]) -> Result<Var> {
    let input = array(builder.info(operand))?;
    let dims = fixed_dims(input).map_err(|_| signature("reduction requires fixed dimensions"))?;
    let shape = Shape::fixed(reduced_shape(&dims, axes)?);
    let result = ValueInfo::new(ArrayInfo::new(shape, input.dtype()));
    Ok(builder.emit1(reduce_add_op(axes), &[operand], result))
}

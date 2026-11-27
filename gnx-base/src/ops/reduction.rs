//! Axis reductions used by array operations and transform rules.

use crate::pass::Builder;
use crate::{ArrayInfo, Shape, fixed_dims};
use gnx_expr::{Eqn, Expr, ExprHandle, Op, SubExprBuilder, Var, VarInfo};
use rkyv::{Archive, Serialize};

use super::{Result, arity, indices, result_rank, signature, stored_indices, strictly_increasing};

#[derive(Archive, Serialize)]
#[rkyv(archived = ReduceOperation)]
pub struct Reduce {
    axes: Vec<u32>,
}

impl ReduceOperation {
    pub fn axes(&self) -> Vec<usize> {
        indices(&self.axes)
    }
}

array_operation!(ReduceOperation, "reduce", |op, eqn| {
    let rank = check_region(eqn)?;
    let axes = op.axes();
    let rank = rank.checked_add(axes.len()).ok_or_else(|| signature("reduce rank overflow"))?;
    strictly_increasing(&axes, rank)
}, axes;
    fn lower(&self, eqn: gnx_expr::HEqn<'_>, ctx: &mut gnx_expr::lower::LoweringContext<'_>) -> Result<()> {
        ctx.lower_closed_regions_apply(eqn)
    }
);

/// Sum along selected axes, using logical OR for boolean arrays.
#[derive(Archive, Serialize)]
#[rkyv(archived = ReduceAddOperation)]
pub struct ReduceAdd {
    axes: Vec<u32>,
}

impl ReduceAddOperation {
    pub fn axes(&self) -> Vec<usize> {
        indices(&self.axes)
    }
}

array_operation!(
    ReduceAddOperation,
    "reduce_add",
    |op, eqn| {
        arity(eqn, 1)?;
        let axes = op.axes();
        let input_rank = result_rank(eqn)?
            .checked_add(axes.len())
            .ok_or_else(|| signature("reduce_add rank overflow"))?;
        strictly_increasing(&axes, input_rank)
    },
    axes
);

#[doc(hidden)]
pub fn array(info: &VarInfo) -> Result<ArrayInfo> {
    ArrayInfo::from_info(info).ok_or_else(|| signature("reduction values must be arrays"))
}

#[doc(hidden)]
pub fn check_region(eqn: Eqn<'_>) -> Result<usize> {
    let count = eqn.outputs().len();
    let mut regions = eqn.subexprs();
    let (Some(region), None) = (regions.next(), regions.next()) else {
        return Err(signature("reduction needs one reducer region"));
    };
    if count == 0 || eqn.inputs().len() != 2 * count || !region.provided_vars().is_empty() {
        return Err(signature("invalid reducer arity or captures"));
    }
    let results = (eqn.outputs())
        .map(|output| array(&output.info_ref().to_owned()))
        .collect::<Result<Vec<_>>>()?;
    let output = &results[0];
    let mut dtypes = Vec::with_capacity(count);
    for result in &results {
        if result.shape() != output.shape() {
            return Err(signature("reduction outputs must share one shape"));
        }
        dtypes.push(result.dtype());
    }
    check_reducer(region.expr(), &dtypes)?;
    Ok(output.shape().dims().len())
}

fn check_reducer(reducer: Expr<'_>, dtypes: &[crate::DType]) -> Result<()> {
    let count = dtypes.len();
    if reducer.closure_inputs().len() != 0
        || reducer.inputs().len() != 2 * count
        || reducer.outputs().len() != count
    {
        return Err(signature("invalid reducer arity or captures"));
    }
    let inputs: Vec<_> = reducer.inputs().collect();
    for (i, &dtype) in dtypes.iter().enumerate() {
        let scalar = VarInfo::new(ArrayInfo::new(Shape::fixed([]), dtype));
        if inputs[i].info_ref() != scalar
            || inputs[i + count].info_ref() != scalar
            || reducer
                .value_info(reducer.outputs()[i])
                .is_none_or(|info| info != scalar)
        {
            return Err(signature(
                "reducer argument and result metadata must match corresponding scalars",
            ));
        }
    }
    Ok(())
}

pub fn reduce_op(axes: &[usize]) -> Op {
    Op::new(Reduce {
        axes: stored_indices(axes),
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
    reducer: ExprHandle,
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
    reducer: ExprHandle,
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
    let dims = fixed_dims(&input).map_err(|_| signature("reduction requires fixed dimensions"))?;
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
        results.push(VarInfo::new(ArrayInfo::new(shape.clone(), operand.dtype())));
    }
    let inputs: Vec<_> = operands.iter().chain(initial).copied().collect();
    check_reducer(reducer.view(), &dtypes)?;
    reducer.view().check()?;
    Ok(builder.emit(op, &inputs, results, vec![SubExprBuilder::new(reducer)]))
}

pub fn reduce_add_op(axes: &[usize]) -> Op {
    Op::new(ReduceAdd {
        axes: stored_indices(axes),
    })
}

/// Emit a sum for transform rules.
pub fn reduce_add(builder: &mut Builder, operand: Var, axes: &[usize]) -> Result<Var> {
    let input = array(builder.info(operand))?;
    let dims = fixed_dims(&input).map_err(|_| signature("reduction requires fixed dimensions"))?;
    let shape = Shape::fixed(reduced_shape(&dims, axes)?);
    let result = VarInfo::new(ArrayInfo::new(shape, input.dtype()));
    Ok(builder.emit1(reduce_add_op(axes), &[operand], result))
}

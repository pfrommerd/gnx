use super::{Dialect, Eqn, Op, Operation, Result, arity, fmt, result_rank, signature};

pub use super::{
    broadcast_in_dim_op as broadcast_in_dim, reshape_op as reshape, rev_op as rev,
    transpose_op as transpose,
};

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct ConcatenateOperation {
    axis: usize,
}
impl ConcatenateOperation {
    pub fn axis(&self) -> usize {
        self.axis
    }
}
gnx_common::array_operation!(
    ConcatenateOperation,
    "concatenate",
    |op, eqn| {
        if eqn.inputs().is_empty() {
            return Err(signature("concatenate needs an operand"));
        }
        arity(eqn, eqn.inputs().len())?;
        if op.axis >= result_rank(eqn)? {
            return Err(signature("concatenate axis is out of range"));
        }
        Ok(())
    },
    axis
);
pub fn concatenate(axis: usize) -> Op {
    Op::new(ConcatenateOperation { axis })
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct PadOperation {
    padding: Vec<(i64, i64, usize)>,
}
impl PadOperation {
    pub fn padding(&self) -> &[(i64, i64, usize)] {
        &self.padding
    }
}
gnx_common::array_operation!(
    PadOperation,
    "pad",
    |op, eqn| {
        arity(eqn, 2)?;
        if op.padding.len() != result_rank(eqn)? {
            return Err(signature("padding must have result rank"));
        }
        Ok(())
    },
    padding
);
pub fn pad(padding: &[(i64, i64, usize)]) -> Op {
    Op::new(PadOperation {
        padding: padding.to_vec(),
    })
}

use gnx_common::{ArrayInfo, DType, Shape, fixed_dims};
use gnx_expr::pass::Builder;
use gnx_expr::{Var, VarInfo};

fn metadata(b: &Builder, operand: Var) -> Result<(Vec<usize>, DType)> {
    let info = b
        .info(operand)
        .downcast::<ArrayInfo>()
        .map_err(|_| signature("expected array operand"))?;
    Ok((
        fixed_dims(info).map_err(|_| signature("operation requires fixed dimensions"))?,
        info.dtype(),
    ))
}

fn info(dimensions: &[usize], dtype: DType) -> VarInfo {
    VarInfo::new(ArrayInfo::new(
        Shape::fixed(dimensions.iter().copied()),
        dtype,
    ))
}

pub fn squeeze(b: &mut Builder, operand: Var, axes: &[usize]) -> Result<Var> {
    let (shape, dtype) = metadata(b, operand)?;
    super::strictly_increasing(axes, shape.len())?;
    if axes.iter().any(|&axis| shape[axis] != 1) {
        return Err(signature("squeeze requires unit dimensions"));
    }
    let result: Vec<_> = shape
        .into_iter()
        .enumerate()
        .filter_map(|(axis, size)| (!axes.contains(&axis)).then_some(size))
        .collect();
    Ok(b.emit1(reshape(), &[operand], info(&result, dtype)))
}

pub fn split(b: &mut Builder, operand: Var, axis: usize, sizes: &[usize]) -> Result<Vec<Var>> {
    let (mut shape, dtype) = metadata(b, operand)?;
    let extent = *shape
        .get(axis)
        .ok_or_else(|| signature("split axis out of range"))?;
    let sum = sizes
        .iter()
        .try_fold(0usize, |total, &size| total.checked_add(size));
    if sum != Some(extent) {
        return Err(signature("split sizes must sum to the operand dimension"));
    }
    let mut ranges: Vec<_> = shape.iter().map(|&size| (0, size, 1)).collect();
    let mut start = 0;
    let mut values = Vec::with_capacity(sizes.len());
    for &size in sizes {
        ranges[axis] = (start, start + size, 1);
        shape[axis] = size;
        values.push(b.emit1(
            super::indexing::slice(&ranges),
            &[operand],
            info(&shape, dtype),
        ));
        start += size;
    }
    Ok(values)
}

pub fn stack(b: &mut Builder, operands: &[Var], axis: usize) -> Result<Var> {
    let first = *operands
        .first()
        .ok_or_else(|| signature("stack needs an operand"))?;
    let (mut shape, dtype) = metadata(b, first)?;
    if axis > shape.len() || operands.iter().any(|&x| b.info(x) != b.info(first)) {
        return Err(signature(
            "stack operands must have matching metadata and a valid axis",
        ));
    }
    shape.insert(axis, 1);
    let values: Vec<_> = operands
        .iter()
        .map(|&operand| b.emit1(reshape(), &[operand], info(&shape, dtype)))
        .collect();
    shape[axis] = operands.len();
    Ok(b.emit1(concatenate(axis), &values, info(&shape, dtype)))
}

pub fn unstack(b: &mut Builder, operand: Var, axis: usize) -> Result<Vec<Var>> {
    let (shape, _) = metadata(b, operand)?;
    let size = *shape
        .get(axis)
        .ok_or_else(|| signature("unstack axis out of range"))?;
    split(b, operand, axis, &vec![1; size])?
        .into_iter()
        .map(|value| squeeze(b, value, &[axis]))
        .collect()
}

pub fn tile(b: &mut Builder, operand: Var, reps: &[usize]) -> Result<Var> {
    let (shape, dtype) = metadata(b, operand)?;
    let rank = shape.len().max(reps.len());
    let input: Vec<_> = std::iter::repeat_n(1, rank - shape.len())
        .chain(shape.iter().copied())
        .collect();
    let reps: Vec<_> = std::iter::repeat_n(1, rank - reps.len())
        .chain(reps.iter().copied())
        .collect();
    let result: Vec<_> = reps
        .iter()
        .zip(&input)
        .map(|(&repeat, &size)| {
            repeat
                .checked_mul(size)
                .ok_or_else(|| signature("tile dimension overflow"))
        })
        .collect::<Result<_>>()?;
    let expanded: Vec<_> = reps
        .iter()
        .zip(&input)
        .flat_map(|(&reps, &size)| [reps, size])
        .collect();
    let mapping: Vec<_> = (rank - shape.len()..rank).map(|i| 2 * i + 1).collect();
    let repeated = b.emit1(
        broadcast_in_dim(&mapping),
        &[operand],
        info(&expanded, dtype),
    );
    Ok(b.emit1(reshape(), &[repeated], info(&result, dtype)))
}

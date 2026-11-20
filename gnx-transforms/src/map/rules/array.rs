//! Batching rules for the `array` dialect.
//!
//! `zeros`, `ones` and `eye` take no operands, so they are never batched and
//! need no rule.

use gnx_common::ops::{
    AddOperation, BroadcastInDimOperation, MulOperation, ReshapeOperation, RevOperation,
    SelectOperation, TransposeOperation, broadcast_in_dim_op, reshape_op, rev_op, transpose_op,
};

use crate::map::def::{defbatch, defbroadcasting, leading, shifted};

use gnx_common::ops::{DivOperation, RemOperation, SubOperation};
defbroadcasting!(
    AddOperation,
    SubOperation,
    MulOperation,
    DivOperation,
    RemOperation,
    SelectOperation
);

defbatch!(BroadcastInDimOperation, |op| Ok(broadcast_in_dim_op(
    &leading(op.dims())
)));
defbatch!(TransposeOperation, |op| Ok(transpose_op(&leading(
    op.perm()
))));
defbatch!(RevOperation, |op| Ok(rev_op(&shifted(op.axes()))));
defbatch!(ReshapeOperation, |_| Ok(reshape_op()));

use crate::map::batch::materialize;
use crate::map::{BatchRule, Batched};
use crate::util::with_axis;
use crate::{TransformError, TransformResult};
use gnx_common::ops::reduction::{ReduceAddOperation, ReduceOperation, reduce_add_op, reduce_op};
use gnx_expr::pass::Builder;
use gnx_expr::{Eqn, Op};

defbatch!(ReduceAddOperation, |op| Ok(reduce_add_op(&shifted(
    op.axes()
))));

pub fn batch_reduce(
    b: &mut Builder,
    size: usize,
    eqn: &Eqn,
    inputs: &[Batched],
    op: Op,
) -> TransformResult<Vec<Batched>> {
    let (operands, initial) = inputs.split_at(eqn.outputs().len());
    if initial.iter().any(|value| value.axis.is_some()) {
        return Err(TransformError::invalid("cannot batch reduction identities"));
    }
    let mut inputs = operands
        .iter()
        .map(|&value| materialize(b, value, size, 0))
        .collect::<TransformResult<Vec<_>>>()?;
    inputs.extend(initial.iter().map(|value| value.var));
    let outputs = eqn
        .outputs()
        .iter()
        .map(|(_, info)| with_axis(info, 0, size))
        .collect::<TransformResult<Vec<_>>>()?;
    Ok(b.emit(op, &inputs, outputs, eqn.subexprs().to_vec())
        .into_iter()
        .map(|var| Batched::at(var, 0))
        .collect())
}

impl BatchRule for ReduceOperation {
    fn batch(
        &self,
        b: &mut Builder,
        size: usize,
        eqn: &Eqn,
        inputs: &[Batched],
        _: &[Vec<Batched>],
    ) -> TransformResult<Vec<Batched>> {
        batch_reduce(b, size, eqn, inputs, reduce_op(&shifted(self.axes())))
    }
}

gnx_expr::register_op_interface!(dyn BatchRule: ReduceOperation);

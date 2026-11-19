//! Batching rules for the `array` dialect.
//!
//! `zeros`, `ones` and `eye` take no operands, so they are never batched and
//! need no rule.

use gnx_math::ops::{
    AddOperation, BroadcastInDimOperation, MulOperation, ReduceSumOperation, ReshapeOperation,
    RevOperation, SelectOperation, TransposeOperation, broadcast_in_dim_op, reduce_sum_op,
    reshape_op, rev_op, transpose_op,
};

use crate::map::def::{defbatch, defbroadcasting, leading, shifted};

defbroadcasting!(AddOperation, MulOperation, SelectOperation);

defbatch!(BroadcastInDimOperation, |op| Ok(broadcast_in_dim_op(
    &leading(op.dims())
)));
defbatch!(ReduceSumOperation, |op| Ok(reduce_sum_op(&shifted(
    op.axes()
))));
defbatch!(TransposeOperation, |op| Ok(transpose_op(&leading(
    op.perm()
))));
defbatch!(RevOperation, |op| Ok(rev_op(&shifted(op.axes()))));
defbatch!(ReshapeOperation, |_| Ok(reshape_op()));

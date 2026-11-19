pub use gnx_array::fixed_dims;
pub use gnx_array::ops::{
    AddOperation, ArrayDialect, BroadcastInDimOperation, EyeOperation, MulOperation, OnesOperation,
    ReshapeOperation, RevOperation, SelectOperation, TransposeOperation, ZerosOperation, add_op,
    array, broadcast_in_dim_op, eye_op, mul_op, ones_op, reshape_op, rev_op, select_op,
    transpose_op, zeros_op,
};

use std::fmt;

use gnx_array::ops::{arity, result_rank, signature, strictly_increasing};
use gnx_expr::{Dialect, Eqn, Op, Operation};

type Result<T> = std::result::Result<T, gnx_expr::LoweringError>;

macro_rules! simple_operations {
    ($($ty:ident, $ctor:ident, $alias:ident, $arity:literal;)*) => {$(
        #[derive(Clone, PartialEq, Eq, Hash)]
        pub struct $ty;

        gnx_array::array_operation!($ty, stringify!($alias), |eqn| arity(eqn, $arity));

        pub fn $ctor() -> Op {
            Op::new($ty)
        }

        pub use $ctor as $alias;
    )*};
}

pub mod bitwise;
pub mod contraction;
pub mod creation;
pub mod dtype;
pub mod elementwise;
pub mod fft;
pub mod indexing;
pub mod linalg;
pub mod ordering;
pub mod reduction;
pub mod shape;
pub mod special;

/// Constructors for the sharding dialect in expression macros.
pub mod shard {
    pub use crate::shard::{constraint_op as constraint, replicate_op as replicate};
}

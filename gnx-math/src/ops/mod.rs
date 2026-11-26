pub use gnx_common::fixed_dims;
pub use gnx_common::ops::{
    AddOperation, BroadcastInDimOperation, EyeOperation, MulOperation, OnesOperation,
    ReshapeOperation, RevOperation, SelectOperation, TransposeOperation, ZerosOperation, add_op,
    array, broadcast_in_dim_op, eye_op, mul_op, ones_op, reshape_op, rev_op, select_op,
    transpose_op, zeros_op,
};

use gnx_common::ops::{arity, result_rank, signature, strictly_increasing};
use gnx_expr::{Eqn, Op};
use rkyv::rancor::{Panic, Strategy};
use rkyv::rend::u64_le;
use rkyv::{Archive, Deserialize, de::Pool};

type Result<T> = std::result::Result<T, gnx_expr::LoweringError>;

/// The native form of parameters stored in an operation.
fn native<T: Archive>(stored: &T::Archived) -> T
where
    T::Archived: Deserialize<T, Strategy<Pool, Panic>>,
{
    match rkyv::deserialize::<T, Panic>(stored) {
        Ok(value) => value,
        Err(never) => match never {},
    }
}

/// A count or axis stored in an operation.
fn count(stored: u64_le) -> usize {
    usize::try_from(stored.to_native()).expect("operation parameter exceeds usize")
}

/// Parameter structures are stored with 32-bit sizes: refuse the ones that
/// would not be stored as they are.
fn stored_exactly(exact: bool) {
    assert!(exact, "operation parameter exceeds u32::MAX");
}

macro_rules! simple_operations {
    ($($ty:ident, $ctor:ident, $alias:ident, $name:literal, $arity:literal;)*) => {$(
        gnx_common::unit_operations! { pub struct $ty; }

        gnx_common::array_operation!($ty, $name, |eqn| arity(eqn, $arity));

        pub fn $ctor() -> Op {
            Op::new($ty)
        }

        pub use $ctor as $alias;
    )*};
}

pub mod bitwise;
pub mod bounded;
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

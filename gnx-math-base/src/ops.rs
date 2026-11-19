//! The `array` dialect: elementwise and structural array operations.
//!
//! Operands of binary elementwise operations must have identical shapes and
//! dtypes; there is no implicit broadcasting. Structural operations carry their
//! index parameters as typed fields and take result metadata from the
//! equation.

use std::fmt;

use gnx_expr::{Dialect, Eqn, LoweringError, Op, Operation};

use gnx_array::ArrayInfo;

pub struct ArrayDialect;
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct AddOperation;
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct MulOperation;
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct ZerosOperation;
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct OnesOperation;
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct EyeOperation;
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct ReshapeOperation;
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct BroadcastInDimOperation {
    dims: Vec<usize>,
}

impl BroadcastInDimOperation {
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }
}
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct TransposeOperation {
    perm: Vec<usize>,
}

impl TransposeOperation {
    pub fn perm(&self) -> &[usize] {
        &self.perm
    }
}
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct RevOperation {
    axes: Vec<usize>,
}

impl RevOperation {
    pub fn axes(&self) -> &[usize] {
        &self.axes
    }
}
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct SelectOperation;

static ARRAY: ArrayDialect = ArrayDialect;

#[doc(hidden)]
pub fn array_dialect() -> &'static dyn Dialect {
    &ARRAY
}

type Result<T> = std::result::Result<T, LoweringError>;

#[doc(hidden)]
pub fn signature(message: &'static str) -> LoweringError {
    LoweringError::RegionSignature(message)
}

impl Dialect for ArrayDialect {
    fn name(&self) -> &'static str {
        "array"
    }
}

#[doc(hidden)]
pub fn arity(eqn: &Eqn, inputs: usize) -> Result<()> {
    if eqn.inputs().len() != inputs || eqn.outputs().len() != 1 || !eqn.subexprs().is_empty() {
        return Err(signature("array operation signature"));
    }
    Ok(())
}

#[doc(hidden)]
pub fn result_rank(eqn: &Eqn) -> Result<usize> {
    let (_, info) = &eqn.outputs()[0];
    Ok(info
        .downcast::<ArrayInfo>()
        .map_err(|_| signature("array operations produce arrays"))?
        .shape()
        .dims()
        .len())
}

#[doc(hidden)]
pub fn strictly_increasing(indices: &[usize], bound: usize) -> Result<()> {
    if indices.windows(2).any(|pair| pair[0] >= pair[1]) || indices.iter().any(|&i| i >= bound) {
        return Err(signature(
            "indices must be strictly increasing and in range",
        ));
    }
    Ok(())
}

#[doc(hidden)]
#[macro_export]
macro_rules! array_operation {
    ($ty:ty, $name:expr, |$eqn:ident| $check:expr) => {
        $crate::array_operation!($ty, $name, |_op, $eqn| $check);
    };
    (
        $ty:ty, $name:expr, |$operation:ident, $eqn:ident| $check:expr
        $(, $field:ident)? $(; $($extra:item)*)?
    ) => {
        impl fmt::Display for $ty {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(concat!("array::", $name))?;
                $(write!(f, "<{}: {:?}>", stringify!($field), self.$field)?;)?
                Ok(())
            }
        }

        impl Operation for $ty {
            fn name(&self) -> &'static str {
                $name
            }
            fn dialect(&self) -> &'static dyn Dialect {
                $crate::ops::array_dialect()
            }
            fn check_expr(&self, $eqn: &Eqn) -> Result<()> {
                let $operation = self;
                $check
            }
            $($($extra)*)?
        }
    };
}

pub mod reduction;

array_operation!(AddOperation, "add", |eqn| arity(eqn, 2));
array_operation!(MulOperation, "mul", |eqn| arity(eqn, 2));
array_operation!(ZerosOperation, "zeros", |eqn| arity(eqn, 0));
array_operation!(OnesOperation, "ones", |eqn| arity(eqn, 0));
array_operation!(ReshapeOperation, "reshape", |eqn| arity(eqn, 1));
array_operation!(EyeOperation, "eye", |eqn| {
    arity(eqn, 0)?;
    if result_rank(eqn)? != 2 {
        return Err(signature("eye produces a matrix"));
    }
    Ok(())
});
array_operation!(SelectOperation, "select", |eqn| arity(eqn, 3));
array_operation!(
    BroadcastInDimOperation,
    "broadcast_in_dim",
    |op, eqn| {
        arity(eqn, 1)?;
        strictly_increasing(op.dims(), result_rank(eqn)?)
    },
    dims
);
array_operation!(
    TransposeOperation,
    "transpose",
    |op, eqn| {
        arity(eqn, 1)?;
        let mut perm = op.perm().to_vec();
        perm.sort_unstable();
        if !perm.iter().copied().eq(0..result_rank(eqn)?) {
            return Err(signature(
                "transpose needs a permutation of the result axes",
            ));
        }
        Ok(())
    },
    perm
);
array_operation!(
    RevOperation,
    "rev",
    |op, eqn| {
        arity(eqn, 1)?;
        strictly_increasing(op.axes(), result_rank(eqn)?)
    },
    axes
);

/// Elementwise addition (logical or for booleans).
pub fn add_op() -> Op {
    Op::new(AddOperation)
}

/// Elementwise multiplication (logical and for booleans).
pub fn mul_op() -> Op {
    Op::new(MulOperation)
}

/// An array of zeros with the result's metadata.
pub fn zeros_op() -> Op {
    Op::new(ZerosOperation)
}

/// An array of ones with the result's metadata.
pub fn ones_op() -> Op {
    Op::new(OnesOperation)
}

/// The identity matrix with the result's (square) metadata.
pub fn eye_op() -> Op {
    Op::new(EyeOperation)
}

/// Reinterpret the operand's elements, in row-major order, with the result's
/// shape. The element counts must agree.
pub fn reshape_op() -> Op {
    Op::new(ReshapeOperation)
}

/// `select(pred, on_true, on_false)` with a boolean `pred` of the same shape.
pub fn select_op() -> Op {
    Op::new(SelectOperation)
}

/// Broadcast operand axis `i` to result axis `dims[i]`; `dims` is strictly
/// increasing and each operand axis has size 1 or the result's size.
pub fn broadcast_in_dim_op(dims: &[usize]) -> Op {
    Op::new(BroadcastInDimOperation {
        dims: dims.to_vec(),
    })
}

/// Result axis `i` is operand axis `perm[i]`.
pub fn transpose_op(perm: &[usize]) -> Op {
    Op::new(TransposeOperation {
        perm: perm.to_vec(),
    })
}

/// Reverse the element order along strictly increasing `axes`.
pub fn rev_op(axes: &[usize]) -> Op {
    Op::new(RevOperation {
        axes: axes.to_vec(),
    })
}

/// Constructors for the array dialect in expression macros.
pub mod array {
    pub use super::{
        add_op as add, broadcast_in_dim_op as broadcast_in_dim, eye_op as eye, mul_op as mul,
        ones_op as ones, reshape_op as reshape, rev_op as rev, select_op as select,
        transpose_op as transpose, zeros_op as zeros,
    };
}

//! The `array` dialect: elementwise and structural array operations.
//!
//! Operands of binary elementwise operations must have identical shapes and
//! dtypes; there is no implicit broadcasting. Structural operations carry their
//! index parameters as typed fields and take result metadata from the
//! equation.
//!
//! An operation without parameters is stored as it is. One with parameters
//! is made from a builder (`BroadcastInDim { .. }`) that is stored as the
//! operation (`BroadcastInDimOperation`).

use gnx_expr::{Eqn, LoweringError, Op};
use rkyv::{Archive, Serialize, rend::u32_le, vec::ArchivedVec};

use crate::ArrayInfo;

/// Defines operations without parameters.
#[doc(hidden)]
#[macro_export]
macro_rules! unit_operations {
    ($($(#[$meta:meta])* $vis:vis struct $ty:ident;)*) => {$(
        $(#[$meta])*
        #[derive($crate::rkyv::Archive, $crate::rkyv::Serialize, $crate::rkyv::Portable)]
        #[rkyv(crate = $crate::rkyv, as = Self)]
        #[repr(C)]
        $vis struct $ty;
    )*};
}

unit_operations! {
    pub struct AddOperation;
    pub struct MulOperation;
    pub struct SubOperation;
    pub struct DivOperation;
    pub struct RemOperation;
    pub struct ZerosOperation;
    pub struct OnesOperation;
    pub struct EyeOperation;
    pub struct ReshapeOperation;
    pub struct SelectOperation;
    pub struct ConvertElementTypeOperation;
}

/// The indices stored in an operation, such as axes.
#[doc(hidden)]
pub fn indices(stored: &ArchivedVec<u32_le>) -> Vec<usize> {
    stored.iter().map(|i| i.to_native() as usize).collect()
}

/// Indices in the form they are stored in an operation.
#[doc(hidden)]
pub fn stored_indices(indices: &[usize]) -> Vec<u32> {
    (indices.iter())
        .map(|&i| u32::try_from(i).expect("index exceeds u32::MAX"))
        .collect()
}

#[derive(Archive, Serialize)]
#[rkyv(archived = BroadcastInDimOperation)]
pub struct BroadcastInDim {
    dims: Vec<u32>,
}

impl BroadcastInDimOperation {
    pub fn dims(&self) -> Vec<usize> {
        indices(&self.dims)
    }
}

#[derive(Archive, Serialize)]
#[rkyv(archived = TransposeOperation)]
pub struct Transpose {
    perm: Vec<u32>,
}

impl TransposeOperation {
    pub fn perm(&self) -> Vec<usize> {
        indices(&self.perm)
    }
}

#[derive(Archive, Serialize)]
#[rkyv(archived = RevOperation)]
pub struct Rev {
    axes: Vec<u32>,
}

impl RevOperation {
    pub fn axes(&self) -> Vec<usize> {
        indices(&self.axes)
    }
}

/// The name of the array dialect.
#[doc(hidden)]
pub fn array_dialect() -> &'static str {
    "array"
}

type Result<T> = std::result::Result<T, LoweringError>;

#[doc(hidden)]
pub fn signature(message: &'static str) -> LoweringError {
    LoweringError::RegionSignature(message)
}

#[doc(hidden)]
pub fn arity(eqn: Eqn<'_>, inputs: usize) -> Result<()> {
    if eqn.inputs().len() != inputs || eqn.outputs().len() != 1 || eqn.subexprs().len() != 0 {
        return Err(signature("array operation signature"));
    }
    Ok(())
}

#[doc(hidden)]
pub fn result_rank(eqn: Eqn<'_>) -> Result<usize> {
    let output = (eqn.outputs().next()).ok_or_else(|| signature("array operation signature"))?;
    Ok(ArrayInfo::from_info(&output.info_ref().to_owned())
        .ok_or_else(|| signature("array operations produce arrays"))?
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

/// Implements `Operation` for a type, as the operation `$name` of the array
/// dialect. `$field` names an accessor of a parameter to show.
#[doc(hidden)]
#[macro_export]
macro_rules! array_operation {
    ($ty:ty, $name:literal, |$eqn:ident| $check:expr) => {
        $crate::array_operation!($ty, $name, |_op, $eqn| $check);
    };
    (
        $ty:ty, $name:literal, |$operation:ident, $eqn:ident| $check:expr
        $(, $field:ident)? $(; $($extra:item)*)?
    ) => {
        impl $crate::gnx_expr::Operation for $ty {
            fn display(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                f.write_str(concat!("array::", $name))?;
                $(write!(f, "<{}: {:?}>", stringify!($field), self.$field())?;)?
                Ok(())
            }
            fn check_expr(
                &self,
                $eqn: $crate::gnx_expr::Eqn<'_>,
            ) -> ::std::result::Result<(), $crate::gnx_expr::LoweringError> {
                let $operation = self;
                $check
            }
            $($($extra)*)?
        }

        $crate::gnx_expr::dialect! {
            dyn $crate::gnx_expr::Operation, "array" => { $name => $ty }
        }
    };
}

pub mod creation;
pub mod indexing;
pub mod reduction;

array_operation!(ConvertElementTypeOperation, "convert_element_type", |eqn| {
    arity(eqn, 1)
});
pub fn convert_element_type_op() -> Op {
    Op::new(ConvertElementTypeOperation)
}
array_operation!(AddOperation, "add", |eqn| arity(eqn, 2));
array_operation!(MulOperation, "mul", |eqn| arity(eqn, 2));
array_operation!(SubOperation, "sub", |eqn| arity(eqn, 2));
array_operation!(DivOperation, "div", |eqn| arity(eqn, 2));
array_operation!(RemOperation, "rem", |eqn| arity(eqn, 2));
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
        strictly_increasing(&op.dims(), result_rank(eqn)?)
    },
    dims
);
array_operation!(
    TransposeOperation,
    "transpose",
    |op, eqn| {
        arity(eqn, 1)?;
        let mut perm = op.perm();
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
        strictly_increasing(&op.axes(), result_rank(eqn)?)
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

pub fn sub_op() -> Op {
    Op::new(SubOperation)
}
pub fn div_op() -> Op {
    Op::new(DivOperation)
}
pub fn rem_op() -> Op {
    Op::new(RemOperation)
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
    Op::new(BroadcastInDim {
        dims: stored_indices(dims),
    })
}

/// Result axis `i` is operand axis `perm[i]`.
pub fn transpose_op(perm: &[usize]) -> Op {
    Op::new(Transpose {
        perm: stored_indices(perm),
    })
}

/// Reverse the element order along strictly increasing `axes`.
pub fn rev_op(axes: &[usize]) -> Op {
    Op::new(Rev {
        axes: stored_indices(axes),
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

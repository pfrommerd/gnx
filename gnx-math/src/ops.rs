//! The `array` dialect: elementwise and structural array operations.
//!
//! Operands of binary elementwise operations must have identical shapes and
//! dtypes; there is no implicit broadcasting. Structural operations carry their
//! index parameters as `I64` list attributes and take result metadata from the
//! equation.

use gnx_expr::{Attr, Dialect, Eqn, LoweringError, Op, Operation};

use crate::{ArrayInfo, Dim};

pub struct ArrayDialect;
pub struct AddOperation;
pub struct MulOperation;
pub struct ZerosOperation;
pub struct OnesOperation;
pub struct EyeOperation;
pub struct ReshapeOperation;
pub struct BroadcastInDimOperation;
pub struct ReduceSumOperation;
pub struct TransposeOperation;
pub struct RevOperation;
pub struct SelectOperation;

static ARRAY: ArrayDialect = ArrayDialect;
static ADD: AddOperation = AddOperation;

gnx_expr::register_dialect!(&ARRAY);

type Result<T> = std::result::Result<T, LoweringError>;

fn signature(message: &'static str) -> LoweringError {
    LoweringError::RegionSignature(message)
}

impl Dialect for ArrayDialect {
    fn name(&self) -> &'static str {
        "array"
    }

    fn create(&self, name: &'static str) -> Option<&'static dyn Operation> {
        Some(match name {
            "add" => &ADD,
            "mul" => &MulOperation,
            "zeros" => &ZerosOperation,
            "ones" => &OnesOperation,
            "eye" => &EyeOperation,
            "reshape" => &ReshapeOperation,
            "broadcast_in_dim" => &BroadcastInDimOperation,
            "reduce_sum" => &ReduceSumOperation,
            "transpose" => &TransposeOperation,
            "rev" => &RevOperation,
            "select" => &SelectOperation,
            _ => return None,
        })
    }
}

/// The `I64` list attribute `key`, as non-negative indices.
pub fn index_list(op: &Op, key: &str) -> Result<Vec<usize>> {
    let Some(Attr::List(items)) = op.attrs().get(key) else {
        return Err(signature("missing index list attribute"));
    };
    items
        .iter()
        .map(|item| match item {
            Attr::I64(value) => {
                usize::try_from(*value).map_err(|_| signature("negative index attribute"))
            }
            _ => Err(signature("expected an i64 index")),
        })
        .collect()
}

pub fn index_attr(values: &[usize]) -> Attr {
    Attr::List(Box::new(
        values
            .iter()
            .map(|&value| Attr::I64(i64::try_from(value).expect("index fits in i64")))
            .collect(),
    ))
}

fn with_indices(op: &'static dyn Operation, key: &str, values: &[usize]) -> Op {
    Op::new_with_attrs(op, [(key.to_owned(), index_attr(values))].into())
}

fn arity(eqn: &Eqn, inputs: usize, attrs: &[&str]) -> Result<()> {
    if eqn.inputs().len() != inputs
        || eqn.outputs().len() != 1
        || !eqn.subexprs().is_empty()
        || eqn.op().attrs().len() != attrs.len()
        || attrs.iter().any(|key| !eqn.op().attrs().contains_key(*key))
    {
        return Err(signature("array operation signature"));
    }
    Ok(())
}

fn result_rank(eqn: &Eqn) -> Result<usize> {
    let (_, info) = &eqn.outputs()[0];
    Ok(info
        .downcast::<ArrayInfo>()
        .map_err(|_| signature("array operations produce arrays"))?
        .shape()
        .dims()
        .len())
}

fn strictly_increasing(indices: &[usize], bound: usize) -> Result<()> {
    if indices.windows(2).any(|pair| pair[0] >= pair[1]) || indices.iter().any(|&i| i >= bound) {
        return Err(signature(
            "indices must be strictly increasing and in range",
        ));
    }
    Ok(())
}

macro_rules! operation {
    ($ty:ty, $name:literal, |$eqn:ident| $check:expr) => {
        impl Operation for $ty {
            fn name(&self) -> &'static str {
                $name
            }
            fn dialect(&self) -> &'static dyn Dialect {
                &ARRAY
            }
            fn check_expr(&self, $eqn: &Eqn) -> Result<()> {
                $check
            }
        }
    };
}

operation!(AddOperation, "add", |eqn| arity(eqn, 2, &[]));
operation!(MulOperation, "mul", |eqn| arity(eqn, 2, &[]));
operation!(ZerosOperation, "zeros", |eqn| arity(eqn, 0, &[]));
operation!(OnesOperation, "ones", |eqn| arity(eqn, 0, &[]));
operation!(ReshapeOperation, "reshape", |eqn| arity(eqn, 1, &[]));
operation!(EyeOperation, "eye", |eqn| {
    arity(eqn, 0, &[])?;
    if result_rank(eqn)? != 2 {
        return Err(signature("eye produces a matrix"));
    }
    Ok(())
});
operation!(SelectOperation, "select", |eqn| arity(eqn, 3, &[]));
operation!(BroadcastInDimOperation, "broadcast_in_dim", |eqn| {
    arity(eqn, 1, &["dims"])?;
    strictly_increasing(&index_list(eqn.op(), "dims")?, result_rank(eqn)?)
});
operation!(ReduceSumOperation, "reduce_sum", |eqn| {
    arity(eqn, 1, &["axes"])?;
    let axes = index_list(eqn.op(), "axes")?;
    strictly_increasing(&axes, result_rank(eqn)? + axes.len())
});
operation!(TransposeOperation, "transpose", |eqn| {
    arity(eqn, 1, &["perm"])?;
    let mut perm = index_list(eqn.op(), "perm")?;
    perm.sort_unstable();
    if !perm.iter().copied().eq(0..result_rank(eqn)?) {
        return Err(signature(
            "transpose needs a permutation of the result axes",
        ));
    }
    Ok(())
});
operation!(RevOperation, "rev", |eqn| {
    arity(eqn, 1, &["axes"])?;
    strictly_increasing(&index_list(eqn.op(), "axes")?, result_rank(eqn)?)
});

/// Elementwise addition (logical or for booleans).
pub fn add_op() -> Op {
    Op::new(&ADD)
}

/// Elementwise multiplication (logical and for booleans).
pub fn mul_op() -> Op {
    Op::new(&MulOperation)
}

/// An array of zeros with the result's metadata.
pub fn zeros_op() -> Op {
    Op::new(&ZerosOperation)
}

/// An array of ones with the result's metadata.
pub fn ones_op() -> Op {
    Op::new(&OnesOperation)
}

/// The identity matrix with the result's (square) metadata.
pub fn eye_op() -> Op {
    Op::new(&EyeOperation)
}

/// Reinterpret the operand's elements, in row-major order, with the result's
/// shape. The element counts must agree.
pub fn reshape_op() -> Op {
    Op::new(&ReshapeOperation)
}

/// `select(pred, on_true, on_false)` with a boolean `pred` of the same shape.
pub fn select_op() -> Op {
    Op::new(&SelectOperation)
}

/// Broadcast operand axis `i` to result axis `dims[i]`; `dims` is strictly
/// increasing and each operand axis has size 1 or the result's size.
pub fn broadcast_in_dim_op(dims: &[usize]) -> Op {
    with_indices(&BroadcastInDimOperation, "dims", dims)
}

/// Sum (logical or for booleans) over strictly increasing `axes`, removing them.
pub fn reduce_sum_op(axes: &[usize]) -> Op {
    with_indices(&ReduceSumOperation, "axes", axes)
}

/// Result axis `i` is operand axis `perm[i]`.
pub fn transpose_op(perm: &[usize]) -> Op {
    with_indices(&TransposeOperation, "perm", perm)
}

/// Reverse the element order along strictly increasing `axes`.
pub fn rev_op(axes: &[usize]) -> Op {
    with_indices(&RevOperation, "axes", axes)
}

/// Fixed dimensions of an array, as required by every structural operation.
pub fn fixed_dims(info: &ArrayInfo) -> Result<Vec<usize>> {
    info.shape()
        .dims()
        .iter()
        .map(|dim| match dim {
            Dim::Fixed(size) => Ok(*size),
            _ => Err(signature("array operation requires fixed dimensions")),
        })
        .collect()
}

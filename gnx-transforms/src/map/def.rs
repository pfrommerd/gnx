//! Declarative batching rules, in the style of JAX's `batching.defbroadcasting`
//! and per-primitive batchers.
//!
//! ```ignore
//! // Elementwise: operands share a batch axis, or are aligned at axis 0.
//! defbroadcasting!(AddOperation, MulOperation, SelectOperation);
//!
//! // Batch along a new leading axis: every operand is moved (or broadcast) to
//! // axis 0, and the closure returns the operation to apply to them.
//! defbatch!(ReduceSumOperation, |op| Ok(reduce_sum_op(&shifted(op.axes()))));
//! ```
//!
//! Operations without operands (`zeros`, `ones`, `eye`, ...) need no rule: an
//! equation whose operands and captures are all unbatched is copied as is, and
//! its unbatched result is shared by every instance. Operations that own
//! regions implement [`BatchRule`](super::BatchRule) directly.

use gnx_expr::pass::Builder;
use gnx_expr::{Eqn, Op, ValueInfo};

use super::batch::{Batched, materialize};
use crate::util::with_axis;
use crate::{TransformError, TransformResult};

#[doc(inline)]
pub use crate::{defbatch, defbroadcasting};

/// Shift indices past a new leading axis. For example, `[0, 2] -> [1, 3]`.
pub fn shifted(indices: &[usize]) -> Vec<usize> {
    indices.iter().map(|&i| i + 1).collect()
}

/// An axis mapping or permutation with a new leading axis that maps to itself:
/// `[1, 0] -> [0, 2, 1]`.
pub fn leading(indices: &[usize]) -> Vec<usize> {
    std::iter::once(0).chain(shifted(indices)).collect()
}

fn single_result(eqn: &Eqn) -> TransformResult<&ValueInfo> {
    match eqn.outputs() {
        [(_, info)] => Ok(info),
        _ => Err(TransformError::invalid(
            "declarative batching rules need exactly one result",
        )),
    }
}

/// Batching rule body for `defbroadcasting!`.
pub fn broadcasting(
    b: &mut Builder,
    size: usize,
    eqn: &Eqn,
    inputs: &[Batched],
) -> TransformResult<Vec<Batched>> {
    let info = single_result(eqn)?;
    let axis = inputs
        .iter()
        .find_map(|value| value.axis)
        .expect("at least one batched operand");
    let (axis, vars) = if inputs.iter().all(|value| value.axis == Some(axis)) {
        (axis, inputs.iter().map(|value| value.var).collect())
    } else {
        let vars = inputs
            .iter()
            .map(|&value| materialize(b, value, size, 0))
            .collect::<TransformResult<Vec<_>>>()?;
        (0, vars)
    };
    let info = with_axis(info, axis, size)?;
    Ok(vec![Batched::at(b.emit1(eqn.op().clone(), &vars, info), axis)])
}

/// Batching rule body for `defbatch!`: apply `op` to the operands with their
/// batch axes moved to 0.
pub fn at_leading_axis(
    b: &mut Builder,
    size: usize,
    eqn: &Eqn,
    inputs: &[Batched],
    op: Op,
) -> TransformResult<Vec<Batched>> {
    let info = with_axis(single_result(eqn)?, 0, size)?;
    let vars = inputs
        .iter()
        .map(|&value| materialize(b, value, size, 0))
        .collect::<TransformResult<Vec<_>>>()?;
    Ok(vec![Batched::at(b.emit1(op, &vars, info), 0)])
}

/// Fix a `defbatch!` closure's signature (for type inference).
#[doc(hidden)]
pub fn batch_op_rule<O, F: Fn(&O) -> TransformResult<Op>>(f: F) -> F {
    f
}

/// Batch elementwise operations: if every operand is batched along the same
/// axis the operation is applied as is; otherwise operands are aligned at
/// axis 0 (unbatched ones broadcast).
///
/// ```ignore
/// defbroadcasting!(AddOperation, MulOperation, SelectOperation);
/// ```
#[macro_export]
macro_rules! defbroadcasting {
    ($($op:ty),+ $(,)?) => {$(
        impl $crate::map::BatchRule for $op {
            fn batch(
                &self,
                b: &mut $crate::__private::gnx_expr::pass::Builder,
                size: usize,
                eqn: &$crate::__private::gnx_expr::Eqn,
                inputs: &[$crate::map::Batched],
                _captures: &[::std::vec::Vec<$crate::map::Batched>],
            ) -> $crate::TransformResult<::std::vec::Vec<$crate::map::Batched>> {
                $crate::map::def::broadcasting(b, size, eqn, inputs)
            }
        }
        $crate::__private::gnx_expr::register_op_interface!(dyn $crate::map::BatchRule: $op);
    )+};
}

/// Batch an operation along a new leading axis: every operand is moved (or
/// broadcast) to axis 0, and the closure maps the original operation to the
/// one applied to the batched operands, typically by shifting its index
/// parameters with [`shifted`] or [`leading`].
///
/// ```ignore
/// defbatch!(TransposeOperation, |op| Ok(transpose_op(&leading(op.perm()))));
/// ```
#[macro_export]
macro_rules! defbatch {
    ($op:ty, $rule:expr $(,)?) => {
        impl $crate::map::BatchRule for $op {
            fn batch(
                &self,
                b: &mut $crate::__private::gnx_expr::pass::Builder,
                size: usize,
                eqn: &$crate::__private::gnx_expr::Eqn,
                inputs: &[$crate::map::Batched],
                _captures: &[::std::vec::Vec<$crate::map::Batched>],
            ) -> $crate::TransformResult<::std::vec::Vec<$crate::map::Batched>> {
                let op = $crate::map::def::batch_op_rule::<$op, _>($rule)(self)?;
                $crate::map::def::at_leading_axis(b, size, eqn, inputs, op)
            }
        }
        $crate::__private::gnx_expr::register_op_interface!(dyn $crate::map::BatchRule: $op);
    };
}

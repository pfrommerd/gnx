//! NumPy-style functions on lazy GNX arrays. Arithmetic uses the same broadcasting
//! and matching-dtype rules as the Rust array operators.
//!
//! ```
//! use gnx_math::{Array, ArrayInfo, DType, Shape, np};
//! let arg = |shape| Array::placeholder(ArrayInfo::new(shape, DType::F32));
//! let x = arg(Shape::fixed([2, 3]));
//! let bias = arg(Shape::fixed([3]));
//! let score = np::sum(&np::tanh(&np::add(&x, &bias)), -1);
//! assert_eq!(score.shape(), Shape::fixed([2]));
//! ```

use crate::{Array, lax};
use gnx_base::ops::signature;
use gnx_expr::LoweringError;

pub use lax::{
    abs, concatenate, convert_element_type as astype, cos, exp, log1p, neg as negative, scalar,
    sin, tanh, try_abs, try_concatenate, try_cos, try_exp, try_log1p, try_neg as try_negative,
    try_sin, try_tanh, zeros,
};

/// Add with right-aligned broadcasting; no implicit dtype promotion.
pub fn try_add(x: &Array, y: &Array) -> Result<Array, LoweringError> {
    x.try_add(y)
}

/// Add with right-aligned broadcasting; panics on incompatible operands.
#[track_caller]
pub fn add(x: &Array, y: &Array) -> Array {
    x + y
}

/// Reduction axes. Negative indices count from the end; an empty list is an identity.
pub enum Axes {
    All,
    List(Vec<isize>),
}
/// Reduce every axis, including for rank-zero arrays.
pub const ALL: Axes = Axes::All;
impl From<isize> for Axes {
    fn from(axis: isize) -> Self {
        Self::List(vec![axis])
    }
}
impl<const N: usize> From<[isize; N]> for Axes {
    fn from(axes: [isize; N]) -> Self {
        Self::List(axes.into())
    }
}
impl From<&[isize]> for Axes {
    fn from(axes: &[isize]) -> Self {
        Self::List(axes.into())
    }
}
impl<const N: usize> From<&[isize; N]> for Axes {
    fn from(axes: &[isize; N]) -> Self {
        Self::from(axes.as_slice())
    }
}
impl From<Vec<isize>> for Axes {
    fn from(axes: Vec<isize>) -> Self {
        Self::List(axes)
    }
}

/// Sum over a signed axis, a list of axes, or [`ALL`]. Keeps the input dtype;
/// for boolean arrays this computes logical OR. Duplicate axes are invalid.
pub fn try_sum(x: &Array, axes: impl Into<Axes>) -> Result<Array, LoweringError> {
    let rank = x.shape().dims().len();
    let axes = match axes.into() {
        Axes::All => (0..rank).collect(),
        Axes::List(list) => {
            let mut axes = Vec::with_capacity(list.len());
            for axis in list {
                let i = if axis < 0 {
                    rank.checked_sub(axis.unsigned_abs())
                } else {
                    Some(axis as usize)
                };
                axes.push(
                    i.filter(|&i| i < rank)
                        .ok_or_else(|| signature("sum axis out of range"))?,
                );
            }
            axes.sort_unstable();
            axes
        }
    };
    lax::try_reduce_sum(x, &axes)
}

/// See [`try_sum`]. Panics on invalid axes or unsupported input.
#[track_caller]
pub fn sum(x: &Array, axes: impl Into<Axes>) -> Array {
    try_sum(x, axes).expect("np::sum")
}

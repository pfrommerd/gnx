//! Explicit, lazy array primitives. Operands are borrowed and results are traced.
//! Binary arithmetic requires matching shapes and dtypes; use [`crate::np`] for broadcasting.

use crate::{Array, ArrayInfo, DType, Dim, IntoShape, Shape, ops};
use gnx_common::{
    invoke_array,
    ops::{signature, strictly_increasing},
};
use gnx_expr::LoweringError;

type Result<T> = std::result::Result<T, LoweringError>;

// Keep the checked implementation and its panicking convenience function together.
macro_rules! primitive {
    ($(#[$doc:meta])* $name:ident, $checked:ident ($($arg:ident: $ty:ty),*) -> $out:ty) => {
        $(#[$doc])*
        #[track_caller]
        pub fn $name($($arg: $ty),*) -> $out {
            $checked($($arg),*).expect(concat!("lax::", stringify!($name)))
        }
    };
    ($(#[$doc:meta])* $name:ident, $checked:ident ($($arg:ident: $ty:ty),*) -> $out:ty $body:block) => {
        primitive!($(#[$doc])* $name, $checked($($arg: $ty),*) -> $out);
        #[doc = concat!("Checked variant of [`", stringify!($name), "`].")]
        pub fn $checked($($arg: $ty),*) -> Result<$out> $body
    };
}
mod elementwise;
mod structural;
pub use elementwise::*;
pub use structural::*;

/// Add arrays with identical metadata.
pub fn try_add(x: &Array, y: &Array) -> Result<Array> {
    if x.shape() != y.shape() {
        return Err(signature("lax::add requires matching shapes"));
    }
    x.try_add(y)
}

/// Add arrays with identical metadata, panicking on incompatible operands.
#[track_caller]
pub fn add(x: &Array, y: &Array) -> Array {
    try_add(x, y).expect("lax::add")
}

/// Broadcast each input axis to its specified output axis, in increasing order.
pub fn try_broadcast_in_dim(x: &Array, shape: impl IntoShape, axes: &[usize]) -> Result<Array> {
    let shape = shape.into_shape();
    let input = x.shape();
    if axes.len() != input.dims().len() {
        return Err(signature("broadcast needs one output axis per input axis"));
    }
    strictly_increasing(axes, shape.dims().len())?;
    if input == shape {
        return Ok(x.clone());
    }
    if !shape.dims().iter().all(|d| matches!(d, Dim::Fixed(_))) {
        return Err(signature(
            "broadcasting to a different shape requires fixed output dimensions",
        ));
    }
    for (dim, &axis) in input.dims().iter().zip(axes) {
        if *dim != Dim::Fixed(1) && *dim != shape.dims()[axis] {
            return Err(signature("incompatible mapped broadcast dimensions"));
        }
    }
    Ok(invoke_array(
        ops::shape::broadcast_in_dim(axes),
        &[x],
        &ArrayInfo::new(shape, x.dtype()),
    ))
}

/// Infallible-call counterpart of [`try_broadcast_in_dim`]; panics on invalid dimensions.
#[track_caller]
pub fn broadcast_in_dim(x: &Array, shape: impl IntoShape, axes: &[usize]) -> Array {
    try_broadcast_in_dim(x, shape, axes).expect("lax::broadcast_in_dim")
}

/// Elementwise hyperbolic tangent, preserving shape and dtype.
pub fn try_tanh(x: &Array) -> Result<Array> {
    use DType::*;
    if !matches!(
        x.dtype(),
        BF16 | F8E4M3FN | F8E5M2 | F8E8M0FNU | F16 | F32 | F64 | C64 | C128
    ) {
        return Err(signature("tanh requires floating-point or complex input"));
    }
    Ok(invoke_array(
        ops::elementwise::tanh(),
        &[x],
        &ArrayInfo::new(x.shape(), x.dtype()),
    ))
}

/// Elementwise hyperbolic tangent; panics on non-floating, non-complex input.
#[track_caller]
pub fn tanh(x: &Array) -> Array {
    try_tanh(x).expect("lax::tanh")
}

/// Sum over sorted, distinct axes, preserving the dtype (boolean sum is OR).
pub fn try_reduce_sum(x: &Array, axes: &[usize]) -> Result<Array> {
    if x.dtype().is_key() {
        return Err(signature("cannot sum PRNG keys"));
    }
    let input = x.shape();
    strictly_increasing(axes, input.dims().len())?;
    if axes.is_empty() {
        return Ok(x.clone());
    }
    if input.dims().iter().any(|d| matches!(d, Dim::Jagged(_))) {
        return Err(signature("reduce_sum does not support jagged dimensions"));
    }
    let shape: Shape = input
        .dims()
        .iter()
        .enumerate()
        .filter_map(|(i, &d)| axes.binary_search(&i).is_err().then_some(d))
        .collect();
    Ok(invoke_array(
        ops::reduction::reduce_add_op(axes),
        &[x],
        &ArrayInfo::new(shape, x.dtype()),
    ))
}

/// Sum over sorted, distinct axes; panics on invalid input.
#[track_caller]
pub fn reduce_sum(x: &Array, axes: &[usize]) -> Array {
    try_reduce_sum(x, axes).expect("lax::reduce_sum")
}

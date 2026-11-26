//! Decoder formulas built from the public array APIs.
use gnx::{Array, array::DType, lax, np};

pub fn dims(x: &Array) -> Vec<usize> {
    x.fixed_dims().expect("fixed checkpoint shapes")
}
pub use gnx::nn::activations::silu;
pub fn keep_sum(x: &Array) -> Array {
    let mut s = dims(x);
    *s.last_mut().expect("keep_sum needs a trailing axis") = 1;
    np::sum(x, -1).reshape(s)
}
pub fn norm(x: &Array, divisor: f64, eps: f64) -> Array {
    let sq = keep_sum(&(x * x));
    x * lax::rsqrt(&(sq / divisor + eps))
}
pub fn rms(x: &Array, w: &Array, eps: f64) -> Array {
    norm(x, *dims(x).last().unwrap() as f64, eps) * (w.astype(DType::F32) + 1.0)
}
/// Decoder contractions use leading batch axes and one contracting axis per operand.
pub fn dot(a: &Array, b: &Array, ac: usize, bc: usize, batch: usize) -> Array {
    lax::dot_general(
        a,
        b,
        lax::DotDimensions {
            lhs_contracting: vec![ac],
            rhs_contracting: vec![bc],
            lhs_batch: (0..batch).collect(),
            rhs_batch: (0..batch).collect(),
        },
    )
}
/// Trace the cast and projection together, including when called inside the jitted MLP.
#[gnx::jit]
pub fn linear(x: &Array, w: &Array) -> Array {
    dot(&x.astype(w.dtype()), w, 0, 1, 0).astype(DType::F32)
}

//! Decoder formulas built from the public array APIs.
use gnx::{
    Array,
    array::{DType, Dim},
    lax, np,
};

pub fn dims(x: &Array) -> Vec<usize> {
    x.shape()
        .dims()
        .iter()
        .map(|d| match d {
            Dim::Fixed(n) => *n,
            _ => unreachable!("fixed checkpoint shapes"),
        })
        .collect()
}
pub fn scalar(x: f64) -> Array {
    np::scalar(x, DType::F32)
}
pub fn silu(x: &Array) -> Array {
    x * lax::logistic(x)
}
pub fn keep_sum(x: &Array) -> Array {
    let mut s = dims(x);
    *s.last_mut().expect("keep_sum needs a trailing axis") = 1;
    np::sum(x, -1).reshape(s)
}
pub fn norm(x: &Array, divisor: f64, eps: f64) -> Array {
    let sq = keep_sum(&(x * x));
    x * lax::rsqrt(&(sq / scalar(divisor) + scalar(eps)))
}
pub fn rms(x: &Array, w: &Array, eps: f64) -> Array {
    norm(x, *dims(x).last().unwrap() as f64, eps) * (np::astype(w, DType::F32) + scalar(1.))
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
    np::astype(&dot(&np::astype(x, w.dtype()), w, 0, 1, 0), DType::F32)
}

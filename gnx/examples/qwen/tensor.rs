//! Small fixed-shape tensor vocabulary for the decoder. All math builds GNX expressions.
use gnx::ops::contraction::{DotDimensions, dot_general};
use gnx::{
    Array, Shape,
    array::{ArrayInfo, DType, Dim},
    expr::{Invocation, Op, ValueInfo, hexpr},
    ops::{self, elementwise as e},
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
fn info(s: &[usize], t: DType) -> ValueInfo {
    ValueInfo::new(ArrayInfo::new(Shape::fixed(s.iter().copied()), t))
}
pub fn apply(op: Op, xs: &[&Array], s: &[usize], t: DType) -> Array {
    Invocation::invoke(
        op,
        vec![],
        xs.iter().map(|x| x.tracer().clone()).collect(),
        vec![info(s, t)],
    )
    .remove(0)
    .try_cast::<Array>()
    .ok()
    .unwrap()
    .into()
}
pub fn cast(x: &Array, t: DType) -> Array {
    if t == x.dtype() {
        x.clone()
    } else {
        apply(ops::dtype::convert_element_type(), &[x], &dims(x), t)
    }
}
pub fn view(x: &Array, axis: usize, start: usize, end: usize) -> Array {
    let mut s = dims(x);
    let mut ranges: Vec<_> = s.iter().map(|&n| (0, n, 1)).collect();
    ranges[axis] = (start, end, 1);
    s[axis] = end - start;
    apply(ops::indexing::slice(&ranges), &[x], &s, x.dtype())
}
pub fn cat(a: &Array, b: &Array, axis: usize) -> Array {
    let mut s = dims(a);
    s[axis] += dims(b)[axis];
    apply(ops::shape::concatenate(axis), &[a, b], &s, a.dtype())
}
pub fn broadcast(x: &Array, s: &[usize], axes: &[usize]) -> Array {
    apply(ops::broadcast_in_dim_op(axes), &[x], s, x.dtype())
}
pub fn scalar(x: f64) -> Array {
    apply(ops::creation::constant(x), &[], &[], DType::F32)
}
pub fn zeros(s: &[usize]) -> Array {
    apply(ops::zeros_op(), &[], s, DType::F32)
}
pub fn unary(op: Op, x: &Array) -> Array {
    apply(op, &[x], &dims(x), x.dtype())
}
pub fn sigmoid(x: &Array) -> Array {
    unary(e::logistic(), x)
}
pub fn silu(x: &Array) -> Array {
    x * sigmoid(x)
}
pub fn sum(x: &Array, axis: usize) -> Array {
    let m = info(&[], DType::F32);
    let combiner = hexpr! { |x: &m, y: &m| => { z: &m = array::add x, y; in (z) } };
    let mut s = dims(x);
    s.remove(axis);
    Invocation::invoke(
        ops::reduction::reduce_op(&[axis]),
        vec![(combiner.into(), vec![])],
        vec![x.tracer().clone(), scalar(0.).into_tracer().into()],
        vec![info(&s, DType::F32)],
    )
    .remove(0)
    .try_cast::<Array>()
    .ok()
    .unwrap()
    .into()
}
pub fn keep_sum(x: &Array) -> Array {
    let mut s = dims(x);
    let axis = s.len() - 1;
    s[axis] = 1;
    sum(x, axis).reshape(s)
}
pub fn norm(x: &Array, divisor: f64, eps: f64) -> Array {
    let sq = keep_sum(&(x * x));
    x * unary(e::rsqrt(), &(sq / scalar(divisor) + scalar(eps)))
}
pub fn rms(x: &Array, w: &Array, eps: f64) -> Array {
    norm(x, *dims(x).last().unwrap() as f64, eps) * (cast(w, DType::F32) + scalar(1.))
}
pub fn dot(a: &Array, b: &Array, ac: usize, bc: usize, batch: usize) -> Array {
    let ad = dims(a);
    let bd = dims(b);
    let free = |s: &[usize], c| {
        s.iter()
            .enumerate()
            .filter_map(|(i, &n)| (i >= batch && i != c).then_some(n))
            .collect::<Vec<_>>()
    };
    let mut s = ad[..batch].to_vec();
    s.extend(free(&ad, ac));
    s.extend(free(&bd, bc));
    apply(
        dot_general(DotDimensions {
            lhs_contracting: vec![ac],
            rhs_contracting: vec![bc],
            lhs_batch: (0..batch).collect(),
            rhs_batch: (0..batch).collect(),
        }),
        &[a, b],
        &s,
        a.dtype(),
    )
}
/// Trace the cast and projection together, including when called inside the jitted MLP.
#[gnx::jit]
pub fn linear(x: &Array, w: &Array) -> Array {
    cast(&dot(&cast(x, w.dtype()), w, 0, 1, 0), DType::F32)
}
pub fn top1(x: &Array) -> (Array, Array) {
    let mut s = dims(x);
    *s.last_mut().unwrap() = 1;
    let mut out = Invocation::invoke(
        ops::ordering::top_k(1),
        vec![],
        vec![x.tracer().clone()],
        vec![info(&s, x.dtype()), info(&s, DType::I32)],
    )
    .into_iter()
    .map(|a| a.try_cast::<Array>().ok().unwrap().into());
    (out.next().unwrap(), out.next().unwrap())
}

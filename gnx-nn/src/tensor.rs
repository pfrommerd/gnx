//! Fixed-shape helpers shared by the layers.
use gnx_math::{Array, DType, lax};

pub(crate) fn dims(x: &Array) -> Vec<usize> {
    x.fixed_dims().expect("layer requires fixed dimensions")
}
pub(crate) fn work(t: DType) -> DType {
    match t {
        DType::BF16 | DType::F16 | DType::F32 => DType::F32,
        DType::F64 => t,
        _ => panic!("expected floating-point dtype"),
    }
}
pub(crate) fn constant(v: f64, shape: impl gnx_math::IntoShape, t: DType) -> Array {
    lax::broadcast_in_dim(&lax::scalar(v, t), shape, &[])
}
pub(crate) fn sum_keep(x: &Array, axes: &[usize]) -> Array {
    let mut d = dims(x);
    for &i in axes {
        d[i] = 1;
    }
    lax::reduce_sum(x, axes).reshape(d)
}
pub(crate) fn project(x: &Array, w: &Array) -> Array {
    lax::dot_general(
        x,
        w,
        lax::DotDimensions {
            lhs_batch: vec![],
            rhs_batch: vec![],
            lhs_contracting: vec![dims(x).len() - 1],
            rhs_contracting: vec![1],
        },
    )
}

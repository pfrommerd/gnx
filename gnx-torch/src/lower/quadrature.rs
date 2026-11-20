use super::{Kind, Tensor, tracing_device};
use tch::TchError;

// Standard eight-point Gauss-Legendre rule. Generate all nodes in the trace
// from static scalar coefficients, independently of example tensor values.
fn panels(edges: &Tensor) -> Result<(Tensor, Tensor), TchError> {
    let n = edges.size()[0] - 1;
    let lo = edges.f_narrow(0, 0, n)?;
    let hi = edges.f_narrow(0, 1, n)?;
    let mid = lo.f_add(&hi)?.f_mul_scalar(0.5)?;
    let half = hi.f_sub(&lo)?.f_mul_scalar(0.5)?;
    let mut nodes = Vec::new();
    let mut weights = Vec::new();
    for (z, w) in [
        (0.1834346424956498, 0.3626837833783620),
        (0.5255324099163290, 0.3137066458778873),
        (0.7966664774136267, 0.2223810344533745),
        (0.9602898564975363, 0.1012285362903763),
    ] {
        for sign in [-1., 1.] {
            nodes.push(mid.f_add(&half.f_mul_scalar(sign * z)?)?);
            weights.push(half.f_mul_scalar(w)?);
        }
    }
    Ok((Tensor::f_cat(&nodes, 0)?, Tensor::f_cat(&weights, 0)?))
}

pub(super) fn gamma_tail() -> Result<(Tensor, Tensor), TchError> {
    let edges = Tensor::f_arange(33, (Kind::Double, tracing_device()))?
        .f_div_scalar(8.)?
        .f_add_scalar(1.)?
        .f_pow_tensor_scalar(3.)?
        .f_sub_scalar(1.)?;
    panels(&edges)
}

pub(super) fn unit_interval() -> Result<(Tensor, Tensor), TchError> {
    let (u, w) =
        panels(&Tensor::f_arange(65, (Kind::Double, tracing_device()))?.f_div_scalar(64.)?)?;
    // Smooth fractional powers at zero with t=(8u)^8/8 on the first
    // eight panels. The remaining panels resolve localized interior mass.
    let left = u.f_lt(0.125)?;
    let scaled = u.f_mul_scalar(8.)?;
    Ok((
        scaled
            .f_pow_tensor_scalar(8.)?
            .f_mul_scalar(0.125)?
            .f_where_self(&left, &u)?,
        scaled
            .f_pow_tensor_scalar(7.)?
            .f_mul_scalar(8.)?
            .f_mul(&w)?
            .f_where_self(&left, &w)?,
    ))
}

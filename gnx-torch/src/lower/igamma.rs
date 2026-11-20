use super::{elementwise::same_operands, *};
use gnx_math::ops::special::IgammaGradAOperation;
use tch::TchError;

rule!(IgammaGradAOperation, |_op, _eqn, [a, x]| {
    same_operands(a, x)?;
    if !a.is_floating_point() {
        return Err(unsupported(
            "igamma_grad_a requires real floating-point operands",
        ));
    }
    shape_derivative(a, x)
});

type Native = Result<Tensor, TchError>;
fn sum_last(t: &Tensor) -> Native {
    t.f_sum_dim_intlist(&[-1i64][..], false, Kind::Double)
}
fn poly(x: &Tensor, coeffs: &[f64]) -> Native {
    let (last, rest) = coeffs.split_last().unwrap();
    rest.iter()
        .rev()
        .try_fold(x.f_full_like(*last)?, |v, c| v.f_mul(x)?.f_add_scalar(*c))
}

// Integrate the power series for exp(-t) term by term on [0,x]. Factoring
// out x^a/Gamma(a+1) removes the pole at a=0. For x<=1 the omitted tail is
// bounded by 1/25! before multiplication by the prefactor.
fn small_argument(a: &Tensor, x: &Tensor) -> Native {
    let k = Tensor::f_arange_start(1, 25, (Kind::Double, tracing_device()))?;
    let logx = x.f_log()?;
    let term = k
        .f_mul(&logx.f_unsqueeze(-1)?)?
        .f_sub(&k.f_add_scalar(1.)?.f_lgamma()?)?
        .f_exp()?
        .f_mul(&k.f_remainder(2)?.f_mul_scalar(-2.)?.f_add_scalar(1.)?)?;
    let denom = a.f_unsqueeze(-1)?.f_add(&k)?;
    // d[a/(a+k)]/da = k/(a+k)^2.
    let s = sum_last(&term.f_mul(&a.f_unsqueeze(-1)?.f_div(&denom)?)?)?.f_add_scalar(1.)?;
    let ds = sum_last(&term.f_mul(&k)?.f_div(&denom)?.f_div(&denom)?)?;
    let ap1 = a.f_add_scalar(1.)?;
    a.f_mul(&logx)?
        .f_sub(&ap1.f_lgamma()?)?
        .f_exp()?
        .f_mul(&ds.f_add(&s.f_mul(&logx.f_sub(&ap1.f_digamma()?)?)?)?)
}

// Integrate the gamma density times its shape score, log(t)-psi(a). Choose
// [0,x] left of the mode in log-space, or minus [x,infinity) on its right:
// this never subtracts two nearly equal CDFs. Substitute t=x*exp(v), with
// |v| scaled to resolve both the central peak and the endpoint of a far tail.
fn tail_integral(a: &Tensor, x: &Tensor) -> Native {
    let upper = x.f_ge_tensor(a)?;
    let scale = a.f_sub(x)?.f_abs()?.f_add(&x.f_sqrt()?)?.f_reciprocal()?;
    let (u, w) = super::quadrature::gamma_tail()?;
    let v = scale
        .f_where_self(&upper, &scale.f_neg()?)?
        .f_unsqueeze(-1)?
        .f_mul(&u)?;
    // expm1(v)-v starts at v^2/2; avoid subtractive cancellation for a
    // narrow peak. Multiply by x before squaring to retain tiny v^2 terms.
    let xp = x.f_unsqueeze(-1)?;
    let curvature = xp
        .f_mul(&v)?
        .f_mul(&v)?
        .f_mul(&poly(&v, &[0.5, 1. / 6., 1. / 24., 1. / 120., 1. / 720.])?)?
        .f_where_self(
            &v.f_abs()?.f_lt(0.001)?,
            &xp.f_mul(&v.f_expm1()?.f_sub(&v)?)?,
        )?;
    let relative = a
        .f_sub(x)?
        .f_unsqueeze(-1)?
        .f_mul(&v)?
        .f_sub(&curvature)?
        .f_exp()?;
    let logx = x.f_log()?;
    // For small shapes factor 1/Gamma(a)=a/Gamma(a+1) into the score,
    // keeping the derivative finite even when psi(a) would overflow.
    let ap1 = a.f_add_scalar(1.)?;
    let log_pref = a.f_mul(&logx)?.f_sub(x)?.f_sub(&ap1.f_lgamma()?)?;
    let score = logx
        .f_sub(&ap1.f_digamma()?)?
        .f_unsqueeze(-1)?
        .f_add(&v)?
        .f_mul(&a.f_unsqueeze(-1)?)?
        .f_add_scalar(1.)?;
    // For large shapes Stirling's expansion avoids subtracting O(a log a)
    // terms in log-density, and O(log a) terms in the score near x=a.
    let large = a.f_ge(16.)?;
    let d = x.f_sub(a)?.f_div(a)?;
    let ratio = d
        .f_log1p()?
        .f_where_self(&d.f_abs()?.f_lt(0.5)?, &logx.f_sub(&a.f_log()?)?)?;
    let rate = d
        .f_square()?
        .f_mul(&poly(&d, &[-0.5, 1. / 3., -0.25, 0.2, -1. / 6.])?)?
        .f_where_self(&d.f_abs()?.f_lt(0.001)?, &ratio.f_sub(&d)?)?;
    let inv = a.f_reciprocal()?;
    let inv2 = inv.f_square()?;
    let correction = inv.f_mul(&poly(
        &inv2,
        &[1. / 12., -1. / 360., 1. / 1260., -1. / 1680., 1. / 1188.],
    )?)?;
    let stirling = a
        .f_mul(&rate)?
        .f_add(
            &a.f_log()?
                .f_sub_scalar(std::f64::consts::TAU.ln())?
                .f_mul_scalar(0.5)?,
        )?
        .f_sub(&correction)?;
    let psi_gap = inv.f_mul_scalar(0.5)?.f_add(&inv2.f_mul(&poly(
        &inv2,
        &[1. / 12., -1. / 120., 1. / 252., -1. / 240., 1. / 132.],
    )?)?)?;
    let stable_score = ratio.f_add(&psi_gap)?.f_unsqueeze(-1)?.f_add(&v)?;
    let integral = sum_last(
        &relative
            .f_mul(&stable_score.f_where_self(&large.f_unsqueeze(-1)?, &score)?)?
            .f_mul(&w)?,
    )?;
    let result = stirling
        .f_where_self(&large, &log_pref)?
        .f_add(&scale.f_log()?)?
        .f_exp()?
        .f_mul(&integral)?;
    result.f_neg()?.f_where_self(&upper, &result)
}
fn shape_derivative(a: &Tensor, x: &Tensor) -> Native {
    let k = a.kind();
    let a = a.f_to_kind(Kind::Double)?;
    let x = x.f_to_kind(Kind::Double)?;
    let invalid = a
        .f_le(0)?
        .f_logical_or(&x.f_lt(0)?)?
        .f_logical_or(&a.f_isnan()?)?
        .f_logical_or(&x.f_isnan()?)?
        .f_logical_or(&a.f_isinf()?.f_logical_and(&x.f_isinf()?)?)?;
    let boundary = x
        .f_eq(0)?
        .f_logical_or(&x.f_isinf()?)?
        .f_logical_or(&a.f_isinf()?)?;
    let inactive = invalid.f_logical_or(&boundary)?;
    let a = a.f_masked_fill(&inactive, 1.)?;
    let x = x.f_masked_fill(&inactive, 1.)?;
    let small = x.f_le(1.)?;
    let left = small_argument(&a, &x.f_clamp_max(1.)?)?;
    let tail = tail_integral(&a, &x.f_clamp_min(1.)?)?;
    left.f_where_self(&small, &tail)?
        .f_masked_fill(&boundary, 0.)?
        .f_masked_fill(&invalid, f64::NAN)?
        .f_to_kind(k)
}

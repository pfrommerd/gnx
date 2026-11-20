use super::{elementwise::same_operands, *};
use gnx_math::ops::special::RegularizedIncompleteBetaOperation;
use tch::TchError;

type Native = Result<Tensor, TchError>;
rule!(
    RegularizedIncompleteBetaOperation,
    |_op, _eqn, [a, b, x]| {
        same_operands(a, b)?;
        same_operands(a, x)?;
        if !x.is_floating_point() {
            return Err(unsupported(
                "regularized_incomplete_beta expects real floating-point operands",
            ));
        }
        betainc(a, b, x)
    }
);

// Stirling correction to log Gamma(z), evaluated only for z >= 16.
fn stirling(z: &Tensor) -> Native {
    let r = z.f_reciprocal()?;
    let s = r.f_square()?;
    let mut p = s.f_full_like(1. / 1188.)?;
    for c in [-1. / 1680., 1. / 1260., -1. / 360., 1. / 12.] {
        p = p.f_mul(&s)?.f_add_scalar(c)?;
    }
    p.f_mul(&r)
}

// log(1+d)-d; preserve the small quadratic residual near the mode.
fn log_residual(d: &Tensor, log_ratio: &Tensor) -> Native {
    let p = d
        .f_div_scalar(-6.)?
        .f_add_scalar(0.2)?
        .f_mul(d)?
        .f_sub_scalar(0.25)?
        .f_mul(d)?
        .f_add_scalar(1. / 3.)?
        .f_mul(d)?
        .f_sub_scalar(0.5)?;
    d.f_square()?
        .f_mul(&p)?
        .f_where_self(&d.f_abs()?.f_lt(0.001)?, &log_ratio.f_sub(d)?)
}

// H=x^a*y^b/B(a,b), y=1-x. Avoid subtracting nearly equal lgammas when
// one shape is huge, and O(a+b) terms near the mode when both are huge.
fn log_prefactor(
    a: &Tensor,
    b: &Tensor,
    x: &Tensor,
    y: &Tensor,
    lx: &Tensor,
    ly: &Tensor,
) -> Native {
    let big = a.f_maximum(b)?;
    let small = a.f_minimum(b)?;
    let ratio = small.f_div(&big)?.f_log1p()?;
    let sum = a.f_add(b)?;
    let log_sum = big.f_log()?.f_add(&ratio)?;
    let small_shapes = a
        .f_mul(lx)?
        .f_add(&b.f_mul(ly)?)?
        .f_add(&sum.f_add_scalar(1.)?.f_lgamma()?)?
        .f_sub(&a.f_add_scalar(1.)?.f_lgamma()?)?
        .f_sub(&b.f_add_scalar(1.)?.f_lgamma()?)?
        .f_add(&b.f_log()?)?
        .f_sub(&log_sum)?;
    let at_mode = small
        .f_log()?
        .f_sub_scalar(std::f64::consts::TAU.ln())?
        .f_mul_scalar(0.5)?
        .f_sub(&stirling(&small)?)?
        .f_where_self(
            &small.f_ge(16.)?,
            &small
                .f_mul(&small.f_log()?)?
                .f_sub(&small)?
                .f_sub(&small.f_lgamma()?)?,
        )?
        .f_sub(&ratio.f_mul_scalar(0.5)?)?
        .f_sub(&stirling(&big)?)?
        .f_add(&stirling(&sum)?)?;
    let delta = b.f_mul(x)?.f_sub(&a.f_mul(y)?)?;
    // d_x=x/(a/(a+b))-1, and a*d_x+b*d_y=0.
    let dx = delta.f_div(a)?;
    let dy = delta.f_neg()?.f_div(b)?;
    let log_ratio = |d: &Tensor, log_t: &Tensor, shape: &Tensor| {
        d.f_log1p()?.f_where_self(
            &d.f_abs()?.f_lt(0.5)?,
            &log_t.f_add(&log_sum.f_sub(&shape.f_log()?)?)?,
        )
    };
    let rx = log_ratio(&dx, lx, a)?;
    let ry = log_ratio(&dy, ly, b)?;
    let near = dx
        .f_abs()?
        .f_lt(0.5)?
        .f_logical_and(&dy.f_abs()?.f_lt(0.5)?)?;
    let dev = a
        .f_mul(&log_residual(&dx, &rx)?)?
        .f_add(&b.f_mul(&log_residual(&dy, &ry)?)?)?
        .f_where_self(&near, &a.f_mul(&rx)?.f_add(&b.f_mul(&ry)?)?)?;
    at_mode
        .f_add(&dev)?
        .f_sub(&a.f_log()?)?
        .f_where_self(&big.f_ge(16.)?, &small_shapes)
}

// I_x(a,b) = H/a * (1+(a+b)*J), by shifting a to a+1 and integrating
// in w=-log(1-t). J is the integral on [0,W] of exp(g(w)-g(W)), where
// W=-log(y), g(w)=a*log(1-exp(-w))-b*w. The positive recurrence removes
// the integrable singularity for a<1 without cancellation.
fn log_tail(a: &Tensor, b: &Tensor, x: &Tensor, y: &Tensor, lx: &Tensor, ly: &Tensor) -> Native {
    let w = ly.f_neg()?;
    let ay = a.f_mul(y)?;
    let delta = ay.f_sub(&b.f_mul(x)?)?;
    // Concavity: left of W the log-density falls at least as fast as its
    // tangent, and -g'' >= a*y/x^2. Discard only exponentially small mass.
    let linear = x.f_div(&delta.f_clamp_min(0.)?)?.f_mul_scalar(82.)?;
    let quadratic = x.f_div(&ay.f_sqrt()?)?.f_mul_scalar(13.)?;
    let len = w.f_minimum(&linear)?.f_minimum(&quadratic)?;
    let (f, weights) = super::quadrature::unit_interval()?;
    let le = len.f_unsqueeze(-1)?;
    let v = le.f_mul(&f.f_rsub_scalar(1.)?)?;
    let t = w
        .f_sub(&len)?
        .f_unsqueeze(-1)?
        .f_add(&le.f_mul(&f)?)?
        .f_neg()?
        .f_expm1()?
        .f_neg()?;
    let xp = x.f_unsqueeze(-1)?;
    let yp = y.f_unsqueeze(-1)?;
    let ap = a.f_unsqueeze(-1)?;
    let bp = b.f_unsqueeze(-1)?;
    let q = v.f_expm1()?.f_mul(&yp)?.f_div(&xp)?;
    let lratio = q
        .f_neg()?
        .f_log1p()?
        .f_where_self(&q.f_lt(0.5)?, &t.f_log()?.f_sub(&lx.f_unsqueeze(-1)?)?)?;
    let simple = ap.f_mul(&lratio)?.f_add(&bp.f_mul(&v)?)?;
    // Near a sharp peak, a*log(t/x) and b*v cancel. Evaluate their
    // residuals before scaling instead. expm1(v)-v starts with v^2/2.
    let u = v
        .f_div_scalar(120.)?
        .f_add_scalar(1. / 24.)?
        .f_mul(&v)?
        .f_add_scalar(1. / 6.)?
        .f_mul(&v)?
        .f_add_scalar(0.5)?;
    let slope = delta.f_unsqueeze(-1)?.f_mul(&v.f_div(&xp)?)?;
    let curvature = ay
        .f_unsqueeze(-1)?
        .f_mul(&v.f_div(&xp)?)?
        .f_mul(&v)?
        .f_mul(&u)?;
    let centered = ap
        .f_mul(&log_residual(&q.f_neg()?, &lratio)?)?
        .f_sub(&slope)?
        .f_sub(&curvature)?;
    let logg = centered.f_where_self(&v.f_lt(0.001)?.f_logical_and(&q.f_lt(0.001)?)?, &simple)?;
    let avg =
        logg.f_exp()?
            .f_mul(&weights)?
            .f_sum_dim_intlist(&[-1i64][..], false, Kind::Double)?;
    let log_multiplier = a
        .f_maximum(b)?
        .f_log()?
        .f_add(&a.f_minimum(b)?.f_div(&a.f_maximum(b)?)?.f_log1p()?)?
        .f_add(&len.f_log()?)?
        .f_add(&avg.f_log()?)?;
    log_prefactor(a, b, x, y, lx, ly)?
        .f_add(&log_multiplier.f_logaddexp(&log_multiplier.f_zeros_like()?)?)
}

fn betainc(a: &Tensor, b: &Tensor, x: &Tensor) -> Native {
    let kind = x.kind();
    let a = a.f_to_kind(Kind::Double)?;
    let b = b.f_to_kind(Kind::Double)?;
    let x = x.f_to_kind(Kind::Double)?;
    let valid = a
        .f_gt(0.)?
        .f_logical_and(&b.f_gt(0.)?)?
        .f_logical_and(&a.f_isfinite()?)?
        .f_logical_and(&b.f_isfinite()?)?
        .f_logical_and(&x.f_ge(0.)?)?
        .f_logical_and(&x.f_le(1.)?)?;
    let low = x.f_eq(0.)?;
    let high = x.f_eq(1.)?;
    let inactive = valid
        .f_logical_not()?
        .f_logical_or(&low)?
        .f_logical_or(&high)?;
    let a = a.f_masked_fill(&inactive, 1.)?;
    let b = b.f_masked_fill(&inactive, 1.)?;
    let x = x.f_masked_fill(&inactive, 0.5)?;
    let y = x.f_rsub_scalar(1.)?;
    let lx = x.f_log()?;
    let ly = x.f_neg()?.f_log1p()?;
    // Reflect at the mean unless an endpoint singularity concentrates mass
    // on the other side. In this exception g(W) is at most 0.5 log units
    // below its peak, so the integral still avoids large intermediate values.
    let concentrated = |p: &Tensor, q: &Tensor, log_t: &Tensor| {
        p.f_lt(1.)?
            .f_logical_and(&p.f_lt_tensor(q)?)?
            .f_logical_and(&p.f_mul(log_t)?.f_gt(-0.5)?)
    };
    let tiny_right = concentrated(&b, &a, &ly)?;
    let tiny_left = concentrated(&a, &b, &lx)?;
    let reflect = b
        .f_mul(&x)?
        .f_gt_tensor(&a.f_mul(&y)?)?
        .f_logical_and(&tiny_right.f_logical_not()?)?
        .f_logical_or(&tiny_left)?;
    let choose = |l: &Tensor, r: &Tensor| r.f_where_self(&reflect, l);
    let log = log_tail(
        &choose(&a, &b)?,
        &choose(&b, &a)?,
        &choose(&x, &y)?,
        &choose(&y, &x)?,
        &choose(&lx, &ly)?,
        &choose(&ly, &lx)?,
    )?;
    log.f_expm1()?
        .f_neg()?
        .f_where_self(&reflect, &log.f_exp()?)?
        .f_clamp(0., 1.)?
        .f_masked_fill(&low, 0.)?
        .f_masked_fill(&high, 1.)?
        .f_masked_fill(&valid.f_logical_not()?, f64::NAN)?
        .f_to_kind(kind)
}

use super::{elementwise::same_operands, *};
use gnx_math::ops::{dtype::*, elementwise::*, special::*};
use gnx_transforms::ad::StopGradientOperation;

macro_rules! unary {
    ($($op:ty => $f:ident),+ $(,)?) => { $(rule!($op, |_op, _eqn, [x]| {x.$f()});)+ };
}
macro_rules! binary {
    ($($op:ty => $f:ident),+ $(,)?) => { $(rule!($op, |_op, _eqn, [a,b]| {same_operands(a,b)?; a.$f(b)});)+ };
}
unary!(ExpOperation=>f_exp, Expm1Operation=>f_expm1, LogOperation=>f_log, Log1pOperation=>f_log1p,
    LogisticOperation=>f_sigmoid, SqrtOperation=>f_sqrt, RsqrtOperation=>f_rsqrt, SinOperation=>f_sin,
    CosOperation=>f_cos, TanhOperation=>f_tanh, TanOperation=>f_tan, AsinOperation=>f_asin, AcosOperation=>f_acos,
    AtanOperation=>f_atan, SinhOperation=>f_sinh, CoshOperation=>f_cosh, AsinhOperation=>f_asinh, AcoshOperation=>f_acosh,
    AtanhOperation=>f_atanh, Exp2Operation=>f_exp2, Log2Operation=>f_log2, IsFiniteOperation=>f_isfinite,
    CeilOperation=>f_ceil, FloorOperation=>f_floor, SquareOperation=>f_square,
    ErfOperation=>f_erf, ErfcOperation=>f_erfc, ErfInvOperation=>f_erfinv, LgammaOperation=>f_lgamma, DigammaOperation=>f_digamma,
    BesselI0eOperation=>f_special_i0e, BesselI1eOperation=>f_special_i1e, RealOperation=>f_real, StopGradientOperation=>f_detach);
binary!(IgammaOperation=>f_igamma, IgammacOperation=>f_igammac, ZetaOperation=>f_special_zeta,
    Atan2Operation=>f_atan2, NextafterOperation=>f_nextafter);
rule!(PolygammaOperation, |_op, _eqn, [order, x]| {
    same_operands(order, x)?;
    if !x.is_floating_point() {
        return Err(unsupported(
            "polygamma expects real floating-point operands",
        ));
    }
    polygamma(order, x)
});

// ATen's polygamma takes a host integer. Keep the order in the graph using
// psi_n(x) = (-1)^(n+1) Gamma(n+1) zeta(n+1,x) for n >= 1.
fn polygamma(order: &Tensor, x: &Tensor) -> Result<Tensor, tch::TchError> {
    let kind = x.kind();
    let n = order.f_to_kind(Kind::Double)?;
    let x = x.f_to_kind(Kind::Double)?;
    let valid = n
        .f_isfinite()?
        .f_logical_and(&n.f_ge(0)?)?
        .f_logical_and(&n.f_eq_tensor(&n.f_floor()?)?)?;
    let zero = n.f_eq(0)?;
    let odd = n.f_remainder(2)?.f_eq(1)?;
    let sign = odd
        .f_to_kind(Kind::Double)?
        .f_mul_scalar(2)?
        .f_sub_scalar(1)?;
    // Supply benign arguments to the unselected zeta lanes (including n=0).
    let s = n
        .f_masked_fill(&valid.f_logical_not()?.f_logical_or(&zero)?, 1.)?
        .f_add_scalar(1)?;
    let pole = x.f_le(0)?.f_logical_and(&x.f_eq_tensor(&x.f_floor()?)?)?;
    let negative = x.f_lt(0)?.f_logical_and(&pole.f_logical_not()?)?;
    // For negative nonintegers, sum the negative terms via the difference of
    // two positive-argument zetas. Direct ATen zeta shifts q by 1 repeatedly,
    // which would take O(|x|) iterations for large negative x.
    let r = x.f_sub(&x.f_floor()?)?;
    let positive_arg = r.f_where_self(&negative, &x)?.f_masked_fill(&pole, 1.)?;
    let main = sign.f_mul(&s.f_special_zeta(&positive_arg)?)?;
    let t = x
        .f_ceil()?
        .f_sub(&x)?
        .f_masked_fill(&negative.f_logical_not()?, 1.)?;
    let end = x
        .f_rsub_scalar(1.)?
        .f_masked_fill(&negative.f_logical_not()?, 1.)?;
    let shifted = main
        .f_add(&s.f_special_zeta(&t)?)?
        .f_sub(&s.f_special_zeta(&end)?)?;
    let higher = shifted
        .f_where_self(&negative, &main)?
        .f_mul(&s.f_lgamma()?.f_exp()?)?;
    let higher = higher
        .f_masked_fill(&pole, f64::NAN)?
        .f_masked_fill(&pole.f_logical_and(&odd)?, f64::INFINITY)?;
    x.f_digamma()?
        .f_where_self(&zero, &higher)?
        .f_masked_fill(&valid.f_logical_not()?, f64::NAN)?
        .f_to_kind(kind)
}
rule!(PowOperation, |_op, _eqn, [x, y]| {
    if x.size() != y.size() {
        return Err(invalid("power operand shapes must match"));
    }
    y.f_to_kind(x.kind()).and_then(|y| x.f_pow(&y))
});
rule!(SignOperation, |_op, _eqn, [x]| {
    if x.is_floating_point() {
        // sign propagates NaNs and preserves the sign of zero.
        (|| x.f_where_self(&x.f_isnan()?.f_logical_or(&x.f_eq(0)?)?, &x.f_sign()?))()
    } else {
        x.f_sgn()
    }
});
rule!(CbrtOperation, |_op, _eqn, [x]| {
    if x.is_complex() {
        x.f_pow_tensor_scalar(1. / 3.)
    } else {
        x.f_abs()
            .and_then(|a| a.f_pow_tensor_scalar(1. / 3.))
            .and_then(|m| m.f_copysign(x))
    }
});
rule!(OneMinusSquareOperation, |_op, _eqn, [x]| {
    (|| x.f_rsub_scalar(1)?.f_mul(&x.f_add_scalar(1)?))()
});
rule!(ClampOperation, |_op, _eqn, [lo, x, hi]| {
    if [lo, hi]
        .iter()
        .any(|v| v.kind() != x.kind() || (v.dim() != 0 && v.size() != x.size()))
    {
        return Err(invalid("clamp bounds must be scalars or match operand"));
    }
    x.f_clamp_tensor(Some(lo), Some(hi))
});
rule!(RoundOperation, |op, _eqn, [x]| {
    match op.method() {
        RoundingMethod::NearestEven => x.f_round(),
        RoundingMethod::AwayFromZero => (|| {
            // Avoid adding 0.5 to a value just below the tie (it may round up).
            let m = x.f_abs()?;
            let i = m.f_floor()?;
            i.f_add(&m.f_sub(&i)?.f_ge(0.5)?.f_to_kind(x.kind())?)?
                .f_copysign(x)
        })(),
    }
});
rule!(IntegerPowOperation, |op, _eqn, [x]| {
    x.f_pow_tensor_scalar(op.exponent())
});
rule!(ConvertElementTypeOperation, |_op, eqn, [x]| {
    x.f_to_kind(result_layout(eqn)?.1)
});
rule!(BitcastConvertTypeOperation, |_op, eqn, [x]| {
    let (target, dst) = result_layout(eqn)?;
    let src = x.kind();
    if x.is_complex() != matches!(dst, Kind::ComplexFloat | Kind::ComplexDouble) {
        return Err(invalid(
            "bitcast cannot cross between complex and real types",
        ));
    }
    if dst == Kind::Bool && src != Kind::Bool {
        return Err(unsupported(
            "bitcast to bool cannot guarantee canonical 0/1 bytes",
        ));
    }
    let (from, to) = (src.elt_size_in_bytes(), dst.elt_size_in_bytes());
    let mut expected = x.size();
    match from.cmp(&to) {
        std::cmp::Ordering::Greater => expected.push((from / to) as i64),
        std::cmp::Ordering::Less if expected.pop() != Some((to / from) as i64) => {
            return Err(invalid(
                "widening bitcast needs a trailing dimension equal to the width ratio",
            ));
        }
        _ => (),
    }
    if target != expected {
        return Err(invalid("bitcast result shape mismatch"));
    }
    (|| {
        let x = x.f_resolve_conj()?.f_resolve_neg()?;
        if src == dst {
            return Ok(x);
        }
        if from == to {
            return x.f_view_copy_dtype(dst);
        }
        // Torch changes the size of an existing last axis; GNX adds or removes
        // one instead. Pack logical elements in a fresh buffer so replay also
        // works on strided inputs or slices with an unaligned storage offset.
        x.f_flatten(0, -1)?
            .f_alias_copy()?
            .f_view_copy_dtype(dst)?
            .f_reshape(target)
    })()
});
rule!(ConjOperation, |_op, eqn, [x]| {
    x.f_to_kind(result_layout(eqn)?.1)
        .and_then(|v| v.f_conj_physical())
});
rule!(ImagOperation, |_op, _eqn, [x]| {
    if x.is_complex() {
        x.f_imag()
    } else {
        x.f_zeros_like()
    }
});
rule!(ComplexOperation, |_op, _eqn, [x, y]| {
    same_operands(x, y)?;
    Tensor::f_complex(x, y)
});
impl TorchLowering for PolynomialOperation {
    fn lower_torch(&self, _: &Eqn, inputs: &[Tensor]) -> io::Result<Vec<Tensor>> {
        let [x, coeffs @ ..] = inputs else {
            return Err(invalid("missing polynomial argument"));
        };
        if coeffs
            .iter()
            .any(|c| c.kind() != x.kind() || (c.dim() != 0 && c.size() != x.size()))
        {
            return Err(invalid("invalid polynomial coefficient"));
        }
        let y = (|| {
            let Some((highest, rest)) = coeffs.split_last() else {
                return x.f_zeros_like();
            };
            rest.iter()
                .rev()
                .try_fold(highest.f_expand_as(x)?, |v, c| v.f_mul(x)?.f_add(c))
        })()
        .map_err(io::Error::other)?;
        Ok(vec![y])
    }
}
crate::register_torch_lowering!(PolynomialOperation);

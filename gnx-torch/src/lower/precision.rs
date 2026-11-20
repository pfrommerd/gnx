use super::*;
use gnx_math::ops::dtype::ReducePrecisionOperation;
use tch::TchError;

rule!(ReducePrecisionOperation, |op, _eqn, [x]| {
    let (src_e, src_m) = match x.kind() {
        Kind::Half => (5, 10),
        Kind::BFloat16 => (8, 7),
        Kind::Float => (8, 23),
        Kind::Double => (11, 52),
        _ => {
            return Err(unsupported(
                "reduce_precision requires a real floating-point operand",
            ));
        }
    };
    let e = op.exponent_bits().min(src_e);
    let m = op.mantissa_bits().min(src_m);
    if e == 0 {
        return Err(invalid(
            "reduce_precision requires at least one exponent bit",
        ));
    }
    if (e, m) == (src_e, src_m) {
        return Ok(vec![x.shallow_clone()]);
    }
    (|| -> Result<Tensor, TchError> {
        let magnitude = x.f_to_kind(Kind::Double)?.f_abs()?;
        let mut rounded = magnitude.shallow_clone();
        if m < src_m {
            // Round on the source format's grid. Subnormals have the same ULP
            // as the smallest normal. All grid spacings are exact powers of two.
            let bias = (1i64 << (src_e - 1)) - 1;
            let (_, exponent) = magnitude.f_frexp()?;
            let exponent = exponent.f_sub_scalar(1)?;
            let step = exponent
                .f_clamp_min(1 - bias)?
                .f_sub_scalar(i64::from(m))?
                .f_to_kind(Kind::Double)?
                .f_exp2()?;
            let units = magnitude.f_div(&step)?;
            let mut integral = units.f_round()?;
            if m == 0 {
                // No fraction bits remain: the retained low bit at a normal
                // midpoint is the *biased exponent's* low bit (not always 1).
                let even = exponent.f_add_scalar(bias)?.f_remainder(2)?.f_eq(0)?;
                integral = integral.f_masked_fill(&units.f_eq(1.5)?.f_logical_and(&even)?, 1.)?;
            }
            rounded = integral.f_mul(&step)?;
        }
        if e < src_e {
            let bias = (1i32 << (e - 1)) - 1;
            // Clip after mantissa rounding, including carries across the lower
            // and upper thresholds. A reduced exponent flushes denormals.
            rounded = rounded
                .f_masked_fill(&rounded.f_lt(2f64.powi(1 - bias))?, 0.)?
                .f_masked_fill(&rounded.f_ge(2f64.powi(bias + 1))?, f64::INFINITY)?;
        }
        let signed = rounded.f_to_kind(x.kind())?.f_copysign(x)?;
        if m == 0 {
            signed.f_masked_fill(&x.f_isnan()?, f64::INFINITY)
        } else {
            x.f_where_self(&x.f_isnan()?, &signed)
        }
    })()
});

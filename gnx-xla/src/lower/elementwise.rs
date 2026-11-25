use super::{LoweringContext, XlaLowering, invalid, native, result_info};
use gnx_expr::Eqn;
use gnx_math::ops::{bitwise::*, dtype::*, elementwise::*, special::*};
use std::io;
use xla::XlaOp;

macro_rules! lower_elementwise {
    ($($ty:ty, $method:ident,  $arity:tt;)*) => {$(
        impl XlaLowering for $ty {
            fn lower_xla(
                &self,
                _: &LoweringContext<'_>,
                _eqn: Eqn<'_>,
                inputs: &[XlaOp],
            ) -> io::Result<Vec<XlaOp>> {
                Ok(vec![lower_elementwise!(@call self, _eqn, inputs, $method, $arity)?])
            }
        }

        crate::register_xla_lowering!($ty);
    )*};
    (@call $op:ident, $eqn:ident, $inputs:ident, conj, 1) => {{
        let [input] = $inputs else { return Err(invalid("conj signature")); };
        let ty = super::element(result_info($eqn)?.dtype()).primitive_type();
        let input = native(input.convert(ty))?;
        native(input.conj())
    }};
    (@call $op:ident, $eqn:ident, $inputs:ident, pow, 2) => {{
        let [lhs, rhs] = $inputs else { return Err(invalid("power signature")); };
        if native(lhs.array_shape())?.dims() != native(rhs.array_shape())?.dims() {
            return Err(invalid("power operand shapes must match"));
        }
        let rhs = native(rhs.convert(native(lhs.ty())?))?;
        native(lhs.pow(&rhs))
    }};
    (@call $op:ident, $eqn:ident, $inputs:ident, bitcast, 1) => {{
        let [input] = $inputs else { return Err(invalid("bitcast signature")); };
        native(input.bitcast(super::element(result_info($eqn)?.dtype()).primitive_type()))
    }};
    (@call $op:ident, $eqn:ident, $inputs:ident, exp2, 1) => {{
        let [input] = $inputs else { return Err(invalid("exp2 signature")); };
        let two = scalar_like(input, 2.)?;
        native(two.pow(input))
    }};
    (@call $op:ident, $eqn:ident, $inputs:ident, log2, 1) => {{
        let [input] = $inputs else { return Err(invalid("log2 signature")); };
        native(native(input.log())?.div_(&scalar_like(input, std::f64::consts::LN_2)?))
    }};
    (@call $op:ident, $eqn:ident, $inputs:ident, one_minus_square, 1) => {{
        let [input] = $inputs else { return Err(invalid("one_minus_square signature")); };
        let one = scalar_like(input, 1.)?;
        native(native(one.sub_(input))?.mul_(&native(one.add_(input))?))
    }};
    (@call $op:ident, $eqn:ident, $inputs:ident, mulhi, 2) => {{
        let [lhs, rhs] = $inputs else { return Err(invalid("mulhi signature")); };
        mulhi(lhs, rhs)
    }};
    (@call $op:ident, $eqn:ident, $inputs:ident, betainc, 3) => {{
        let [a, b, x] = $inputs else { return Err(invalid("beta signature")); };
        if native(a.shape())? != native(b.shape())? || native(a.shape())? != native(x.shape())? {
            return Err(invalid("beta operands must have the same shape and dtype"));
        }
        native(a.betainc(b, x))
    }};
    (@call $op:ident, $eqn:ident, $inputs:ident, convert, 1) => {{
        let [input] = $inputs else { return Err(invalid("conversion signature")); };
        native(input.convert(super::element(result_info($eqn)?.dtype()).primitive_type()))
    }};
    (@call $op:ident, $eqn:ident, $inputs:ident, square, 1) => {{
        let [input] = $inputs else { return Err(invalid("square signature")); };
        native(input.mul_(input))
    }};
    (@call $op:ident, $eqn:ident, $inputs:ident, $method:ident, 1) => {{
        let [input] = $inputs else { return Err(invalid("unary operation signature")); };
        native(input.$method())
    }};
    (@call $op:ident, $eqn:ident, $inputs:ident, $method:ident, 2) => {{
        let [lhs, rhs] = $inputs else { return Err(invalid("binary operation signature")); };
        if native(lhs.shape())? != native(rhs.shape())? {
            return Err(invalid("binary elementwise operands must have the same shape and dtype"));
        }
        native(lhs.$method(rhs))
    }};
    (@call $op:ident, $eqn:ident, $inputs:ident, clamp, 3) => {{
        let [min, input, max] = $inputs else { return Err(invalid("clamp signature")); };
        native(input.clamp(min, max))
    }};
}

lower_elementwise! {
    SubOperation, sub_, 2;
    DivOperation, div_, 2;
    RemOperation, rem_, 2;
    NegOperation, neg, 1;
    AbsOperation, abs, 1;
    SignOperation, sign, 1;
    MinOperation, min, 2;
    MaxOperation, max, 2;
    PowOperation, pow, 2;
    ExpOperation, exp, 1;
    Expm1Operation, expm1, 1;
    LogOperation, log, 1;
    Log1pOperation, log1p, 1;
    LogisticOperation, logistic, 1;
    SqrtOperation, sqrt, 1;
    RsqrtOperation, rsqrt, 1;
    CbrtOperation, cbrt, 1;
    SinOperation, sin, 1;
    CosOperation, cos, 1;
    TanhOperation, tanh, 1;
    Atan2Operation, atan2, 2;
    EqOperation, eq, 2;
    NeOperation, ne, 2;
    LtOperation, lt, 2;
    LeOperation, le, 2;
    GtOperation, gt, 2;
    GeOperation, ge, 2;
    IsFiniteOperation, is_finite, 1;
    CeilOperation, ceil, 1;
    FloorOperation, floor, 1;
    ClampOperation, clamp, 3;
    SquareOperation, square, 1;
    AndOperation, and, 2;
    OrOperation, or, 2;
    XorOperation, xor, 2;
    NotOperation, not, 1;
    ClzOperation, clz, 1;
    RealOperation, real, 1;
    ImagOperation, imag, 1;
    ConvertElementTypeOperation, convert, 1;
    ErfOperation, erf, 1;
    TanOperation, tan, 1;
    AsinOperation, asin, 1;
    AcosOperation, acos, 1;
    AtanOperation, atan, 1;
    SinhOperation, sinh, 1;
    CoshOperation, cosh, 1;
    AsinhOperation, asinh, 1;
    AcoshOperation, acosh, 1;
    AtanhOperation, atanh, 1;
    Exp2Operation, exp2, 1;
    Log2Operation, log2, 1;
    OneMinusSquareOperation, one_minus_square, 1;
    NextafterOperation, nextafter, 2;
    EqToOperation, eq_to, 2;
    LtToOperation, lt_to, 2;
    LeToOperation, le_to, 2;
    MulhiOperation, mulhi, 2;
    PopulationCountOperation, population_count, 1;
    ShiftLeftOperation, shift_left, 2;
    ShiftRightArithmeticOperation, shift_right_arithmetic, 2;
    ShiftRightLogicalOperation, shift_right_logical, 2;
    ConjOperation, conj, 1;
    BitcastConvertTypeOperation, bitcast, 1;
    ComplexOperation, complex, 2;
    ErfcOperation, erfc, 1;
    ErfInvOperation, erf_inv, 1;
    LgammaOperation, lgamma, 1;
    DigammaOperation, digamma, 1;
    BesselI0eOperation, bessel_i0e, 1;
    BesselI1eOperation, bessel_i1e, 1;
    IgammaOperation, igamma, 2;
    IgammacOperation, igammac, 2;
    IgammaGradAOperation, igamma_grad_a, 2;
    PolygammaOperation, polygamma, 2;
    ZetaOperation, zeta, 2;
    RegularizedIncompleteBetaOperation, betainc, 3;
}

fn scalar_like(input: &XlaOp, value: f64) -> io::Result<XlaOp> {
    let scalar = native(input.builder().c0(value))?;
    native(scalar.convert(native(input.ty())?))
}

fn mulhi(lhs: &XlaOp, rhs: &XlaOp) -> io::Result<XlaOp> {
    use xla::PrimitiveType::*;
    if native(lhs.shape())? != native(rhs.shape())? {
        return Err(invalid("mulhi operands must have the same shape and dtype"));
    }
    let ty = native(lhs.ty())?;
    let (wide, width) = match ty {
        S8 => (S16, 8),
        S16 => (S32, 16),
        S32 => (S64, 32),
        U8 => (U16, 8),
        U16 => (U32, 16),
        U32 => (U64, 32),
        S64 | U64 => return mulhi64(lhs, rhs, ty == S64),
        _ => return Err(invalid("mulhi requires integer operands")),
    };
    let left = native(lhs.convert(wide))?;
    let right = native(rhs.convert(wide))?;
    let prod = native(left.mul_(&right))?;
    let shift = scalar_like(&prod, width as f64)?;
    let high = native(prod.shift_right_logical(&shift))?;
    native(high.convert(ty))
}

fn mulhi64(lhs: &XlaOp, rhs: &XlaOp, signed: bool) -> io::Result<XlaOp> {
    use xla::PrimitiveType::{S64, U64};
    let u = native(lhs.bitcast(U64))?;
    let v = native(rhs.bitcast(U64))?;
    let builder = lhs.builder();
    let mask = native(builder.c0(u32::MAX as u64))?;
    let half = native(builder.c0(32_u64))?;
    let split = |x: &XlaOp| {
        Ok::<_, io::Error>((native(x.and(&mask))?, native(x.shift_right_logical(&half))?))
    };
    let (u0, u1) = split(&u)?;
    let (v0, v1) = split(&v)?;
    let prod = native(u0.mul_(&v0))?;
    let carry = native(prod.shift_right_logical(&half))?;
    let middle = native(native(u1.mul_(&v0))?.add_(&carry))?;
    let (low_middle, high_middle) = split(&middle)?;
    let cross = native(native(u0.mul_(&v1))?.add_(&low_middle))?;
    let high_cross = native(cross.shift_right_logical(&half))?;
    let mut high = native(native(u1.mul_(&v1))?.add_(&high_middle))?;
    high = native(high.add_(&high_cross))?;
    if signed {
        let zero = native(
            native(builder.c0(0_u64))?.broadcast_in_dim(native(u.array_shape())?.dims(), &[]),
        )?;
        let signed_zero = native(builder.c0(0_i64))?;
        let u_negative = native(lhs.lt(&signed_zero))?;
        let v_negative = native(rhs.lt(&signed_zero))?;
        high = native(high.sub_(&native(u_negative.select(&v, &zero))?))?;
        high = native(high.sub_(&native(v_negative.select(&u, &zero))?))?;
        native(high.bitcast(S64))
    } else {
        Ok(high)
    }
}

impl XlaLowering for RoundOperation {
    fn lower_xla(
        &self,
        _: &LoweringContext<'_>,
        _: Eqn<'_>,
        inputs: &[XlaOp],
    ) -> io::Result<Vec<XlaOp>> {
        let x = super::unary("round", inputs)?;
        Ok(vec![match self.method() {
            RoundingMethod::AwayFromZero => native(x.round())?,
            RoundingMethod::NearestEven => native(x.round_even())?,
        }])
    }
}

impl XlaLowering for IntegerPowOperation {
    fn lower_xla(
        &self,
        _: &LoweringContext<'_>,
        _: Eqn<'_>,
        inputs: &[XlaOp],
    ) -> io::Result<Vec<XlaOp>> {
        let x = super::unary("integer_pow", inputs)?;
        use xla::PrimitiveType::*;
        if self.exponent() < 0
            && matches!(native(x.ty())?, S8 | S16 | S32 | S64 | U8 | U16 | U32 | U64)
        {
            return Err(invalid("integer operands require nonnegative exponents"));
        }
        let mut n = self.exponent().unsigned_abs();
        let mut value = x.clone();
        let one = scalar_like(x, 1.)?;
        let mut product = None;
        while n > 0 {
            if n % 2 == 1 {
                product = Some(match product {
                    None => value.clone(),
                    Some(previous) => native(value.mul_(&previous))?,
                });
            }
            n /= 2;
            if n > 0 {
                value = native(value.mul_(&value))?;
            }
        }
        let result = match product {
            Some(result) if self.exponent() < 0 => native(one.div_(&result))?,
            Some(result) => result,
            None => native(one.broadcast_in_dim(native(x.array_shape())?.dims(), &[]))?,
        };
        Ok(vec![result])
    }
}

impl XlaLowering for PolynomialOperation {
    fn lower_xla(
        &self,
        _: &LoweringContext<'_>,
        _: Eqn<'_>,
        inputs: &[XlaOp],
    ) -> io::Result<Vec<XlaOp>> {
        let [x, coeffs @ ..] = inputs else {
            return Err(invalid("polynomial signature"));
        };
        let shape = native(x.array_shape())?;
        let Some((highest, lower)) = coeffs.split_last() else {
            return Ok(vec![native(
                scalar_like(x, 0.)?.broadcast_in_dim(shape.dims(), &[]),
            )?]);
        };
        let expand = |coef: &XlaOp| {
            let input = native(coef.array_shape())?;
            if input.ty() != shape.ty()
                || (!input.dims().is_empty() && input.dims() != shape.dims())
            {
                return Err(invalid(
                    "polynomial coefficients must be scalars or match the operand",
                ));
            }
            if input.dims().is_empty() {
                native(coef.broadcast_in_dim(shape.dims(), &[]))
            } else {
                Ok(coef.clone())
            }
        };
        let mut y = expand(highest)?;
        for coef in lower.iter().rev() {
            y = native(native(y.mul_(x))?.add_(&expand(coef)?))?;
        }
        Ok(vec![y])
    }
}

impl XlaLowering for ReducePrecisionOperation {
    fn lower_xla(
        &self,
        _: &LoweringContext<'_>,
        _: Eqn<'_>,
        inputs: &[XlaOp],
    ) -> io::Result<Vec<XlaOp>> {
        let x = super::unary("reduce_precision", inputs)?;
        Ok(vec![native(x.reduce_precision(
            self.exponent_bits().into(),
            self.mantissa_bits().into(),
        ))?])
    }
}

crate::register_xla_lowering!(
    RoundOperation,
    IntegerPowOperation,
    PolynomialOperation,
    ReducePrecisionOperation
);

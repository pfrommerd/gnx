use super::{Op, arity};

pub use super::{add_op as add, mul_op as mul};
pub use gnx_common::ops::{DivOperation, RemOperation, SubOperation, div_op, rem_op, sub_op};
pub use gnx_common::ops::{div_op as div, rem_op as rem, sub_op as sub};

simple_operations! {
    NegOperation, neg_op, neg, "neg", 1;
    AbsOperation, abs_op, abs, "abs", 1;
    SignOperation, sign_op, sign, "sign", 1;
    MinOperation, min_op, min, "min", 2;
    MaxOperation, max_op, max, "max", 2;
    PowOperation, pow_op, pow, "pow", 2;
    ExpOperation, exp_op, exp, "exp", 1;
    Expm1Operation, expm1_op, expm1, "expm1", 1;
    LogOperation, log_op, log, "log", 1;
    Log1pOperation, log1p_op, log1p, "log1p", 1;
    LogisticOperation, logistic_op, logistic, "logistic", 1;
    SqrtOperation, sqrt_op, sqrt, "sqrt", 1;
    RsqrtOperation, rsqrt_op, rsqrt, "rsqrt", 1;
    CbrtOperation, cbrt_op, cbrt, "cbrt", 1;
    SinOperation, sin_op, sin, "sin", 1;
    CosOperation, cos_op, cos, "cos", 1;
    TanhOperation, tanh_op, tanh, "tanh", 1;
    Atan2Operation, atan2_op, atan2, "atan2", 2;
    EqOperation, eq_op, eq, "eq", 2;
    NeOperation, ne_op, ne, "ne", 2;
    LtOperation, lt_op, lt, "lt", 2;
    LeOperation, le_op, le, "le", 2;
    GtOperation, gt_op, gt, "gt", 2;
    GeOperation, ge_op, ge, "ge", 2;
    IsFiniteOperation, is_finite_op, is_finite, "is_finite", 1;
    CeilOperation, ceil_op, ceil, "ceil", 1;
    FloorOperation, floor_op, floor, "floor", 1;
    ClampOperation, clamp_op, clamp, "clamp", 3;
    SquareOperation, square_op, square, "square", 1;
    TanOperation, tan_op, tan, "tan", 1;
    AsinOperation, asin_op, asin, "asin", 1;
    AcosOperation, acos_op, acos, "acos", 1;
    AtanOperation, atan_op, atan, "atan", 1;
    SinhOperation, sinh_op, sinh, "sinh", 1;
    CoshOperation, cosh_op, cosh, "cosh", 1;
    AsinhOperation, asinh_op, asinh, "asinh", 1;
    AcoshOperation, acosh_op, acosh, "acosh", 1;
    AtanhOperation, atanh_op, atanh, "atanh", 1;
    Exp2Operation, exp2_op, exp2, "exp2", 1;
    Log2Operation, log2_op, log2, "log2", 1;
    OneMinusSquareOperation, one_minus_square_op, one_minus_square, "one_minus_square", 1;
    NextafterOperation, nextafter_op, nextafter, "nextafter", 2;
    EqToOperation, eq_to_op, eq_to, "eq_to", 2;
    LtToOperation, lt_to_op, lt_to, "lt_to", 2;
    LeToOperation, le_to_op, le_to, "le_to", 2;
    MulhiOperation, mulhi_op, mulhi, "mulhi", 2;
}

#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Hash, rkyv::Archive, rkyv::Serialize, rkyv::Portable,
)]
#[rkyv(as = Self)]
#[repr(u8)]
pub enum RoundingMethod {
    AwayFromZero,
    NearestEven,
}

#[derive(rkyv::Archive, rkyv::Serialize)]
#[rkyv(archived = RoundOperation)]
pub struct Round {
    method: RoundingMethod,
}
impl RoundOperation {
    pub fn method(&self) -> RoundingMethod {
        self.method
    }
}
gnx_common::array_operation!(RoundOperation, "round", |eqn| arity(eqn, 1));
pub fn round(method: RoundingMethod) -> Op {
    Op::new(Round { method })
}

#[derive(rkyv::Archive, rkyv::Serialize)]
#[rkyv(archived = IntegerPowOperation)]
pub struct IntegerPow {
    exponent: i64,
}
impl IntegerPowOperation {
    pub fn exponent(&self) -> i64 {
        self.exponent.to_native()
    }
}
gnx_common::array_operation!(
    IntegerPowOperation,
    "integer_pow",
    |op, eqn| {
        let _ = op;
        arity(eqn, 1)
    },
    exponent
);
pub fn integer_pow(exponent: i64) -> Op {
    Op::new(IntegerPow { exponent })
}

gnx_common::unit_operations! {
    pub struct PolynomialOperation;
}
gnx_common::array_operation!(PolynomialOperation, "polynomial", |eqn| {
    if eqn.inputs().is_empty() {
        return Err(super::signature("polynomial needs an evaluation point"));
    }
    arity(eqn, eqn.inputs().len())
});
/// Coefficients follow the evaluation point in order from lowest to highest degree.
pub fn polynomial() -> Op {
    Op::new(PolynomialOperation)
}

use super::{Dialect, Eqn, Op, Operation, Result, arity, fmt};

pub use super::{add_op as add, mul_op as mul};
pub use gnx_array::ops::{DivOperation, RemOperation, SubOperation, div_op, rem_op, sub_op};
pub use gnx_array::ops::{div_op as div, rem_op as rem, sub_op as sub};

simple_operations! {
    NegOperation, neg_op, neg, 1;
    AbsOperation, abs_op, abs, 1;
    SignOperation, sign_op, sign, 1;
    MinOperation, min_op, min, 2;
    MaxOperation, max_op, max, 2;
    PowOperation, pow_op, pow, 2;
    ExpOperation, exp_op, exp, 1;
    Expm1Operation, expm1_op, expm1, 1;
    LogOperation, log_op, log, 1;
    Log1pOperation, log1p_op, log1p, 1;
    LogisticOperation, logistic_op, logistic, 1;
    SqrtOperation, sqrt_op, sqrt, 1;
    RsqrtOperation, rsqrt_op, rsqrt, 1;
    CbrtOperation, cbrt_op, cbrt, 1;
    SinOperation, sin_op, sin, 1;
    CosOperation, cos_op, cos, 1;
    TanhOperation, tanh_op, tanh, 1;
    Atan2Operation, atan2_op, atan2, 2;
    EqOperation, eq_op, eq, 2;
    NeOperation, ne_op, ne, 2;
    LtOperation, lt_op, lt, 2;
    LeOperation, le_op, le, 2;
    GtOperation, gt_op, gt, 2;
    GeOperation, ge_op, ge, 2;
    IsFiniteOperation, is_finite_op, is_finite, 1;
    CeilOperation, ceil_op, ceil, 1;
    FloorOperation, floor_op, floor, 1;
    ClampOperation, clamp_op, clamp, 3;
    SquareOperation, square_op, square, 1;
    TanOperation, tan_op, tan, 1;
    AsinOperation, asin_op, asin, 1;
    AcosOperation, acos_op, acos, 1;
    AtanOperation, atan_op, atan, 1;
    SinhOperation, sinh_op, sinh, 1;
    CoshOperation, cosh_op, cosh, 1;
    AsinhOperation, asinh_op, asinh, 1;
    AcoshOperation, acosh_op, acosh, 1;
    AtanhOperation, atanh_op, atanh, 1;
    Exp2Operation, exp2_op, exp2, 1;
    Log2Operation, log2_op, log2, 1;
    OneMinusSquareOperation, one_minus_square_op, one_minus_square, 1;
    NextafterOperation, nextafter_op, nextafter, 2;
    EqToOperation, eq_to_op, eq_to, 2;
    LtToOperation, lt_to_op, lt_to, 2;
    LeToOperation, le_to_op, le_to, 2;
    MulhiOperation, mulhi_op, mulhi, 2;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RoundingMethod {
    AwayFromZero,
    NearestEven,
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct RoundOperation {
    method: RoundingMethod,
}
impl RoundOperation {
    pub fn method(&self) -> RoundingMethod {
        self.method
    }
}
gnx_array::array_operation!(RoundOperation, "round", |eqn| arity(eqn, 1));
pub fn round(method: RoundingMethod) -> Op {
    Op::new(RoundOperation { method })
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct IntegerPowOperation {
    exponent: i64,
}
impl IntegerPowOperation {
    pub fn exponent(&self) -> i64 {
        self.exponent
    }
}
gnx_array::array_operation!(
    IntegerPowOperation,
    "integer_pow",
    |op, eqn| {
        let _ = op;
        arity(eqn, 1)
    },
    exponent
);
pub fn integer_pow(exponent: i64) -> Op {
    Op::new(IntegerPowOperation { exponent })
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct PolynomialOperation;
gnx_array::array_operation!(PolynomialOperation, "polynomial", |eqn| {
    if eqn.inputs().is_empty() {
        return Err(super::signature("polynomial needs an evaluation point"));
    }
    arity(eqn, eqn.inputs().len())
});
/// Coefficients follow the evaluation point in order from lowest to highest degree.
pub fn polynomial() -> Op {
    Op::new(PolynomialOperation)
}

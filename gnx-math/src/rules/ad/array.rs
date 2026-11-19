use gnx_transforms::TransformError;
use gnx_transforms::ad::def::{Ctx, Val, defjvp, deflinear, deftranspose};

use crate::ops::{bitwise::*, dtype::*, elementwise::*};

fn unary<'a>(cx: &'a Ctx<'a>, op: gnx_expr::Op, x: Val<'a>) -> Val<'a> {
    cx.emit(op, &[x], x.info())
}

fn binary<'a>(cx: &'a Ctx<'a>, op: gnx_expr::Op, x: Val<'a>, y: Val<'a>) -> Val<'a> {
    cx.emit(op, &[x, y], x.info())
}

fn one<'a>(cx: &'a Ctx<'a>, value: Val<'a>) -> Val<'a> {
    cx.emit(crate::ops::ones_op(), &[], value.info())
}

fn divide<'a>(cx: &'a Ctx<'a>, x: Val<'a>, y: Val<'a>) -> Val<'a> {
    binary(cx, div_op(), x, y)
}

fn negate<'a>(cx: &'a Ctx<'a>, x: Val<'a>) -> Val<'a> {
    unary(cx, neg_op(), x)
}

defjvp!(SubOperation, |_, t, _| Ok(t), |cx, t, _| Ok(negate(cx, t)));
deftranspose!(SubOperation, |_, ct, _| Ok(ct), |cx, ct, _| Ok(negate(
    cx, ct
)));
deflinear!(NegOperation, |cx, ct, _| Ok(negate(cx, ct)));
defjvp!(DivOperation, |cx, t, [_, y]| Ok(divide(cx, t, y)), |cx,
                                                             t,
                                                             [
    _,
    y,
]| Ok(
    negate(cx, divide(cx, t * cx.ans()?, y))
));
deftranspose!(
    DivOperation,
    |cx, ct, [_, y]| Ok(divide(cx, ct, y.val()?)),
    None
);
defjvp!(SquareOperation, |_, t, [x]| Ok((x + x) * t));
defjvp!(ExpOperation, |cx, t, _| Ok(cx.ans()? * t));
defjvp!(Expm1Operation, |cx, t, [x]| Ok(unary(cx, exp_op(), x) * t));
defjvp!(LogOperation, |cx, t, [x]| Ok(divide(cx, t, x)));
defjvp!(Log1pOperation, |cx, t, [x]| Ok(divide(
    cx,
    t,
    one(cx, x) + x
)));
defjvp!(SqrtOperation, |cx, t, _| {
    let ans = cx.ans()?;
    Ok(divide(cx, t, ans + ans))
});
defjvp!(RsqrtOperation, |cx, t, _| {
    let ans = cx.ans()?;
    let unit = one(cx, ans);
    Ok(negate(cx, divide(cx, ans * ans * ans * t, unit + unit)))
});
defjvp!(CbrtOperation, |cx, t, _| {
    let square = cx.ans()? * cx.ans()?;
    Ok(divide(cx, t, square + square + square))
});
defjvp!(SinOperation, |cx, t, [x]| Ok(unary(cx, cos_op(), x) * t));
defjvp!(CosOperation, |cx, t, [x]| Ok(negate(
    cx,
    unary(cx, sin_op(), x)
) * t));
defjvp!(TanhOperation, |cx, t, _| {
    let y = cx.ans()?;
    Ok(binary(cx, sub_op(), one(cx, y), y * y) * t)
});
defjvp!(LogisticOperation, |cx, t, _| {
    let y = cx.ans()?;
    Ok(y * binary(cx, sub_op(), one(cx, y), y) * t)
});
defjvp!(
    Atan2Operation,
    |cx, t, [x, y]| Ok(divide(cx, y, x * x + y * y) * t),
    |cx, t, [x, y]| Ok(negate(cx, divide(cx, x, x * x + y * y)) * t)
);

fn predicate<'a>(
    cx: &'a Ctx<'a>,
    op: gnx_expr::Op,
    lhs: Val<'a>,
    rhs: Val<'a>,
) -> gnx_transforms::TransformResult<Val<'a>> {
    Ok(cx.emit(
        op,
        &[lhs, rhs],
        gnx_transforms::util::info(&lhs.dims()?, gnx_array::DType::Bool),
    ))
}

fn extremum_tangent<'a>(
    cx: &'a Ctx<'a>,
    t: Val<'a>,
    x: Val<'a>,
    other: Val<'a>,
) -> gnx_transforms::TransformResult<Val<'a>> {
    let ans = cx.ans()?;
    let unit = one(cx, t);
    let zero = t.zeros_like();
    let tie = predicate(cx, eq_op(), other, ans)?;
    let wins = predicate(cx, eq_op(), x, ans)?;
    let weight = cx.select(tie, divide(cx, unit, unit + unit), unit);
    Ok(cx.select(wins, weight, zero) * t)
}

defjvp!(
    MinOperation,
    |cx, t, [x, y]| extremum_tangent(cx, t, x, y),
    |cx, t, [x, y]| extremum_tangent(cx, t, y, x)
);
defjvp!(
    MaxOperation,
    |cx, t, [x, y]| extremum_tangent(cx, t, x, y),
    |cx, t, [x, y]| extremum_tangent(cx, t, y, x)
);

defjvp!(IsFiniteOperation, None);
defjvp!(SignOperation, None);
defjvp!(CeilOperation, None);
defjvp!(FloorOperation, None);
defjvp!(EqOperation, None, None);
defjvp!(NeOperation, None, None);
defjvp!(LtOperation, None, None);
defjvp!(LeOperation, None, None);
defjvp!(GtOperation, None, None);
defjvp!(GeOperation, None, None);
defjvp!(AndOperation, None, None);
defjvp!(OrOperation, None, None);
defjvp!(XorOperation, None, None);
defjvp!(NotOperation, None);
defjvp!(ClzOperation, None);
defjvp!(ConvertElementTypeOperation, |cx, t, [x]| {
    if gnx_transforms::util::is_inexact(&x.info()) && gnx_transforms::util::is_inexact(cx.out()) {
        Ok(cx.apply(&[t]))
    } else {
        Ok(cx.zeros(cx.out()))
    }
});
deftranspose!(ConvertElementTypeOperation, |cx, ct, [x]| {
    Ok(cx.emit(convert_element_type_op(), &[ct], x.info().clone()))
});

use crate::ops::special::*;

fn constant<'a>(
    cx: &'a Ctx<'a>,
    value: f64,
    like: Val<'a>,
) -> gnx_transforms::TransformResult<Val<'a>> {
    let ty = gnx_transforms::util::dtype(&like.info())?;
    let scalar = cx.emit(
        crate::ops::creation::constant(value),
        &[],
        gnx_transforms::util::info(&[], ty),
    );
    Ok(cx.emit(crate::ops::creation::full(), &[scalar], like.info()))
}

fn sub<'a>(cx: &'a Ctx<'a>, x: Val<'a>, y: Val<'a>) -> Val<'a> {
    binary(cx, sub_op(), x, y)
}

fn square<'a>(cx: &'a Ctx<'a>, x: Val<'a>) -> Val<'a> {
    unary(cx, square_op(), x)
}

fn exp<'a>(cx: &'a Ctx<'a>, x: Val<'a>) -> Val<'a> {
    unary(cx, exp_op(), x)
}

fn log<'a>(cx: &'a Ctx<'a>, x: Val<'a>) -> Val<'a> {
    unary(cx, log_op(), x)
}

fn rsqrt<'a>(cx: &'a Ctx<'a>, x: Val<'a>) -> Val<'a> {
    unary(cx, rsqrt_op(), x)
}

fn without_zero<'a>(cx: &'a Ctx<'a>, x: Val<'a>) -> gnx_transforms::TransformResult<Val<'a>> {
    Ok(cx.select(predicate(cx, eq_op(), x, x.zeros_like())?, one(cx, x), x))
}

defjvp!(AbsOperation, |cx, t, [x]| {
    if matches!(
        gnx_transforms::util::dtype(&x.info())?,
        gnx_array::DType::C64 | gnx_array::DType::C128
    ) {
        let real = cx.ans()?;
        let norm = without_zero(cx, real)?;
        let norm = cx.emit(convert_element_type_op(), &[norm], x.info());
        return Ok(cx.emit(
            real_op(),
            &[t * divide(cx, unary(cx, conj_op(), x), norm)],
            real.info(),
        ));
    }
    let positive = predicate(cx, ge_op(), x, x.zeros_like())?;
    Ok(cx.select(positive, t, negate(cx, t)))
});
defjvp!(TanOperation, |cx, t, _| {
    let ans = cx.ans()?;
    Ok(t * (one(cx, ans) + square(cx, ans)))
});
defjvp!(AsinOperation, |cx, t, [x]| Ok(t * rsqrt(
    cx,
    unary(cx, one_minus_square_op(), x)
)));
defjvp!(AcosOperation, |cx, t, [x]| Ok(negate(
    cx,
    t * rsqrt(cx, unary(cx, one_minus_square_op(), x))
)));
defjvp!(AtanOperation, |cx, t, [x]| Ok(divide(
    cx,
    t,
    one(cx, x) + square(cx, x)
)));
defjvp!(SinhOperation, |cx, t, [x]| Ok(t * unary(cx, cosh_op(), x)));
defjvp!(CoshOperation, |cx, t, [x]| Ok(t * unary(cx, sinh_op(), x)));
defjvp!(AsinhOperation, |cx, t, [x]| Ok(t * rsqrt(
    cx,
    one(cx, x) + square(cx, x)
)));
defjvp!(AcoshOperation, |cx, t, [x]| Ok(t
    * rsqrt(cx, sub(cx, x, one(cx, x)))
    * rsqrt(cx, x + one(cx, x))));
defjvp!(AtanhOperation, |cx, t, [x]| Ok(divide(
    cx,
    t,
    unary(cx, one_minus_square_op(), x)
)));
defjvp!(Exp2Operation, |cx, t, [x]| Ok(t
    * constant(cx, std::f64::consts::LN_2, x)?
    * cx.ans()?));
defjvp!(Log2Operation, |cx, t, [x]| Ok(divide(
    cx,
    t,
    x * constant(cx, std::f64::consts::LN_2, x)?
)));
defjvp!(OneMinusSquareOperation, |cx, t, [x]| Ok(negate(
    cx,
    (x + x) * t
)));

defjvp!(
    PowOperation,
    |cx, t, [x, y]| {
        let raw = gnx_transforms::util::dtype(&y.info())?;
        let y = cx.emit(convert_element_type_op(), &[y], x.info());
        let power = binary(cx, pow_op(), x, sub(cx, y, one(cx, y)));
        let product = if matches!(
            raw,
            gnx_array::DType::I8
                | gnx_array::DType::I16
                | gnx_array::DType::I32
                | gnx_array::DType::I64
                | gnx_array::DType::U8
                | gnx_array::DType::U16
                | gnx_array::DType::U32
                | gnx_array::DType::U64
        ) {
            let zero = y.zeros_like();
            cx.select(
                predicate(cx, eq_op(), y, zero)?,
                zero,
                without_zero(cx, y)? * power,
            )
        } else {
            y * power
        };
        Ok(t * product)
    },
    |cx, t, [x, _]| Ok(t * log(cx, without_zero(cx, x)?) * cx.ans()?)
);
defjvp!(IntegerPowOperation, |cx, t, [x]| {
    let n = cx.operation::<IntegerPowOperation>()?.exponent();
    let derivative = if n == 0 {
        return Ok(t.zeros_like());
    } else if let Some(n1) = n.checked_sub(1) {
        unary(cx, integer_pow(n1), x)
    } else {
        divide(cx, cx.ans()?, x)
    };
    Ok(t * constant(cx, n as f64, x)? * derivative)
});

fn in_strict_interval<'a>(
    cx: &'a Ctx<'a>,
    x: Val<'a>,
    low: Val<'a>,
    high: Val<'a>,
) -> gnx_transforms::TransformResult<Val<'a>> {
    let x = clamp_broadcast(cx, x)?;
    let low = clamp_broadcast(cx, low)?;
    let high = clamp_broadcast(cx, high)?;
    let below = predicate(cx, lt_op(), x, high)?;
    let above = predicate(cx, gt_op(), x, low)?;
    Ok(binary(cx, and_op(), below, above))
}

fn clamp_broadcast<'a>(
    cx: &'a Ctx<'a>,
    value: Val<'a>,
) -> gnx_transforms::TransformResult<Val<'a>> {
    if &value.info() == cx.out() {
        return Ok(value);
    }
    if !value.dims()?.is_empty() {
        return Err(TransformError::invalid(
            "clamp bound must be scalar or match the operand",
        ));
    }
    Ok(cx.emit(
        crate::ops::broadcast_in_dim_op(&[]),
        &[value],
        cx.out().clone(),
    ))
}

defjvp!(
    ClampOperation,
    |cx, t, [min, x, max]| {
        let t = clamp_broadcast(cx, t)?;
        Ok(cx.select(in_strict_interval(cx, min, x, max)?, t, t.zeros_like()))
    },
    |cx, t, [min, x, max]| Ok(cx.select(in_strict_interval(cx, x, min, max)?, t, t.zeros_like())),
    |cx, t, [_, x, max]| {
        let t = clamp_broadcast(cx, t)?;
        Ok(cx.select(
            predicate(cx, lt_op(), clamp_broadcast(cx, max)?, x)?,
            t,
            t.zeros_like(),
        ))
    }
);

deflinear!(ConjOperation, |cx, ct, [input]| {
    if matches!(
        gnx_transforms::util::dtype(&input.info())?,
        gnx_array::DType::C64 | gnx_array::DType::C128
    ) {
        Ok(cx.emit(conj_op(), &[ct], input.info()))
    } else {
        Ok(cx.emit(real_op(), &[ct], input.info()))
    }
});
deflinear!(RealOperation, |cx, ct, [input]| Ok(cx.emit(
    complex_op(),
    &[ct, ct.zeros_like()],
    input.info()
)));
deflinear!(ImagOperation, |cx, ct, [input]| Ok(cx.emit(
    complex_op(),
    &[ct.zeros_like(), negate(cx, ct)],
    input.info()
)));
deflinear!(
    ComplexOperation,
    |cx, ct, [re, _]| Ok(cx.emit(real_op(), &[ct], re.info())),
    |cx, ct, [_, im]| Ok(cx.emit(imag_op(), &[negate(cx, ct)], im.info()))
);
deflinear!(ReducePrecisionOperation, |cx, ct, _| Ok(cx.apply(&[ct])));

defjvp!(RoundOperation, None);
defjvp!(EqToOperation, None, None);
defjvp!(LtToOperation, None, None);
defjvp!(LeToOperation, None, None);
defjvp!(PopulationCountOperation, None);
defjvp!(MulhiOperation, None, None);
defjvp!(ShiftLeftOperation, None, None);
defjvp!(ShiftRightArithmeticOperation, None, None);
defjvp!(ShiftRightLogicalOperation, None, None);
defjvp!(BitcastConvertTypeOperation, None);

defjvp!(ErfOperation, |cx, t, [x]| Ok(t
    * constant(cx, 2. / std::f64::consts::PI.sqrt(), x)?
    * exp(cx, negate(cx, square(cx, x)))));
defjvp!(ErfcOperation, |cx, t, [x]| Ok(t
    * constant(cx, -2. / std::f64::consts::PI.sqrt(), x)?
    * exp(cx, negate(cx, square(cx, x)))));
defjvp!(ErfInvOperation, |cx, t, [x]| Ok(t
    * constant(cx, std::f64::consts::PI.sqrt() / 2., x)?
    * exp(cx, square(cx, cx.ans()?))));
defjvp!(LgammaOperation, |cx, t, [x]| Ok(t * unary(
    cx,
    digamma_op(),
    x
)));
defjvp!(DigammaOperation, |cx, t, [x]| Ok(t * binary(
    cx,
    polygamma_op(),
    one(cx, x),
    x
)));
defjvp!(
    PolygammaOperation,
    |_, _, _| Err(TransformError::invalid(
        "polygamma does not differentiate its order"
    )),
    |cx, t, [n, x]| Ok(t * binary(cx, polygamma_op(), n + one(cx, n), x))
);

fn gamma_dx<'a>(cx: &'a Ctx<'a>, a: Val<'a>, x: Val<'a>) -> Val<'a> {
    let term = sub(
        cx,
        sub(cx, sub(cx, a, one(cx, a)) * log(cx, x), x),
        unary(cx, lgamma_op(), a),
    );
    exp(cx, term)
}

defjvp!(
    IgammaOperation,
    |cx, t, [a, x]| Ok(t * binary(cx, igamma_grad_a_op(), a, x)),
    |cx, t, [a, x]| Ok(t * gamma_dx(cx, a, x))
);
defjvp!(
    IgammacOperation,
    |cx, t, [a, x]| Ok(negate(cx, t * binary(cx, igamma_grad_a_op(), a, x))),
    |cx, t, [a, x]| Ok(negate(cx, t * gamma_dx(cx, a, x)))
);
defjvp!(
    RegularizedIncompleteBetaOperation,
    |_, _, _| Err(TransformError::invalid(
        "betainc does not differentiate its shape parameters"
    )),
    |_, _, _| Err(TransformError::invalid(
        "betainc does not differentiate its shape parameters"
    )),
    |cx, t, [a, b, x]| {
        let g = |v| unary(cx, lgamma_op(), v);
        let beta = sub(cx, g(a) + g(b), g(a + b));
        let l = (sub(cx, b, one(cx, b)) * unary(cx, log1p_op(), negate(cx, x)))
            + sub(cx, a, one(cx, a)) * log(cx, x);
        Ok(t * exp(cx, sub(cx, l, beta)))
    }
);
defjvp!(BesselI0eOperation, |cx, t, [x]| {
    Ok(t * sub(
        cx,
        unary(cx, bessel_i1e_op(), x),
        unary(cx, sign_op(), x) * cx.ans()?,
    ))
});
defjvp!(BesselI1eOperation, |cx, t, [x]| {
    let epsilon = match gnx_transforms::util::dtype(&x.info())? {
        gnx_array::DType::F64 => f64::EPSILON,
        gnx_array::DType::F16 => 0.0009765625,
        gnx_array::DType::BF16 => 0.0078125,
        _ => f32::EPSILON as f64,
    };
    let eps = constant(cx, epsilon, x)?;
    let normal = predicate(cx, gt_op(), unary(cx, abs_op(), x), eps)?;
    let safe = cx.select(normal, x, eps);
    let partial = sub(
        cx,
        unary(cx, bessel_i0e_op(), safe),
        cx.ans()? * (unary(cx, sign_op(), safe) + divide(cx, one(cx, safe), safe)),
    );
    Ok(t * cx.select(normal, partial, constant(cx, 0.5, x)?))
});

defjvp!(NextafterOperation, |_, t, _| Ok(t), None);
defjvp!(RemOperation, |_, t, _| Ok(t), |cx, t, [x, y]| {
    let z = divide(cx, x, y);
    let magnitude = unary(cx, floor_op(), unary(cx, abs_op(), z));
    Ok(negate(cx, t) * unary(cx, sign_op(), z) * magnitude)
});

impl gnx_transforms::ad::JvpRule for PolynomialOperation {
    fn jvp(
        &self,
        b: &mut gnx_expr::pass::Builder,
        eqn: &gnx_expr::Eqn,
        inputs: &[gnx_transforms::ad::Dual],
        _: &[Vec<gnx_transforms::ad::Dual>],
    ) -> gnx_transforms::TransformResult<Vec<gnx_transforms::ad::Dual>> {
        let mut recipe = gnx_expr::pass::Builder::new();
        let metadata = &eqn.outputs()[0].1;
        let args: Vec<_> = inputs
            .iter()
            .map(|input| recipe.input(b.info(input.primal).clone()))
            .collect();
        let [x, coeffs @ ..] = args.as_slice() else {
            return Err(TransformError::invalid("polynomial needs an operand"));
        };
        let mut expanded = Vec::new();
        for &coefficient in coeffs {
            let value = if recipe.info(coefficient) == metadata {
                coefficient
            } else {
                if !gnx_transforms::util::dims(recipe.info(coefficient))?.is_empty() {
                    return Err(TransformError::invalid(
                        "non-scalar polynomial coefficient metadata",
                    ));
                }
                recipe.emit1(
                    crate::ops::broadcast_in_dim_op(&[]),
                    &[coefficient],
                    metadata.clone(),
                )
            };
            expanded.push(value);
        }
        let mut result = match expanded.pop() {
            Some(coef) => coef,
            None => gnx_transforms::util::zeros(&mut recipe, metadata),
        };
        for coefficient in expanded.into_iter().rev() {
            let product = gnx_transforms::util::mul(&mut recipe, result, *x);
            result = gnx_transforms::util::add(&mut recipe, product, coefficient);
        }
        gnx_transforms::ad::jvp::jvp_expr(b, &recipe.finish(vec![result]), &[], inputs)
    }
}
gnx_expr::register_op_interface!(dyn gnx_transforms::ad::JvpRule: PolynomialOperation);

use super::{TorchLowering, invalid, unsupported};
use gnx_expr::Eqn;
use gnx_math::ops::{AddOperation, MulOperation, SelectOperation, elementwise::*};
use std::io;
use tch::{Kind, TchError, Tensor};

pub(super) fn same_operands(a: &Tensor, b: &Tensor) -> io::Result<()> {
    if a.kind() != b.kind() || a.size() != b.size() {
        return Err(invalid("operands must have the same shape and dtype"));
    }
    Ok(())
}

// Signed integer keys: preserve positive encodings and reverse the order
// within negative encodings. Comparing the keys never performs FP arithmetic,
// so signaling NaNs, payloads and both zero signs survive unchanged.
fn total_key(x: &Tensor) -> Result<Tensor, TchError> {
    let (kind, mask) = match x.kind() {
        Kind::Half | Kind::BFloat16 => (Kind::Int16, i16::MAX as i64),
        Kind::Float => (Kind::Int, i32::MAX as i64),
        Kind::Double => (Kind::Int64, i64::MAX),
        _ => return Ok(x.shallow_clone()),
    };
    let bits = x.f_contiguous()?.f_view_copy_dtype(kind)?;
    bits.f_bitwise_xor(mask)?
        .f_where_self(&bits.f_lt(0)?, &bits)
}

fn numeric(k: Kind) -> io::Result<()> {
    if k == Kind::Bool {
        Err(unsupported("numeric operand required"))
    } else {
        Ok(())
    }
}
fn ordered(k: Kind) -> io::Result<()> {
    if matches!(k, Kind::ComplexFloat | Kind::ComplexDouble) {
        Err(unsupported("complex ordering is not supported"))
    } else {
        Ok(())
    }
}

// XLA integer division truncates toward zero. Its zero-divisor results are
// all ones (quotient) and the dividend (remainder). Signed MIN / -1 wraps.
// Select a safe divisor *before* invoking ATen, including during tracing.
fn integer_divrem(a: &Tensor, b: &Tensor, min: Option<i64>, rem: bool) -> Result<Tensor, TchError> {
    let zero = b.f_eq(0)?;
    let unsafe_divisor = match min {
        Some(min) => zero.f_logical_or(&a.f_eq(min)?.f_logical_and(&b.f_eq(-1)?)?)?,
        None => zero.shallow_clone(),
    };
    let safe = b.f_masked_fill(&unsafe_divisor, 1)?;
    if rem {
        a.f_where_self(&zero, &a.f_fmod_tensor(&safe)?)
    } else {
        a.f_div_tensor_mode(&safe, "trunc")?
            .f_masked_fill(&zero, if min.is_some() { -1 } else { 255 })
    }
}

fn divrem(a: &Tensor, b: &Tensor, rem: bool) -> Result<Tensor, TchError> {
    let min = match a.kind() {
        Kind::Uint8 => None,
        Kind::Int8 => Some(i8::MIN.into()),
        Kind::Int16 => Some(i16::MIN.into()),
        Kind::Int => Some(i32::MIN.into()),
        Kind::Int64 => Some(i64::MIN),
        _ if rem => return a.f_fmod_tensor(b),
        _ => return a.f_div(b),
    };
    integer_divrem(a, b, min, rem)
}

macro_rules! binary {
    ($op:ty, $a:ident, $b:ident, $body:block) => {
        rule!($op, |_op, _eqn, [$a, $b]| {
            same_operands($a, $b)?;
            $body
        });
    };
}
// add and mul also serve GNX's boolean OR and AND arithmetic.
rule!(AddOperation, |_op, _eqn, [a, b]| { a.f_add(b) });
rule!(MulOperation, |_op, _eqn, [a, b]| { a.f_mul(b) });
binary!(SubOperation, a, b, {
    numeric(a.kind())?;
    a.f_sub(b)
});
binary!(DivOperation, a, b, {
    numeric(a.kind())?;
    divrem(a, b, false)
});
binary!(RemOperation, a, b, {
    numeric(a.kind())?;
    if a.is_complex() {
        return Err(unsupported("complex remainder is not supported"));
    }
    divrem(a, b, true)
});
rule!(NegOperation, |_op, _eqn, [a]| {
    numeric(a.kind())?;
    a.f_neg()
});
rule!(AbsOperation, |_op, _eqn, [a]| {
    numeric(a.kind())?;
    a.f_abs()
});

fn extrema(a: &Tensor, b: &Tensor, max: bool) -> Result<Tensor, TchError> {
    let value = if max {
        a.f_maximum(b)?
    } else {
        a.f_minimum(b)?
    };
    if !a.is_floating_point() {
        return Ok(value);
    }
    // ATen's signed-zero tie depends on operand order. GNX/HLO orders
    // -0 below +0. Other equal values need no special handling.
    let tie = if max {
        b.f_where_self(&a.f_signbit()?, a)?
    } else {
        a.f_where_self(&a.f_signbit()?, b)?
    };
    tie.f_where_self(&a.f_eq_tensor(b)?, &value)
}
binary!(MinOperation, a, b, {
    ordered(a.kind())?;
    extrema(a, b, false)
});
binary!(MaxOperation, a, b, {
    ordered(a.kind())?;
    extrema(a, b, true)
});
macro_rules! total_compare {
    ($op:ty,$method:ident) => {
        binary!($op, a, b, {
            ordered(a.kind())?;
            total_key(a).and_then(|x| x.$method(&total_key(b)?))
        });
    };
}
total_compare!(EqToOperation, f_eq_tensor);
total_compare!(LtToOperation, f_lt_tensor);
total_compare!(LeToOperation, f_le_tensor);
binary!(EqOperation, a, b, { a.f_eq_tensor(b) });
binary!(NeOperation, a, b, { a.f_ne_tensor(b) });
binary!(LtOperation, a, b, {
    ordered(a.kind())?;
    a.f_lt_tensor(b)
});
binary!(LeOperation, a, b, {
    ordered(a.kind())?;
    a.f_le_tensor(b)
});
binary!(GtOperation, a, b, {
    ordered(a.kind())?;
    a.f_gt_tensor(b)
});
binary!(GeOperation, a, b, {
    ordered(a.kind())?;
    a.f_ge_tensor(b)
});
rule!(SelectOperation, |_op, _eqn, [pred, a, b]| {
    same_operands(a, b)?;
    if pred.kind() != Kind::Bool || !(pred.dim() == 0 || pred.size() == a.size()) {
        return Err(invalid(
            "select requires a scalar or matching boolean predicate",
        ));
    }
    a.f_where_self(pred, b)
});

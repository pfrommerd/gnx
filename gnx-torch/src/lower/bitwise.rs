use super::{elementwise::same_operands, *};
use gnx_math::ops::{bitwise::*, elementwise::MulhiOperation};
use tch::TchError;

fn bits(x: &Tensor) -> io::Result<i64> {
    Ok(match x.kind() {
        Kind::Int8 | Kind::Uint8 => 8,
        Kind::Int16 => 16,
        Kind::Int => 32,
        Kind::Int64 => 64,
        _ => return Err(unsupported("integer operand required")),
    })
}
macro_rules! binary {
    ($op:ty,$f:ident) => {
        rule!($op, |_op, _eqn, [a, b]| {
            same_operands(a, b)?;
            a.$f(b)
        });
    };
}
binary!(AndOperation, f_bitwise_and_tensor);
binary!(OrOperation, f_bitwise_or_tensor);
binary!(XorOperation, f_bitwise_xor_tensor);
rule!(NotOperation, |_op, _eqn, [x]| { x.f_bitwise_not() });

// Widen small integers so intermediate shifts never rely on C++ signed
// overflow or the platform's treatment of a shift larger than the word.
fn wide(x: &Tensor, w: i64) -> Result<Tensor, TchError> {
    let x = x.f_to_kind(Kind::Int64)?;
    if w == 64 {
        Ok(x)
    } else {
        x.f_bitwise_and((1i64 << w) - 1)
    }
}
fn shift(x: &Tensor, n: &Tensor, w: i64, mode: u8) -> Result<Tensor, TchError> {
    let n = n.f_to_kind(Kind::Int64)?;
    let count = n.f_clamp(0, w - 1)?;
    let bad = n.f_lt(0)?.f_logical_or(&n.f_ge(w)?)?;
    let a = wide(x, w)?;
    let t = match mode {
        0 => a.f_bitwise_left_shift(&count)?.f_masked_fill(&bad, 0)?,
        1 => {
            let signed = if x.kind() == Kind::Uint8 {
                x.f_to_kind(Kind::Int8)?
            } else {
                x.shallow_clone()
            }
            .f_to_kind(Kind::Int64)?;
            signed.f_bitwise_right_shift(&count.f_masked_fill(&bad, w - 1)?)?
        }
        _ => {
            let half = a
                .f_bitwise_right_shift_tensor_scalar(1)?
                .f_bitwise_and(i64::MAX)?;
            let y = half.f_bitwise_right_shift(&count.f_sub_scalar(1)?.f_clamp_min(0)?)?;
            a.f_where_self(&count.f_eq(0)?, &y)?
                .f_masked_fill(&bad, 0)?
        }
    };
    t.f_to_kind(x.kind())
}
macro_rules! sh {
    ($op:ty,$mode:literal) => {
        rule!($op, |_op, _eqn, [a, b]| {
            same_operands(a, b)?;
            shift(a, b, bits(a)?, $mode)
        });
    };
}
sh!(ShiftLeftOperation, 0);
sh!(ShiftRightArithmeticOperation, 1);
sh!(ShiftRightLogicalOperation, 2);
fn pop(mut x: Tensor) -> Result<Tensor, TchError> {
    x = x.f_sub(
        &x.f_bitwise_right_shift_tensor_scalar(1)?
            .f_bitwise_and(0x5555555555555555i64)?,
    )?;
    let m = 0x3333333333333333i64;
    x = x
        .f_bitwise_and(m)?
        .f_add(&x.f_bitwise_right_shift_tensor_scalar(2)?.f_bitwise_and(m)?)?;
    x = x
        .f_add(&x.f_bitwise_right_shift_tensor_scalar(4)?)?
        .f_bitwise_and(0x0f0f0f0f0f0f0f0fi64)?;
    x.f_mul_scalar(0x0101010101010101i64)?
        .f_bitwise_right_shift_tensor_scalar(56)?
        .f_bitwise_and(127)
}
rule!(PopulationCountOperation, |_op, _eqn, [x]| {
    wide(x, bits(x)?)
        .and_then(pop)
        .and_then(|v| v.f_to_kind(x.kind()))
});
rule!(ClzOperation, |_op, _eqn, [x]| {
    let w = bits(x)?;
    (|| {
        let mut a = wide(x, w)?;
        let mut i = 1;
        while i < w {
            a = a.f_bitwise_or_tensor(&a.f_bitwise_right_shift_tensor_scalar(i)?)?;
            i *= 2;
        }
        pop(a)?.f_rsub_scalar(w)?.f_to_kind(x.kind())
    })()
});
rule!(MulhiOperation, |_op, _eqn, [a, b]| {
    same_operands(a, b)?;
    let w = bits(a)?;
    (|| {
        let x = a.f_to_kind(Kind::Int64)?;
        let y = b.f_to_kind(Kind::Int64)?;
        if w < 64 {
            return x
                .f_mul(&y)?
                .f_bitwise_right_shift_tensor_scalar(w)?
                .f_to_kind(a.kind());
        }
        let mask = 0xffffffffi64;
        let hi = |t: &Tensor| t.f_bitwise_right_shift_tensor_scalar(32);
        let lo = |t: &Tensor| t.f_bitwise_and(mask);
        let (xh, xl, yh, yl) = (hi(&x)?, lo(&x)?, hi(&y)?, lo(&y)?);
        let carry = lo(&hi(&xl.f_mul(&yl)?)?)?;
        let mid = xh.f_mul(&yl)?.f_add(&carry)?;
        let cross = xl.f_mul(&yh)?.f_add(&lo(&mid)?)?;
        xh.f_mul(&yh)?.f_add(&hi(&mid)?)?.f_add(&hi(&cross)?)
    })()
});

use super::*;
use gnx_math::ops::{EyeOperation, RevOperation, creation::*, fft::*, indexing::*, shape::*};
use tch::TchError;

rule!(ConstantOperation, |op, eqn, []| {
    let (_, k) = result_layout(eqn)?;
    if metadata(eqn.outputs().next().unwrap().info_ref())?.dtype() == DType::U32 {
        return Ok(vec![
            Tensor::f_full(
                [],
                i64::from(op.value() as u32 as i32),
                (k, tracing_device()),
            )
            .map_err(io::Error::other)?,
        ]);
    }
    Tensor::f_full([], op.value(), (Kind::Double, tracing_device())).and_then(|v| v.f_to_kind(k))
});
rule!(FullOperation, |_op, eqn, [x]| {
    if x.dim() != 0 {
        return Err(invalid("full expects scalar"));
    }
    x.f_expand(result_layout(eqn)?.0, false)
});
rule!(EyeOperation, |_op, eqn, []| {
    let (s, k) = result_layout(eqn)?;
    Tensor::f_eye_m(s[0], s[1], (k, tracing_device()))
});
rule!(ScalarBoundOperation, |op, eqn, []| {
    let k = result_layout(eqn)?.1;
    let lo = op.lower();
    let bounds = match k {
        Kind::Int64 => (i64::MIN, i64::MAX),
        Kind::Int => (i32::MIN.into(), i32::MAX.into()),
        Kind::Int16 => (i16::MIN.into(), i16::MAX.into()),
        Kind::Int8 => (i8::MIN.into(), i8::MAX.into()),
        Kind::Uint8 => (0, 255),
        Kind::Bool => (0, 1),
        Kind::ComplexFloat | Kind::ComplexDouble => {
            return Err(unsupported("no complex scalar bounds"));
        }
        _ => {
            return Ok(vec![
                Tensor::f_full(
                    [],
                    if lo { f64::NEG_INFINITY } else { f64::INFINITY },
                    (k, tracing_device()),
                )
                .map_err(io::Error::other)?,
            ]);
        }
    };
    Tensor::f_full(
        [],
        if lo { bounds.0 } else { bounds.1 },
        (k, tracing_device()),
    )
});
rule!(IotaOperation, |op, eqn, []| {
    let (s, k) = result_layout(eqn)?;
    let mut shape = vec![1; s.len()];
    shape[op.axis()] = s[op.axis()];
    Tensor::f_arange(s[op.axis()], (Kind::Int64, tracing_device()))
        .and_then(|v| v.f_to_kind(k)?.f_reshape(shape)?.f_expand(s, false))
});
rule!(RevOperation, |op, _eqn, [x]| {
    x.f_flip(axes(&op.axes())?)
});
rule!(SliceOperation, |op, _eqn, [x]| {
    let mut v = x.shallow_clone();
    for (i, &(start, end, step)) in op.ranges().iter().enumerate() {
        let r = axes(&[start, end, step])?;
        if i >= x.dim() || r[1] > x.size()[i] {
            return Err(invalid("slice exceeds operand extent"));
        }
        v = v
            .f_slice(i as i64, r[0], r[1], r[2])
            .map_err(io::Error::other)?;
    }
    Ok::<_, TchError>(v)
});
rule!(FftOperation, |op, _eqn, [x]| {
    let s = axes(&op.lengths())?;
    if s.is_empty() {
        return Ok(vec![x.shallow_clone()]);
    }
    if s.len() > x.dim() {
        return Err(invalid("FFT rank exceeds operand rank"));
    }
    let d: Vec<_> = ((x.dim() - s.len()) as i64..x.dim() as i64).collect();
    match op.kind() {
        FftKind::Fft => x.f_fft_fftn(&s[..], &d[..], "backward"),
        FftKind::Ifft => x.f_fft_ifftn(&s[..], &d[..], "backward"),
        FftKind::Rfft => x.f_fft_rfftn(&s[..], &d[..], "backward"),
        FftKind::Irfft => x.f_fft_irfftn(&s[..], &d[..], "backward"),
    }
});
// Map output positions back to input coordinates. The fill stays a tensor:
// replacing it by a host scalar would specialize replay to the trace example.
impl TorchLowering for PadOperation {
    fn lower_torch(&self, eqn: Eqn<'_>, args: &[Tensor]) -> io::Result<Vec<Tensor>> {
        let [x, fill] = args else {
            return Err(invalid("pad expects two operands"));
        };
        Ok(vec![pad(x, fill, &self.padding(), &result_layout(eqn)?.0)?])
    }
}
crate::register_torch_lowering!(PadOperation);
pub(super) fn pad(
    x: &Tensor,
    fill: &Tensor,
    padding: &[(i64, i64, usize)],
    target: &[i64],
) -> io::Result<Tensor> {
    if target.len() != x.dim()
        || padding.len() != x.dim()
        || fill.dim() != 0
        || fill.kind() != x.kind()
    {
        return Err(invalid("invalid pad signature"));
    }
    let mut v = x.shallow_clone();
    for (axis, (&n, &(low, high, interior))) in x.size().iter().zip(padding).enumerate() {
        if low == 0 && high == 0 && interior == 0 && n == target[axis] {
            continue;
        }
        let step = i64::try_from(interior)
            .ok()
            .and_then(|i| i.checked_add(1))
            .ok_or_else(|| invalid("padding stride overflow"))?;
        let expected = i128::from(n)
            + i128::from((n - 1).max(0)) * i128::from(step - 1)
            + i128::from(low)
            + i128::from(high);
        if expected != i128::from(target[axis]) {
            return Err(invalid("incorrect padded extent"));
        }
        if n == 0 {
            let mut shape = v.size();
            shape[axis] = target[axis];
            v = fill.f_expand(shape, false).map_err(io::Error::other)?;
            continue;
        }
        v = (|| {
            let p = Tensor::f_arange(target[axis], (Kind::Int64, tracing_device()))?
                .f_sub_scalar(low)?;
            let src = p.f_floor_divide_scalar(step)?;
            let valid = src
                .f_ge(0)?
                .f_logical_and(&src.f_lt(n)?)?
                .f_logical_and(&p.f_remainder(step)?.f_eq(0)?)?;
            let mut mask_shape = vec![1; x.dim()];
            mask_shape[axis] = target[axis];
            v.f_index_select(axis as i64, &src.f_clamp(0, n - 1)?)?
                .f_where_self(&valid.f_reshape(mask_shape)?, fill)
        })()
        .map_err(io::Error::other)?;
    }
    Ok(v)
}
rule!(gnx_math::shard::Replicate, |_op, _eqn, [x]| { x.f_alias() });
rule!(gnx_math::shard::Constraint, |_op, eqn, [x]| {
    let spec = gnx_math::shard::Sharding::from_op_ref(eqn.op(), x.dim()).map_err(invalid)?;
    if !spec.is_replicated() {
        return Err(unsupported("named sharding axes require a device mesh"));
    }
    x.f_alias()
});
impl TorchLowering for ConcatenateOperation {
    fn lower_torch(&self, _: Eqn<'_>, x: &[Tensor]) -> io::Result<Vec<Tensor>> {
        if x.iter().any(|v| v.kind() != x[0].kind()) {
            return Err(invalid("concatenate dtype mismatch"));
        }
        Ok(vec![
            Tensor::f_cat(x, self.axis() as i64).map_err(io::Error::other)?,
        ])
    }
}

fn integer(k: Kind) -> bool {
    matches!(
        k,
        Kind::Uint8 | Kind::Int8 | Kind::Int16 | Kind::Int | Kind::Int64
    )
}
// Build row-major linear indices for the slice. Only extents are static;
// starts remain tensor arguments of the saved module (including wrapping).
fn indices(shape: &[i64], slice: &[i64], starts: &[Tensor]) -> io::Result<Tensor> {
    if shape.len() != slice.len()
        || starts.len() != shape.len()
        || starts.iter().any(|s| s.dim() != 0 || !integer(s.kind()))
        || shape.iter().zip(slice).any(|(n, m)| m > n)
    {
        return Err(invalid("invalid dynamic slice"));
    }
    (|| {
        let mut idx = Tensor::f_zeros([], (Kind::Int64, tracing_device()))?;
        for ((start, &n), &m) in starts.iter().zip(shape).zip(slice) {
            let t = start.f_to_kind(Kind::Int64)?;
            let t = t
                .f_add_scalar(n)?
                .f_where_self(&t.f_lt(0)?, &t)?
                .f_clamp(0, n - m)?;
            let axis = Tensor::f_arange(m, (Kind::Int64, tracing_device()))?.f_add(&t)?;
            idx = idx.f_mul_scalar(n)?.f_unsqueeze(-1)?.f_add(&axis)?;
        }
        idx.f_reshape([-1])
    })()
    .map_err(io::Error::other)
}
impl TorchLowering for DynamicSliceOperation {
    fn lower_torch(&self, eqn: Eqn<'_>, args: &[Tensor]) -> io::Result<Vec<Tensor>> {
        let [x, starts @ ..] = args else {
            return Err(invalid("missing slice operand"));
        };
        let shape = result_layout(eqn)?.0;
        let i = indices(&x.size(), &shape, starts)?;
        Ok(vec![
            x.f_reshape([-1])
                .and_then(|v| v.f_index_select(0, &i)?.f_reshape(shape))
                .map_err(io::Error::other)?,
        ])
    }
}
impl TorchLowering for DynamicUpdateSliceOperation {
    fn lower_torch(&self, _: Eqn<'_>, args: &[Tensor]) -> io::Result<Vec<Tensor>> {
        let [x, u, starts @ ..] = args else {
            return Err(invalid("missing update operand"));
        };
        let shape = x.size();
        let i = indices(&shape, &u.size(), starts)?;
        Ok(vec![
            x.f_reshape([-1])
                .and_then(|v| v.f_index_copy(0, &i, &u.f_reshape([-1])?)?.f_reshape(shape))
                .map_err(io::Error::other)?,
        ])
    }
}
impl TorchLowering for SelectNOperation {
    fn lower_torch(&self, _: Eqn<'_>, args: &[Tensor]) -> io::Result<Vec<Tensor>> {
        let [s, x, rest @ ..] = args else {
            return Err(invalid("expected selector and cases"));
        };
        if !(integer(s.kind()) || s.kind() == Kind::Bool && rest.len() <= 1)
            || !(s.dim() == 0 || s.size() == x.size())
        {
            return Err(invalid("invalid select_n selector"));
        }
        let mut out = x.shallow_clone();
        for (i, c) in rest.iter().enumerate() {
            super::elementwise::same_operands(x, c)?;
            out = s
                .f_ge(i as i64 + 1)
                .and_then(|p| c.f_where_self(&p, &out))
                .map_err(io::Error::other)?;
        }
        Ok(vec![out])
    }
}
crate::register_torch_lowering!(
    ConcatenateOperation,
    DynamicSliceOperation,
    DynamicUpdateSliceOperation,
    SelectNOperation
);

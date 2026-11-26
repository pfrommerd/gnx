use super::{TorchLowering, axes, emit_results, invalid, layout, metadata};
use gnx_base::ops::reduction::ReduceOperation;
use gnx_expr::Eqn;
use gnx_math::ops::reduction::ReduceWindowOperation;
use std::io;
use tch::Tensor;

// GNX checks that the reducer is closed and takes (lhs..., rhs...).
fn validate(eqn: Eqn<'_>, operands: &[Tensor], initial: &[Tensor], out: &[i64]) -> io::Result<()> {
    let shape = operands[0].size();
    for ((x, init), output) in operands.iter().zip(initial).zip(eqn.outputs()) {
        let (dims, kind) = layout(&metadata(output.info_ref())?)?;
        if x.size() != shape
            || init.dim() != 0
            || x.kind() != kind
            || init.kind() != kind
            || dims != out
        {
            return Err(invalid(
                "reduce operand, initializer, or result metadata mismatch",
            ));
        }
    }
    Ok(())
}
fn identities(initial: &[Tensor], out: &[i64]) -> io::Result<Vec<Tensor>> {
    initial
        .iter()
        .map(|x| x.f_expand(out, false).map_err(io::Error::other))
        .collect()
}

// Each lane is arranged as [output axes..., reduction axes...]. Fold one
// row per output position with the same variadic scalar region in both ops.
fn fold(
    eqn: Eqn<'_>,
    lanes: &[Tensor],
    initial: &[Tensor],
    out: &[i64],
) -> io::Result<Vec<Tensor>> {
    let rows = out
        .iter()
        .try_fold(1i64, |p, &n| p.checked_mul(n))
        .ok_or_else(|| invalid("reduce output size overflow"))?;
    let matrices = lanes
        .iter()
        .map(|x| x.f_reshape([rows, -1]))
        .collect::<Result<Vec<_>, _>>()
        .map_err(io::Error::other)?;
    let body = super::control_flow::region(eqn, 0)?;
    let mut outputs: Vec<Vec<Tensor>> = (0..lanes.len()).map(|_| Vec::new()).collect();
    for row in 0..rows {
        let lanes = matrices
            .iter()
            .map(|x| x.f_select(0, row))
            .collect::<Result<Vec<_>, _>>()
            .map_err(io::Error::other)?;
        let mut acc: Vec<_> = initial.iter().map(Tensor::shallow_clone).collect();
        for column in 0..lanes[0].size()[0] {
            for x in &lanes {
                acc.push(x.f_select(0, column).map_err(io::Error::other)?);
            }
            acc = emit_results(body, &acc)?;
        }
        for (ys, y) in outputs.iter_mut().zip(acc) {
            ys.push(y);
        }
    }
    outputs
        .into_iter()
        .map(|rows| {
            Tensor::f_stack(&rows, 0)
                .and_then(|v| v.f_reshape(out))
                .map_err(io::Error::other)
        })
        .collect()
}
impl TorchLowering for ReduceOperation {
    fn lower_torch(&self, eqn: Eqn<'_>, inputs: &[Tensor]) -> io::Result<Vec<Tensor>> {
        let (operands, initial) = inputs.split_at(eqn.outputs().len());
        let shape = operands[0].size();
        let reduced = self.axes();
        if reduced.iter().any(|&a| a >= shape.len()) {
            return Err(invalid("reduce axis outside operand rank"));
        }
        let kept: Vec<_> = (0..shape.len()).filter(|a| !reduced.contains(a)).collect();
        let out: Vec<_> = kept.iter().map(|&a| shape[a]).collect();
        validate(eqn, operands, initial, &out)?;
        if reduced.is_empty() {
            return Ok(operands.iter().map(Tensor::shallow_clone).collect());
        }
        if shape.contains(&0) {
            return identities(initial, &out);
        }
        let perm = axes(&[&kept[..], &reduced[..]].concat())?;
        let lanes = operands
            .iter()
            .map(|x| x.f_permute(&perm))
            .collect::<Result<Vec<_>, _>>()
            .map_err(io::Error::other)?;
        fold(eqn, &lanes, initial, &out)
    }
}
impl TorchLowering for ReduceWindowOperation {
    fn lower_torch(&self, eqn: Eqn<'_>, inputs: &[Tensor]) -> io::Result<Vec<Tensor>> {
        let (operands, initial) = inputs.split_at(eqn.outputs().len());
        let shape = operands[0].size();
        let w = self.window();
        let rank = shape.len();
        if rank != w.dimensions.len() {
            return Err(invalid("reduce_window operand rank mismatch"));
        }
        let mut padded = Vec::new();
        let mut spans = Vec::new();
        let mut padding = Vec::new();
        let mut out = Vec::new();
        for (i, &n) in shape.iter().enumerate() {
            let (lo, hi) = w.padding[i];
            let base = if n == 0 {
                0
            } else {
                1 + i128::from(n - 1) * (w.base_dilation[i] as i128)
            };
            let extent = base + i128::from(lo) + i128::from(hi);
            let span = (w.dimensions[i] as i128 - 1)
                .checked_mul(w.window_dilation[i] as i128)
                .and_then(|n| n.checked_add(1))
                .ok_or_else(|| invalid("window span overflow"))?;
            let size = if extent < span {
                0
            } else {
                1 + (extent - span) / (w.strides[i] as i128)
            };
            out.push(i64::try_from(size).map_err(invalid)?);
            padded.push(extent);
            spans.push(span);
            padding.push((lo, hi, w.base_dilation[i] - 1));
        }
        validate(eqn, operands, initial, &out)?;
        if out.contains(&0) || shape.contains(&0) {
            return identities(initial, &out);
        }
        let padded = padded
            .into_iter()
            .map(i64::try_from)
            .collect::<Result<Vec<_>, _>>()
            .map_err(invalid)?;
        let mut lanes = Vec::new();
        for (x, init) in operands.iter().zip(initial) {
            let mut v = super::structural::pad(x, init, &padding, &padded)?;
            for (i, &span) in spans.iter().enumerate() {
                let span = i64::try_from(span).map_err(invalid)?;
                let stride = i64::try_from(w.strides[i]).map_err(invalid)?;
                let step = i64::try_from(w.window_dilation[i]).map_err(invalid)?;
                // unfold appends the window axis, keeping output axes in place.
                v = v
                    .f_unfold(i as i64, span, stride)
                    .and_then(|v| v.f_slice(-1, 0, span, step))
                    .map_err(io::Error::other)?;
            }
            lanes.push(v);
        }
        fold(eqn, &lanes, initial, &out)
    }
}
crate::register_torch_lowering!(ReduceOperation, ReduceWindowOperation);

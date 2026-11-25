use super::*;
use gnx_math::ops::ordering::*;
use tch::TchError;
rule!(CumulativeOperation, |op, _eqn, [x]| {
    let axis = op.axis() as i64;
    let flip = |x: &Tensor| {
        if op.reverse() {
            x.f_flip([axis])
        } else {
            Ok(x.shallow_clone())
        }
    };
    (|| {
        let v = flip(x)?;
        let y = match op.kind() {
            CumulativeKind::Sum | CumulativeKind::Product if v.kind() == Kind::Bool => {
                let i = v.f_to_kind(Kind::Int64)?;
                if op.kind() == CumulativeKind::Sum {
                    i.f_cummax(axis)?.0
                } else {
                    i.f_cummin(axis)?.0
                }
                .f_to_kind(Kind::Bool)?
            }
            CumulativeKind::Sum => v.f_cumsum(axis, v.kind())?,
            CumulativeKind::Product => v.f_cumprod(axis, v.kind())?,
            CumulativeKind::Min => v.f_cummin(axis)?.0,
            CumulativeKind::Max => v.f_cummax(axis)?.0,
            CumulativeKind::LogSumExp => v.f_logcumsumexp(axis)?,
        };
        flip(&y)
    })()
});
impl TorchLowering for SortOperation {
    fn lower_torch(&self, _: Eqn<'_>, inputs: &[Tensor]) -> io::Result<Vec<Tensor>> {
        if self.axis() >= inputs[0].dim() || inputs.iter().any(|v| v.size() != inputs[0].size()) {
            return Err(invalid("invalid sort dimensions"));
        }
        let axis = self.axis() as i64;
        (|| {
            let mut perm: Option<Tensor> = None;
            let mut add = |v: &Tensor| -> Result<(), TchError> {
                let k = match &perm {
                    Some(p) => v.f_gather(axis, p, false)?,
                    None => v.shallow_clone(),
                };
                let p = k.f_argsort_stable(true, axis, false)?;
                perm = Some(match &perm {
                    Some(old) => old.f_gather(axis, &p, false)?,
                    None => p,
                });
                Ok(())
            };
            for key in inputs[..self.keys()].iter().rev() {
                if key.is_complex() {
                    add(&key.f_imag()?)?;
                    add(&key.f_real()?)?
                } else {
                    add(key)?
                }
            }
            inputs
                .iter()
                .map(|v| v.f_gather(axis, perm.as_ref().expect("checked sort keys"), false))
                .collect::<Result<Vec<_>, TchError>>()
        })()
        .map_err(io::Error::other)
    }
}
fn top(x: &Tensor, axis: usize, k: i64, largest: bool) -> io::Result<Vec<Tensor>> {
    if axis >= x.dim() || k < 0 || k > x.size()[axis] {
        return Err(invalid("invalid top-k axis or count"));
    }
    let axis = axis as i64;
    (|| {
        let idx = x
            .f_argsort_stable(true, axis, largest)?
            .f_narrow(axis, 0, k)?;
        Ok::<_, TchError>(vec![
            x.f_gather(axis, &idx, false)?,
            idx.f_to_kind(Kind::Int)?,
        ])
    })()
    .map_err(io::Error::other)
}
impl TorchLowering for TopKOperation {
    fn lower_torch(&self, _: Eqn<'_>, args: &[Tensor]) -> io::Result<Vec<Tensor>> {
        let x = &args[0];
        top(
            x,
            x.dim().saturating_sub(1),
            self.k().try_into().map_err(invalid)?,
            true,
        )
    }
}
impl TorchLowering for ApproxTopKOperation {
    fn lower_torch(&self, eqn: Eqn<'_>, args: &[Tensor]) -> io::Result<Vec<Tensor>> {
        let x = &args[0];
        let n = if self.aggregate() {
            i64::try_from(self.k()).map_err(invalid)?
        } else {
            let output = eqn.outputs().next();
            let output = output.ok_or_else(|| invalid("missing approx_top_k result"))?;
            *layout(&metadata(output.info_ref())?)?
                .0
                .get(self.axis())
                .ok_or_else(|| invalid("invalid approx_top_k axis"))?
        };
        top(x, self.axis(), n, self.largest())
    }
}
crate::register_torch_lowering!(SortOperation, TopKOperation, ApproxTopKOperation);

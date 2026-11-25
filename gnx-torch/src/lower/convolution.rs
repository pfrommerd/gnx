use super::*;
use gnx_math::ops::contraction::{ConvGeneralDilatedOperation, DotDimensions};

impl TorchLowering for ConvGeneralDilatedOperation {
    fn lower_torch(&self, eqn: Eqn<'_>, args: &[Tensor]) -> io::Result<Vec<Tensor>> {
        let [x, w] = args else {
            return Err(invalid("convolution needs input and kernel"));
        };
        let p = self.parameters();
        let r = p.strides.len();
        let rank = r + 2;
        let order = |a, b, s: &[usize]| {
            let mut perm = vec![a, b];
            perm.extend_from_slice(s);
            if perm.len() != rank
                || perm
                    .iter()
                    .enumerate()
                    .any(|(i, &v)| v >= rank || perm[..i].contains(&v))
            {
                return Err(invalid("invalid convolution axis permutation"));
            }
            axes(&perm)
        };
        let x = x
            .f_permute(order(p.input.batch, p.input.feature, &p.input.spatial)?)
            .map_err(io::Error::other)?;
        let w = w
            .f_permute(order(
                p.kernel.output_feature,
                p.kernel.input_feature,
                &p.kernel.spatial,
            )?)
            .map_err(io::Error::other)?;
        let dst = order(p.output.batch, p.output.feature, &p.output.spatial)?;
        let mut inv = vec![0; rank];
        for (i, &j) in dst.iter().enumerate() {
            inv[j as usize] = i as i64;
        }
        let mut padding = vec![(0, 0, 0); 2];
        let mut shape = x.size();
        let ws = w.size();
        let mut windows = vec![];
        let mut extents = vec![];
        for i in 0..r {
            let (lo, hi) = p.padding[i];
            let dil = p.lhs_dilation[i] - 1;
            let n = shape[2 + i];
            let expanded = i128::from(n)
                + i128::from((n - 1).max(0)) * (dil as i128)
                + i128::from(lo)
                + i128::from(hi);
            shape[2 + i] = i64::try_from(expanded).map_err(invalid)?;
            if shape[2 + i] < 0 || ws[2 + i] == 0 {
                return Err(invalid("invalid padded convolution extent"));
            }
            padding.push((lo, hi, dil));
            let eff = 1 + i128::from(ws[2 + i] - 1) * (p.rhs_dilation[i] as i128);
            let win = i64::try_from(eff).map_err(invalid)?;
            windows.push(win);
            extents.push(if win > shape[2 + i] {
                0
            } else {
                (shape[2 + i] - win) / i64::try_from(p.strides[i]).map_err(invalid)? + 1
            });
        }
        let (fg, bg) = (
            i64::try_from(p.feature_groups).map_err(invalid)?,
            i64::try_from(p.batch_groups).map_err(invalid)?,
        );
        if shape[0] % bg != 0
            || shape[1] % fg != 0
            || shape[1] / fg != ws[1]
            || ws[0] % fg != 0
            || ws[0] % bg != 0
            || fg > 1 && bg > 1
        {
            return Err(invalid("incompatible convolution groups"));
        }
        let n = shape[0] / bg;
        let groups = fg.max(bg);
        let oc = ws[0] / groups;
        let (expected, k) = result_layout(eqn)?;
        let mut out = vec![n, ws[0]];
        out.extend(&extents);
        if inv
            .iter()
            .map(|&i| out[i as usize])
            .ne(expected.iter().copied())
        {
            return Err(invalid("incorrect convolution result shape"));
        }
        if out.contains(&0) {
            return Ok(vec![
                Tensor::f_zeros(expected, (k, tracing_device())).map_err(io::Error::other)?,
            ]);
        }
        let zero = Tensor::f_zeros([], (x.kind(), tracing_device())).map_err(io::Error::other)?;
        let mut x = super::structural::pad(&x, &zero, &padding, &shape)?;
        for (i, &win) in windows.iter().enumerate() {
            x = x
                .f_unfold(
                    (2 + i) as i64,
                    win,
                    i64::try_from(p.strides[i]).map_err(invalid)?,
                )
                .and_then(|v| v.f_slice((rank + i) as i64, 0, win, p.rhs_dilation[i] as i64))
                .map_err(io::Error::other)?;
        }
        let dims = DotDimensions {
            lhs_batch: vec![],
            rhs_batch: vec![],
            lhs_contracting: std::iter::once(1).chain(rank..rank + r).collect(),
            rhs_contracting: (1..rank).collect(),
        };
        let mut pieces = vec![];
        for g in 0..groups {
            let x = x
                .f_narrow(0, if bg > 1 { g * n } else { 0 }, n)
                .and_then(|v| v.f_narrow(1, if fg > 1 { g * ws[1] } else { 0 }, ws[1]))
                .map_err(io::Error::other)?;
            let w = w.f_narrow(0, g * oc, oc).map_err(io::Error::other)?;
            pieces.push(super::contraction::dot(&x, &w, &dims, k)?);
        }
        let y = Tensor::f_cat(&pieces, -1)
            .and_then(|v| v.f_movedim_int(-1, 1)?.f_permute(inv))
            .map_err(io::Error::other)?;
        Ok(vec![y])
    }
}
crate::register_torch_lowering!(ConvGeneralDilatedOperation);

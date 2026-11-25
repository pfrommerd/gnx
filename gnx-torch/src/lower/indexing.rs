use super::*;
use gnx_math::ops::indexing::*;
use tch::TchError;

fn grid(shape: &[i64], axis: usize) -> Result<Tensor, TchError> {
    let mut s = vec![1; shape.len()];
    s[axis] = shape[axis];
    Tensor::f_arange(shape[axis], (Kind::Int64, tracing_device()))?.f_reshape(s)
}
struct Window<'a> {
    offset: &'a [usize],
    collapsed: &'a [usize],
    map: &'a [usize],
    operand_batch: &'a [usize],
    index_batch: &'a [usize],
}
impl Window<'_> {
    // For every gathered or scattered element, compute a linear operand index
    // and validity bit without converting any index value into a Rust scalar.
    fn positions(
        &self,
        shape: &[i64],
        out: &[i64],
        indices: &Tensor,
        sizes: Option<&[usize]>,
        clip: bool,
    ) -> io::Result<(Tensor, Tensor)> {
        let ix = indices.size();
        let unique = |a: &[usize], n: usize| {
            a.iter()
                .enumerate()
                .all(|(i, &v)| v < n && !a[..i].contains(&v))
        };
        if !matches!(
            indices.kind(),
            Kind::Int8 | Kind::Uint8 | Kind::Int16 | Kind::Int | Kind::Int64
        ) || ix.last() != Some(&(self.map.len() as i64))
            || !unique(self.offset, out.len())
            || !unique(self.map, shape.len())
            || !unique(self.collapsed, shape.len())
            || !unique(self.operand_batch, shape.len())
            || !unique(self.index_batch, ix.len().saturating_sub(1))
            || self.operand_batch.len() != self.index_batch.len()
            || self
                .operand_batch
                .iter()
                .any(|a| self.collapsed.contains(a) || self.map.contains(a))
            || sizes.is_some_and(|s| s.len() != shape.len())
        {
            return Err(invalid("invalid index dimension numbers"));
        }
        let nonwindow: Vec<_> = (0..out.len())
            .filter(|a| !self.offset.contains(a))
            .collect();
        let win: Vec<_> = (0..shape.len())
            .filter(|a| !self.collapsed.contains(a) && !self.operand_batch.contains(a))
            .collect();
        if nonwindow.len() + 1 != ix.len()
            || win.len() != self.offset.len()
            || nonwindow.iter().enumerate().any(|(i, &a)| out[a] != ix[i])
        {
            return Err(invalid("invalid gather/scatter window shape"));
        }
        let mut broadcast = vec![1; out.len()];
        for (i, &a) in nonwindow.iter().enumerate() {
            broadcast[a] = ix[i]
        }
        let mut linear =
            Tensor::f_zeros(out, (Kind::Int64, tracing_device())).map_err(io::Error::other)?;
        let mut valid =
            Tensor::f_ones(out, (Kind::Bool, tracing_device())).map_err(io::Error::other)?;
        for (axis, &n) in shape.iter().enumerate() {
            let offset = win.iter().position(|&a| a == axis).map(|i| self.offset[i]);
            let extent = offset.map_or(1, |a| out[a]);
            if sizes.is_some_and(|s| s[axis] as i64 != extent) || extent > n {
                return Err(invalid("invalid indexed slice extent"));
            }
            let batch = self
                .operand_batch
                .iter()
                .position(|&a| a == axis)
                .map(|i| self.index_batch[i]);
            if batch.is_some_and(|i| ix[i] != n) {
                return Err(invalid("indexed batch size mismatch"));
            }
            let (pos, ok) = (|| {
                let mut ok = valid.shallow_clone();
                let mut coord = if let Some(i) = self.map.iter().position(|&a| a == axis) {
                    let start = indices
                        .f_select(-1, i as i64)?
                        .f_to_kind(Kind::Int64)?
                        .f_reshape(&broadcast)?;
                    if sizes.is_some() {
                        ok = ok
                            .f_logical_and(&start.f_ge(0)?)?
                            .f_logical_and(&start.f_le(n - extent)?)?;
                    }
                    if clip || sizes.is_some() {
                        start.f_clamp(0, n - extent)?
                    } else {
                        start
                    }
                } else if let Some(i) = batch {
                    grid(out, nonwindow[i])?
                } else {
                    Tensor::f_zeros([], (Kind::Int64, tracing_device()))?
                };
                if let Some(a) = offset {
                    coord = coord.f_add(&grid(out, a)?)?
                }
                if sizes.is_none() {
                    ok = ok
                        .f_logical_and(&coord.f_ge(0)?)?
                        .f_logical_and(&coord.f_lt(n)?)?;
                }
                Ok::<_, TchError>((linear.f_mul_scalar(n)?.f_add(&coord)?, ok))
            })()
            .map_err(io::Error::other)?;
            linear = pos;
            valid = ok;
        }
        Ok((linear, valid))
    }
}
impl TorchLowering for GatherOperation {
    fn lower_torch(&self, e: Eqn<'_>, args: &[Tensor]) -> io::Result<Vec<Tensor>> {
        let [x, ix, fill @ ..] = args else {
            return Err(invalid("missing gather input"));
        };
        let p = self.parameters();
        let d = &p.dimensions;
        let out = result_layout(e)?.0;
        let (pos, ok) = Window {
            offset: &d.offset,
            collapsed: &d.collapsed,
            map: &d.start_index_map,
            operand_batch: &d.operand_batch,
            index_batch: &d.index_batch,
        }
        .positions(&x.size(), &out, ix, Some(&p.slice_sizes), true)?;
        let y = (|| {
            let y = x
                .f_reshape([-1])?
                .f_index_select(0, &pos.f_reshape([-1])?)?
                .f_reshape(&out)?;
            if let [fill] = fill {
                y.f_where_self(&ok, fill)
            } else {
                Ok(y)
            }
        })()
        .map_err(io::Error::other)?;
        Ok(vec![y])
    }
}
impl TorchLowering for ScatterOperation {
    fn lower_torch(&self, e: Eqn<'_>, args: &[Tensor]) -> io::Result<Vec<Tensor>> {
        let [x, ix, updates] = args else {
            return Err(invalid("expected scatter operands"));
        };
        let p = self.parameters();
        let d = &p.dimensions;
        let shape = x.size();
        let (pos, ok) = Window {
            offset: &d.update_window,
            collapsed: &d.inserted_window,
            map: &d.index_map,
            operand_batch: &d.operand_batch,
            index_batch: &d.index_batch,
        }
        .positions(
            &shape,
            &updates.size(),
            ix,
            None,
            p.mode == BoundsMode::Clip,
        )?;
        // Send dropped updates to a disposable trailing slot (keeps all shapes static).
        let n = i64::try_from(x.numel()).map_err(invalid)?;
        let pos = pos
            .f_masked_fill(&ok.f_logical_not().map_err(io::Error::other)?, n)
            .and_then(|p| p.f_reshape([-1]))
            .map_err(io::Error::other)?;
        let mut flat = x
            .f_reshape([-1])
            .and_then(|v| {
                Tensor::f_cat(&[v, Tensor::f_zeros([1], (x.kind(), tracing_device()))?], 0)
            })
            .map_err(io::Error::other)?;
        let u = updates.f_reshape([-1]).map_err(io::Error::other)?;
        flat = match self.update() {
            ScatterUpdate::Apply => {
                let body = super::control_flow::region(e, 0)?;
                for i in 0..pos.size()[0] {
                    let idx = pos.f_narrow(0, i, 1).map_err(io::Error::other)?;
                    let original = flat
                        .f_index_select(0, &idx)
                        .and_then(|v| v.f_reshape([]))
                        .map_err(io::Error::other)?;
                    let val = emit(body, &[original])?;
                    flat = flat
                        .f_index_copy(0, &idx, &val.f_reshape([1]).map_err(io::Error::other)?)
                        .map_err(io::Error::other)?;
                }
                Ok(flat)
            }
            ScatterUpdate::Replace => flat.f_index_copy(0, &pos, &u),
            ScatterUpdate::Add => flat.f_index_add(0, &pos, &u),
            ScatterUpdate::Subtract => u.f_neg().and_then(|u| flat.f_index_add(0, &pos, &u)),
            mode => flat.f_internal_scatter_reduce(
                0,
                &pos,
                &u,
                match mode {
                    ScatterUpdate::Min => "amin",
                    ScatterUpdate::Max => "amax",
                    _ => "prod",
                },
                true,
            ),
        }
        .map_err(io::Error::other)?;
        Ok(vec![
            flat.f_narrow(0, 0, n)
                .and_then(|v| v.f_reshape(shape))
                .map_err(io::Error::other)?,
        ])
    }
}
crate::register_torch_lowering!(GatherOperation, ScatterOperation);

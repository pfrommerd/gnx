use super::*;
use gnx_math::ops::contraction::*;

fn free(rank: usize, batch: &[usize], contract: &[usize]) -> io::Result<Vec<usize>> {
    let mut used = vec![false; rank];
    for &i in batch.iter().chain(contract) {
        if i >= rank || std::mem::replace(&mut used[i], true) {
            return Err(invalid("invalid or overlapping dot axes"));
        }
    }
    Ok((0..rank).filter(|&i| !used[i]).collect())
}
fn product(shape: &[i64], axes: &[usize]) -> io::Result<i64> {
    axes.iter().try_fold(1i64, |p, &i| {
        p.checked_mul(shape[i])
            .ok_or_else(|| invalid("dot extent overflow"))
    })
}
pub(super) fn dot(x: &Tensor, y: &Tensor, d: &DotDimensions, k: Kind) -> io::Result<Tensor> {
    let (a, b) = (x.size(), y.size());
    let (lb, rb, lc, rc) = (
        &d.lhs_batch,
        &d.rhs_batch,
        &d.lhs_contracting,
        &d.rhs_contracting,
    );
    let lf = free(a.len(), lb, lc)?;
    let rf = free(b.len(), rb, rc)?;
    for (l, r) in [(lb, rb), (lc, rc)] {
        if l.len() != r.len() || l.iter().zip(r).any(|(&i, &j)| a[i] != b[j]) {
            return Err(invalid("paired dot dimensions differ"));
        }
    }
    let perm = |u: &[usize], v: &[usize], w: &[usize]| axes(&[u, v, w].concat());
    let (lp, rp) = (perm(lb, &lf, lc)?, perm(rb, rc, &rf)?);
    let (batch, m, n, inner) = (
        product(&a, lb)?,
        product(&a, &lf)?,
        product(&b, &rf)?,
        product(&a, lc)?,
    );
    let out: Vec<_> = lb
        .iter()
        .chain(&lf)
        .map(|&i| a[i])
        .chain(rf.iter().map(|&i| b[i]))
        .collect();
    (|| {
        let acc = if k == Kind::Bool { Kind::Int64 } else { k };
        let lhs = x
            .f_to_kind(acc)?
            .f_permute(lp)?
            .f_reshape([batch, m, inner])?;
        let rhs = y
            .f_to_kind(acc)?
            .f_permute(rp)?
            .f_reshape([batch, inner, n])?;
        lhs.f_bmm(&rhs)?.f_to_kind(k)?.f_reshape(out)
    })()
    .map_err(io::Error::other)
}
impl TorchLowering for DotGeneralOperation {
    fn lower_torch(&self, eqn: &Eqn, args: &[Tensor]) -> io::Result<Vec<Tensor>> {
        let [a, b] = args else {
            return Err(invalid("dot expects two arguments"));
        };
        Ok(vec![dot(a, b, self.dimensions(), result_layout(eqn)?.1)?])
    }
}
fn scaled(x: &Tensor, s: Option<&Tensor>, contract: &[usize]) -> io::Result<Tensor> {
    let v = x.f_to_kind(Kind::BFloat16).map_err(io::Error::other)?;
    let Some(s) = s else { return Ok(v) };
    if x.dim() != s.dim() {
        return Err(invalid("scale rank mismatch"));
    }
    let mut small = vec![];
    let mut large = vec![];
    for (i, (n, m)) in x.size().into_iter().zip(s.size()).enumerate() {
        small.push(m);
        large.push(m);
        if contract.contains(&i) {
            if m <= 0 || n % m != 0 || n / m < 2 {
                return Err(invalid("invalid contracting scale extent"));
            }
            small.push(1);
            large.push(n / m);
        } else if n != m {
            return Err(invalid("noncontracting scale extent mismatch"));
        }
    }
    s.f_to_kind(Kind::BFloat16)
        .and_then(|s| {
            s.f_reshape(small)?
                .f_expand(large, false)?
                .f_reshape(x.size())?
                .f_mul(&v)
        })
        .map_err(io::Error::other)
}
impl TorchLowering for ScaledDotOperation {
    fn lower_torch(&self, eqn: &Eqn, args: &[Tensor]) -> io::Result<Vec<Tensor>> {
        let [a, b, rest @ ..] = args else {
            return Err(invalid("missing scaled dot arguments"));
        };
        let mut scales = rest.iter();
        let d = self.dimensions();
        let a = scaled(
            a,
            if self.lhs_scaled() {
                scales.next()
            } else {
                None
            },
            &d.lhs_contracting,
        )?;
        let b = scaled(
            b,
            if self.rhs_scaled() {
                scales.next()
            } else {
                None
            },
            &d.rhs_contracting,
        )?;
        Ok(vec![dot(&a, &b, d, result_layout(eqn)?.1)?])
    }
}
fn ragged(x: &Tensor, g: &Tensor, axis: usize, prefix: &[usize]) -> io::Result<Tensor> {
    let gs = g.size();
    let Some((&count, pdims)) = gs.split_last() else {
        return Err(invalid("missing group axis"));
    };
    let xs = x.size();
    if axis >= xs.len()
        || !pdims.is_empty()
            && (pdims.len() != prefix.len()
                || prefix.iter().zip(pdims).any(|(&i, n)| xs.get(i) != Some(n)))
    {
        return Err(invalid("invalid ragged group sizes"));
    }
    let mut mapping = if pdims.is_empty() {
        vec![]
    } else {
        prefix.iter().map(|i| i + 1).collect()
    };
    mapping.push(0);
    let mut perm: Vec<_> = (0..mapping.len()).collect();
    perm.sort_by_key(|&i| mapping[i]);
    let mut s = vec![1; xs.len() + 1];
    for (i, &d) in mapping.iter().enumerate() {
        s[d] = gs[i];
    }
    s[0] = count;
    let mut pos = vec![1; xs.len() + 1];
    pos[axis + 1] = xs[axis];
    let perm = axes(&perm)?;
    (|| {
        let end = g.f_cumsum(-1, Kind::Int64)?;
        let start = end.f_sub(g)?;
        let j = Tensor::f_arange(xs[axis], (Kind::Int64, tracing_device()))?.f_reshape(&pos)?;
        let mask = j
            .f_ge_tensor(&start.f_permute(&perm)?.f_reshape(&s)?)?
            .f_logical_and(&j.f_lt_tensor(&end.f_permute(&perm)?.f_reshape(&s)?)?)?;
        x.f_unsqueeze(0)?
            .f_where_self(&mask, &Tensor::f_zeros([], (x.kind(), tracing_device()))?)
    })()
    .map_err(io::Error::other)
}
impl TorchLowering for RaggedDotGeneralOperation {
    fn lower_torch(&self, e: &Eqn, args: &[Tensor]) -> io::Result<Vec<Tensor>> {
        let [x, y, g] = args else {
            return Err(invalid("missing ragged dot operands"));
        };
        let p = self.dimensions();
        let d = &p.dot;
        let a = p.lhs_ragged;
        let k = result_layout(e)?.1;
        if d.lhs_batch.contains(&a) {
            return Ok(vec![dot(x, y, d, k)?]);
        }
        let shift = |s: &[usize]| s.iter().map(|i| i + 1).collect::<Vec<_>>();
        let mut next = d.clone();
        next.lhs_contracting = shift(&d.lhs_contracting);
        next.lhs_batch = shift(&d.lhs_batch);
        let result = if let Some(i) = d.lhs_contracting.iter().position(|&v| v == a) {
            if i >= d.rhs_contracting.len() {
                return Err(invalid("invalid ragged contracting axes"));
            }
            let x = ragged(
                x,
                g,
                a,
                &[&d.lhs_batch[..], &d.lhs_contracting[..i]].concat(),
            )?;
            let y = ragged(
                y,
                g,
                d.rhs_contracting[i],
                &[&d.rhs_batch[..], &d.rhs_contracting[..i]].concat(),
            )?;
            next.rhs_batch = shift(&d.rhs_batch);
            next.rhs_batch.insert(0, 0);
            next.lhs_batch.insert(0, 0);
            next.rhs_contracting = shift(&d.rhs_contracting);
            dot(&x, &y, &next, k)?
        } else {
            let mut prefix = d.lhs_batch.clone();
            prefix.extend(
                (0..a).filter(|i| !d.lhs_batch.contains(i) && !d.lhs_contracting.contains(i)),
            );
            let x = ragged(x, g, a, &prefix)?;
            next.lhs_contracting.push(0);
            next.rhs_contracting.push(
                p.rhs_group
                    .ok_or_else(|| invalid("missing rhs group axis"))?,
            );
            dot(&x, y, &next, k)?
        };
        Ok(vec![result])
    }
}
crate::register_torch_lowering!(
    DotGeneralOperation,
    ScaledDotOperation,
    RaggedDotGeneralOperation
);

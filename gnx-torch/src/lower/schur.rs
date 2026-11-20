use super::*;
use gnx_math::ops::linalg::SchurOperation;
use tch::TchError;

// An eigenvector spans an invariant line. For a real matrix, the real and
// imaginary parts for a conjugate pair span an invariant plane instead.
// Only a basis of that first subspace is needed: QR completes it to a
// unitary basis even when the matrix has no full basis of eigenvectors.
fn real_step(
    a: &Tensor,
    w: &Tensor,
    v: &Tensor,
    skip: &Tensor,
) -> Result<(Tensor, Tensor), TchError> {
    let n = a.size()[a.dim() - 1];
    let pair = w
        .f_select(-1, 0)?
        .f_imag()?
        .f_ne(0.)?
        .f_logical_and(&skip.f_logical_not()?)?;
    let basis = Tensor::f_cat(&[v.f_real()?, v.f_imag()?], -1)?;
    let (u, _) = Tensor::f_linalg_qr(&basis, "complete")?;
    // Standardize a real 2x2 conjugate block: rotate its diagonal entries
    // to their common mean. atan2 also handles already canonical blocks.
    let plane = u.f_narrow(-1, 0, 2)?;
    let block = plane.f_transpose(-2, -1)?.f_matmul(&a.f_matmul(&plane)?)?;
    let at = |i, j| block.f_select(-2, i)?.f_select(-1, j);
    let diff = at(1, 1)?
        .f_mul_scalar(0.5)?
        .f_sub(&at(0, 0)?.f_mul_scalar(0.5)?)?;
    let sum = at(0, 1)?
        .f_mul_scalar(0.5)?
        .f_add(&at(1, 0)?.f_mul_scalar(0.5)?)?;
    let angle = diff
        .f_atan2(&sum)?
        .f_mul_scalar(0.5)?
        .f_masked_fill(&pair.f_logical_not()?, 0.)?;
    let (c, s) = (angle.f_cos()?, angle.f_sin()?);
    let rot = Tensor::f_stack(
        &[
            Tensor::f_stack(&[&c, &s.f_neg()?], -1)?,
            Tensor::f_stack(&[s, c], -1)?,
        ],
        -2,
    )?;
    let u = Tensor::f_cat(&[plane.f_matmul(&rot)?, u.f_narrow(-1, 2, n - 2)?], -1)?;
    let eye = Tensor::f_eye(n, (a.kind(), tracing_device()))?;
    Ok((
        eye.f_where_self(&skip.f_unsqueeze(-1)?.f_unsqueeze(-1)?, &u)?,
        pair,
    ))
}

fn decompose(a: &Tensor, n: i64, vectors: bool) -> Result<Vec<Tensor>, TchError> {
    let shape = a.size();
    let mut t = a.shallow_clone();
    let mut q = if vectors {
        Some(Tensor::f_eye(n, (a.kind(), tracing_device()))?.f_expand(&shape, false)?)
    } else {
        None
    };
    let real = !a.is_complex();
    let mut pairs = Vec::new();
    let mut skip = Tensor::f_zeros(&shape[..shape.len() - 2], (Kind::Bool, tracing_device()))?;
    // Deflate the trailing matrix without ever reading a tensor on the host.
    // skip is per batch: a plane deflated at i already handles column i+1.
    for i in 0..n - 1 {
        let m = n - i;
        let slab = t.f_narrow(-1, i, m)?;
        let active = slab.f_narrow(-2, i, m)?;
        let (w, v) = active.f_linalg_eig()?;
        let v = v.f_narrow(-1, 0, 1)?;
        let u = if real {
            let (u, pair) = real_step(&active, &w, &v, &skip)?;
            skip = pair.shallow_clone();
            pairs.push(pair);
            u
        } else {
            Tensor::f_linalg_qr(&v, "complete")?.0
        };
        let right = slab.f_matmul(&u)?;
        let transformed = Tensor::f_cat(
            &[
                right.f_narrow(-2, 0, i)?,
                u.f_adjoint()?.f_matmul(&right.f_narrow(-2, i, m)?)?,
            ],
            -2,
        )?;
        t = Tensor::f_cat(&[t.f_narrow(-1, 0, i)?, transformed], -1)?;
        if let Some(ref mut q) = q {
            *q = Tensor::f_cat(
                &[q.f_narrow(-1, 0, i)?, q.f_narrow(-1, i, m)?.f_matmul(&u)?],
                -1,
            )?;
        }
    }
    if !pairs.is_empty() {
        let mask = Tensor::f_stack(&pairs, -1)?;
        let off = t
            .f_diagonal(-1, -2, -1)?
            .f_masked_fill(&mask.f_logical_not()?, 0.)?;
        let d = t.f_diagonal(0, -2, -1)?;
        let first = d.f_narrow(-1, 0, n - 1)?;
        let second = d.f_narrow(-1, 1, n - 1)?;
        let mean = first.f_mul_scalar(0.5)?.f_add(&second.f_mul_scalar(0.5)?)?;
        let diag = Tensor::f_cat(
            &[mean.f_where_self(&mask, &first)?, d.f_narrow(-1, n - 1, 1)?],
            -1,
        )?;
        let diag = Tensor::f_cat(
            &[
                diag.f_narrow(-1, 0, 1)?,
                mean.f_where_self(&mask, &diag.f_narrow(-1, 1, n - 1)?)?,
            ],
            -1,
        )?;
        t = t
            .f_triu(1)?
            .f_add(&diag.f_diag_embed(0, -2, -1)?)?
            .f_add(&off.f_diag_embed(-1, -2, -1)?)?;
    } else {
        t = t.f_triu(0)?;
    }
    let mut out = vec![t];
    out.extend(q);
    Ok(out)
}
impl TorchLowering for SchurOperation {
    fn lower_torch(&self, _: &Eqn, args: &[Tensor]) -> io::Result<Vec<Tensor>> {
        let [a] = args else {
            return Err(invalid("schur requires one operand"));
        };
        let shape = a.size();
        let [.., m, n] = shape.as_slice() else {
            return Err(invalid("schur requires matrices"));
        };
        if m != n {
            return Err(invalid("schur requires square matrices"));
        }
        if !matches!(
            a.kind(),
            Kind::Float | Kind::Double | Kind::ComplexFloat | Kind::ComplexDouble
        ) {
            return Err(unsupported("schur requires f32, f64, c64 or c128"));
        }
        decompose(a, *n, self.vectors()).map_err(io::Error::other)
    }
}
crate::register_torch_lowering!(SchurOperation);

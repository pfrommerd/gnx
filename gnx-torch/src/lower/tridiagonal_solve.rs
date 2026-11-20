use super::*;
use gnx_math::ops::linalg::TridiagonalSolveOperation;
use tch::TchError;

// As in LAPACK gtsv, pivot only between adjacent rows. A row interchange
// introduces at most one extra superdiagonal; no dense n-by-n matrix is built.
fn pivot_norm(x: &Tensor) -> Result<Tensor, TchError> {
    if x.is_complex() {
        x.f_real()?.f_abs()?.f_add(&x.f_imag()?.f_abs()?)
    } else {
        x.f_abs()
    }
}
fn solve(dl: &Tensor, d: &Tensor, du: &Tensor, b: &Tensor) -> Result<(Tensor, Tensor), TchError> {
    let n = d.size()[d.dim() - 1];
    // Coefficients and pivot masks end in a singleton axis so they broadcast
    // over RHS columns, not over batches. Ignore dl[...,0] and du[...,n-1].
    let mut diag = d.f_narrow(-1, 0, 1)?;
    let zero = diag.f_zeros_like()?;
    let mut upper = if n > 1 {
        du.f_narrow(-1, 0, 1)?
    } else {
        zero.shallow_clone()
    };
    let mut rhs = b.f_select(-2, 0)?;
    let mut rows = Vec::new();
    let mut singular = Tensor::f_zeros(diag.size(), (Kind::Bool, tracing_device()))?;
    for i in 1..n {
        let lower = dl.f_narrow(-1, i, 1)?;
        let next_d = d.f_narrow(-1, i, 1)?;
        let next_u = if i + 1 < n {
            du.f_narrow(-1, i, 1)?
        } else {
            zero.shallow_clone()
        };
        let next_rhs = b.f_select(-2, i)?;
        let swap = pivot_norm(&lower)?.f_gt_tensor(&pivot_norm(&diag)?)?;
        let pivot = lower.f_where_self(&swap, &diag)?;
        let bad = pivot.f_eq(0)?;
        singular = singular.f_logical_or(&bad)?;
        let pivot = pivot.f_masked_fill(&bad, 1.)?;
        let factor = diag.f_where_self(&swap, &lower)?.f_div(&pivot)?;
        let u = next_d.f_where_self(&swap, &upper)?;
        let u2 = next_u.f_where_self(&swap, &zero)?;
        let top = next_rhs.f_where_self(&swap, &rhs)?;
        diag = upper
            .f_where_self(&swap, &next_d)?
            .f_sub(&factor.f_mul(&u)?)?;
        upper = factor
            .f_neg()?
            .f_mul(&next_u)?
            .f_where_self(&swap, &next_u)?;
        rhs = rhs
            .f_where_self(&swap, &next_rhs)?
            .f_sub(&factor.f_mul(&top)?)?;
        rows.push((pivot, u, u2, top));
    }
    let bad = diag.f_eq(0)?;
    singular = singular.f_logical_or(&bad)?;
    let mut solution = vec![rhs.f_div(&diag.f_masked_fill(&bad, 1.)?)?];
    for (pivot, u, u2, top) in rows.into_iter().rev() {
        let len = solution.len();
        let mut r = top.f_sub(&u.f_mul(&solution[len - 1])?)?;
        if len > 1 {
            r = r.f_sub(&u2.f_mul(&solution[len - 2])?)?;
        }
        solution.push(r.f_div(&pivot)?);
    }
    solution.reverse();
    Ok((Tensor::f_stack(&solution, -2)?, singular))
}

rule!(TridiagonalSolveOperation, |op, _eqn, [dl, d, du, b]| {
    let shape = d.size();
    let rhs_shape = b.size();
    if shape.is_empty()
        || dl.size() != shape
        || du.size() != shape
        || rhs_shape.split_last().map(|(_, s)| s) != Some(shape.as_slice())
        || [dl, du, b].iter().any(|v| v.kind() != d.kind())
    {
        return Err(invalid(
            "tridiagonal_solve expects three matching [...,n] diagonals and a [...,n,k] RHS of the same dtype",
        ));
    }
    let eps = match d.kind() {
        Kind::Float | Kind::ComplexFloat => f32::EPSILON as f64,
        Kind::Double | Kind::ComplexDouble => f64::EPSILON,
        _ => {
            return Err(unsupported(
                "tridiagonal_solve requires f32, f64, c64 or c128",
            ));
        }
    };
    if b.numel() == 0 {
        return Ok(vec![b.shallow_clone()]);
    }
    (|| {
        let (x, failed) = solve(dl, d, du, b)?;
        if !op.perturb_singular() {
            return x.f_masked_fill(&failed.f_unsqueeze(-1)?, f64::NAN);
        }
        // GNX retries exactly singular systems with eps*max(1,max(abs(d)))
        // added to every diagonal. Keep this decision live per batch on replay.
        let delta = d
            .f_abs()?
            .f_amax([-1], true)?
            .f_clamp_min(1.)?
            .f_mul_scalar(eps)?;
        let (retry, still_singular) = solve(dl, &d.f_add(&delta)?, du, b)?;
        retry
            .f_masked_fill(&still_singular.f_unsqueeze(-1)?, f64::NAN)?
            .f_where_self(&failed.f_unsqueeze(-1)?, &x)
    })()
});

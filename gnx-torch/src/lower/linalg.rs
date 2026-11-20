use super::*;
use gnx_math::ops::linalg::*;
use tch::TchError;

fn sym(x: &Tensor, yes: bool) -> Result<Tensor, TchError> {
    if yes {
        x.f_add(&x.f_adjoint()?)?.f_mul_scalar(0.5)
    } else {
        Ok(x.shallow_clone())
    }
}
fn chol(x: &Tensor, upper: bool) -> Result<Tensor, TchError> {
    let (l, info) = x.f_linalg_cholesky_ex(upper, false)?;
    l.f_masked_fill(&info.f_ne(0)?.f_unsqueeze(-1)?.f_unsqueeze(-1)?, f64::NAN)
}
rule!(CholeskyOperation, |op, _eqn, [x]| {
    sym(x, op.symmetrize()).and_then(|a| chol(&a, false))
});
rule!(CholeskyUpdateOperation, |_op, _eqn, [r, w]| {
    (|| {
        let w = w.f_unsqueeze(-1)?;
        let a = r
            .f_transpose(-1, -2)?
            .f_matmul(r)?
            .f_add(&w.f_matmul(&w.f_transpose(-1, -2)?)?)?;
        chol(&a, true)
    })()
});
rule!(HouseholderProductOperation, |_op, _eqn, [x, tau]| {
    x.f_linalg_householder_product(tau)
});
rule!(OrmqrOperation, |op, _eqn, [a, tau, c]| {
    let p = op.options();
    a.f_ormqr(tau, c, p.left, p.transpose)
});
rule!(SymmetricProductOperation, |op, _eqn, [x, c]| {
    let [alpha, beta] = op.coefficients();
    (|| {
        let t = x.f_matmul(&x.f_transpose(-1, -2)?)?.f_mul_scalar(alpha)?;
        if beta == 0. {
            Ok(t)
        } else {
            t.f_add(&c.f_mul_scalar(beta)?)
        }
    })()
});
rule!(TriangularSolveOperation, |op, _eqn, [a, b]| {
    let p = op.options();
    (|| {
        let mut a = a.shallow_clone();
        if p.transpose {
            a = a.f_transpose(-1, -2)?
        }
        if p.conjugate {
            a = a.f_conj_physical()?
        }
        let rhs = if a.dim() == b.dim() + 1 {
            b.f_unsqueeze(if p.left_side { -1 } else { -2 })?
        } else {
            b.shallow_clone()
        };
        a.f_linalg_solve_triangular(&rhs, p.lower == p.transpose, p.left_side, p.unit_diagonal)?
            .f_reshape(b.size())
    })()
});

macro_rules! multi {
    ($ty:ty,|$op:ident,$args:ident|$b:block) => {
        impl TorchLowering for $ty {
            fn lower_torch(&self, _: &Eqn, $args: &[Tensor]) -> io::Result<Vec<Tensor>> {
                let $op = self;
                let result: Result<Vec<Tensor>, TchError> = (|| $b)();
                result.map_err(io::Error::other)
            }
        }
        crate::register_torch_lowering!($ty);
    };
}
multi!(EighOperation, |op, x| {
    let a = sym(&x[0], op.symmetrize())?;
    let (w, v) = a.f_linalg_eigh(if op.lower() { "L" } else { "U" })?;
    Ok(vec![v, w])
});
// LAPACK is free to return the spectrum of A^H in a different order from
// that of A. Match conjugate eigenvalues independently for every batch item.
// Consume each column once so repeated eigenvalues do not reuse one vector.
fn align_left(values: &Tensor, adjoint_values: &Tensor, left: Tensor) -> Result<Tensor, TchError> {
    let n = values.size()[values.dim() - 1];
    if n < 2 {
        return Ok(left);
    }
    let targets = values.f_conj_physical()?;
    let mut used = Tensor::f_zeros(values.size(), (Kind::Bool, tracing_device()))?;
    let mut columns = Vec::new();
    for j in 0..n {
        let distance = adjoint_values
            .f_sub(&targets.f_narrow(-1, j, 1)?)?
            .f_abs()?;
        let index = distance
            .f_masked_fill(&used, f64::INFINITY)?
            .f_argmin(-1, true)?;
        used = used.f_scatter_value(-1, &index, 1i64)?;
        columns.push(index);
    }
    let order = Tensor::f_cat(&columns, -1)?
        .f_unsqueeze(-2)?
        .f_expand(left.size(), false)?;
    left.f_gather(-1, &order, false)
}

impl TorchLowering for EigOperation {
    fn lower_torch(&self, _: &Eqn, args: &[Tensor]) -> io::Result<Vec<Tensor>> {
        let [a] = args else {
            return Err(invalid("eig requires one operand"));
        };
        let shape = a.size();
        if shape.len() < 2 || shape[shape.len() - 1] != shape[shape.len() - 2] {
            return Err(invalid("eig requires square matrices"));
        }
        let kind = match a.kind() {
            Kind::Float | Kind::ComplexFloat => Kind::ComplexFloat,
            Kind::Double | Kind::ComplexDouble => Kind::ComplexDouble,
            _ => return Err(unsupported("eig requires f32, f64, c64 or c128")),
        };
        (|| {
            let a = a.f_to_kind(kind)?;
            match (self.left(), self.right()) {
                (false, false) => Ok(vec![a.f_linalg_eigvals()?]),
                (false, true) => {
                    let (w, r) = a.f_linalg_eig()?;
                    Ok(vec![w, r])
                }
                (true, false) => {
                    let (w, l) = a.f_adjoint()?.f_linalg_eig()?;
                    Ok(vec![w.f_conj_physical()?, l])
                }
                (true, true) => {
                    let (w, r) = a.f_linalg_eig()?;
                    // A second solve also works when the right eigenbasis is singular.
                    let (aw, l) = a.f_adjoint()?.f_linalg_eig()?;
                    let l = align_left(&w, &aw, l)?;
                    Ok(vec![w, l, r])
                }
            }
        })()
        .map_err(|e: TchError| io::Error::other(e))
    }
}
crate::register_torch_lowering!(EigOperation);

/// Reduce A by unitary similarity. geqrf on a single column supplies the
/// LAPACK reflector P=I-tau*v*vᴴ, including the complex phase convention.
/// Keep completed columns packed; only the unreduced block sees Pᴴ A P.
fn hessenberg(a: &Tensor) -> Result<(Tensor, Tensor), TchError> {
    let shape = a.size();
    let [batch @ .., rows, cols] = shape.as_slice() else {
        return Err(TchError::Shape("hessenberg expects matrices".into()));
    };
    if rows != cols {
        return Err(TchError::Shape("hessenberg expects square matrices".into()));
    }
    let n = *rows;
    let mut packed = a.shallow_clone();
    let mut factors = Vec::new();
    for i in 0..n - 1 {
        let head = i + 1;
        let tail = n - head;
        let column = packed.f_narrow(-1, i, 1)?;
        let (qr, tau) = column.f_narrow(-2, head, tail)?.f_geqrf()?;
        let v = Tensor::f_cat(
            &[
                qr.f_narrow(-2, 0, 1)?.f_ones_like()?,
                qr.f_narrow(-2, 1, tail - 1)?,
            ],
            -2,
        )?;
        let vh = v.f_adjoint()?;
        let t = tau.f_unsqueeze(-1)?;
        let slab = packed.f_narrow(-1, head, tail)?;
        let right = slab.f_sub(&slab.f_matmul(&v)?.f_mul(&t)?.f_matmul(&vh)?)?;
        let block = right.f_narrow(-2, head, tail)?;
        let block =
            block.f_sub(&v.f_matmul(&vh.f_matmul(&block)?.f_mul(&t.f_conj_physical()?)?)?)?;
        let slab = Tensor::f_cat(&[right.f_narrow(-2, 0, head)?, block], -2)?;
        let col = Tensor::f_cat(&[column.f_narrow(-2, 0, head)?, qr], -2)?;
        packed = Tensor::f_cat(&[packed.f_narrow(-1, 0, i)?, col, slab], -1)?;
        factors.push(tau);
    }
    let tau = if factors.is_empty() {
        let mut dims = batch.to_vec();
        dims.push(0);
        Tensor::f_zeros(dims, (a.kind(), tracing_device()))?
    } else {
        Tensor::f_cat(&factors, -1)?
    };
    Ok((packed, tau))
}
impl TorchLowering for HessenbergOperation {
    fn lower_torch(&self, _: &Eqn, inputs: &[Tensor]) -> io::Result<Vec<Tensor>> {
        let [a] = inputs else {
            return Err(invalid("hessenberg expects one operand"));
        };
        if !matches!(
            a.kind(),
            Kind::Float | Kind::Double | Kind::ComplexFloat | Kind::ComplexDouble
        ) {
            return Err(unsupported("hessenberg requires f32, f64, c64 or c128"));
        }
        let (packed, tau) = hessenberg(a).map_err(io::Error::other)?;
        Ok(vec![packed, tau])
    }
}
crate::register_torch_lowering!(HessenbergOperation);

impl TorchLowering for TridiagonalOperation {
    fn lower_torch(&self, _: &Eqn, args: &[Tensor]) -> io::Result<Vec<Tensor>> {
        let [a] = args else {
            return Err(invalid("tridiagonal expects one operand"));
        };
        let shape = a.size();
        if !matches!(shape.as_slice(), [..,m,n] if m==n) {
            return Err(invalid("tridiagonal requires square matrices"));
        }
        if !matches!(
            a.kind(),
            Kind::Float | Kind::Double | Kind::ComplexFloat | Kind::ComplexDouble
        ) {
            return Err(unsupported("tridiagonal requires f32, f64, c64 or c128"));
        }
        tridiagonal(a, self.lower()).map_err(io::Error::other)
    }
}

// Hessenberg reduction of a Hermitian matrix is tridiagonal. Reversing
// both axes lets the same reflectors implement the upper-storage convention:
// the implicit unit is then last, and the product runs in reverse order.
fn tridiagonal(a: &Tensor, lower: bool) -> Result<Vec<Tensor>, TchError> {
    let orient = |x: &Tensor| {
        if lower {
            Ok(x.shallow_clone())
        } else {
            x.f_flip([-2, -1])
        }
    };
    let x = orient(a)?;
    let triangle = x.f_tril(-1)?;
    let diagonal = x.f_diagonal(0, -2, -1)?.f_real()?.f_diag_embed(0, -2, -1)?;
    let hermitian = triangle.f_add(&triangle.f_adjoint()?)?.f_add(&diagonal)?;
    let (packed, tau) = hessenberg(&hermitian)?;
    let d = packed.f_diagonal(0, -2, -1)?.f_real()?;
    let e = packed.f_diagonal(-1, -2, -1)?.f_real()?;
    // Keep only the packed reflector triangle. The LAPACK format leaves
    // the unselected triangle untouched and stores the real diagonal in place.
    let packed = packed
        .f_tril(-1)?
        .f_add(&d.f_diag_embed(0, -2, -1)?)?
        .f_add(&x.f_triu(1)?)?;
    let orient_vector = |v: Tensor| if lower { Ok(v) } else { v.f_flip([-1]) };
    Ok(vec![
        orient(&packed)?,
        orient_vector(d)?,
        orient_vector(e)?,
        orient_vector(tau)?,
    ])
}
crate::register_torch_lowering!(TridiagonalOperation);

multi!(SvdOperation, |op, x| {
    let (u, s, v) = x[0].f_svd(!op.full(), op.vectors())?;
    Ok(if op.vectors() {
        vec![u, s, v.f_adjoint()?]
    } else {
        vec![s]
    })
});
fn permutation(p: &Tensor, n: i64) -> Result<Tensor, TchError> {
    let p = p.f_to_kind(Kind::Int64)?;
    let mut dims = p.size();
    let count = *dims
        .last()
        .ok_or_else(|| TchError::Shape("pivot array needs a final axis".into()))?;
    *dims.last_mut().unwrap() = n;
    let mut seq = Tensor::f_arange(n, (Kind::Int64, tracing_device()))?.f_expand(dims, false)?;
    for i in 0..count {
        let j = p.f_narrow(-1, i, 1)?;
        let vi = seq.f_narrow(-1, i, 1)?;
        let vj = seq.f_gather(-1, &j, false)?;
        seq = seq
            .f_scatter(-1, &j, &vi)?
            .f_scatter(-1, &j.f_full_like(i)?, &vj)?;
    }
    seq.f_to_kind(Kind::Int)
}
rule!(LuPivotsOperation, |op, _eqn, [p]| {
    permutation(p, i64::try_from(op.size()).map_err(invalid)?)
});
multi!(LuOperation, |_op, x| {
    let (a, p, _) = Tensor::f_linalg_lu_factor_ex(&x[0], true, false)?;
    let p = p.f_sub_scalar(1)?;
    let s = a.size();
    let perm = permutation(&p, s[s.len() - 2])?;
    Ok(vec![a, p, perm])
});

// LAPACK-style column pivoting: choose the largest trailing column norm at
// each step. All permutations and pivot choices are tensor data in the trace.
fn pivot_columns(a: &Tensor) -> Result<(Tensor, Tensor), TchError> {
    let shape = a.size();
    if shape.len() < 2 {
        return Err(TchError::Shape("matrix expected".into()));
    }
    let (m, n) = (shape[shape.len() - 2], shape[shape.len() - 1]);
    let mut piv_shape = shape.clone();
    piv_shape.remove(shape.len() - 2);
    let mut p = Tensor::f_arange(n, (Kind::Int64, tracing_device()))?.f_expand(piv_shape, false)?;
    let mut r = a.shallow_clone();
    for i in 0..m.min(n) {
        let trailing = r.f_narrow(-1, i, n - i)?;
        let norms = trailing
            .f_abs()?
            .f_square()?
            .f_sum_dim_intlist(&[-2i64][..], false, None)?;
        let j = norms.f_argmax(-1, true)?.f_add_scalar(i)?;
        let pi = p.f_narrow(-1, i, 1)?;
        let pj = p.f_gather(-1, &j, false)?;
        p = p
            .f_scatter(-1, &j, &pi)?
            .f_scatter(-1, &j.f_full_like(i)?, &pj)?;
        let mut col = shape.clone();
        *col.last_mut().unwrap() = 1;
        let j = j.f_unsqueeze(-2)?.f_expand(&col, false)?;
        let ri = r.f_narrow(-1, i, 1)?;
        let rj = r.f_gather(-1, &j, false)?;
        r = r
            .f_scatter(-1, &j, &ri)?
            .f_scatter(-1, &j.f_full_like(i)?, &rj)?;
        let (q, _) = Tensor::f_linalg_qr(&rj, "reduced")?;
        r = r.f_sub(&q.f_matmul(&q.f_adjoint()?.f_matmul(&r)?)?)?;
    }
    Ok((
        a.f_gather(-1, &p.f_unsqueeze(-2)?.f_expand(&shape, false)?, false)?,
        p.f_to_kind(Kind::Int)?,
    ))
}
multi!(QrOperation, |op, x| {
    let mode = if op.full() { "complete" } else { "reduced" };
    if op.pivoting() {
        let (a, p) = pivot_columns(&x[0])?;
        let (q, r) = Tensor::f_linalg_qr(&a, mode)?;
        Ok(vec![q, r, p])
    } else {
        let (q, r) = Tensor::f_linalg_qr(&x[0], mode)?;
        Ok(vec![q, r])
    }
});

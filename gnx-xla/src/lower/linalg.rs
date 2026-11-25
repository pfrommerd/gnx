use std::io;

use gnx_expr::Eqn;
use gnx_math::ops::linalg::*;
use xla::{PrimitiveType, XlaOp};

use super::{LoweringContext, XlaLowering, invalid, items, native, unary};

fn transpose(x: &XlaOp, conjugate: bool) -> io::Result<XlaOp> {
    let rank = native(x.array_shape())?.dims().len();
    if rank < 2 {
        return Err(invalid("matrix rank must be at least two"));
    }
    let mut perm: Vec<_> = (0..rank as i64).collect();
    perm.swap(rank - 1, rank - 2);
    let value = native(x.transpose(&perm))?;
    maybe_conjugate(&value, conjugate)
}

fn maybe_conjugate(x: &XlaOp, conjugate: bool) -> io::Result<XlaOp> {
    if conjugate && matches!(native(x.ty())?, PrimitiveType::C64 | PrimitiveType::C128) {
        native(x.conj())
    } else {
        Ok(x.clone())
    }
}

fn scalar(x: &XlaOp, value: f64) -> io::Result<XlaOp> {
    native(native(x.builder().c0(value))?.convert(native(x.ty())?))
}

fn symmetrize(x: &XlaOp) -> io::Result<XlaOp> {
    let sum = native(x.add_(&transpose(x, true)?))?;
    native(sum.mul_(&scalar(x, 0.5)?))
}

fn matmul(x: &XlaOp, y: &XlaOp) -> io::Result<XlaOp> {
    let rank = native(x.array_shape())?.dims().len();
    if rank < 2 || native(y.array_shape())?.dims().len() != rank {
        return Err(invalid(
            "matrix operands must have matching ranks of at least two",
        ));
    }
    let batch = (0..(rank as i64 - 2)).collect::<Vec<_>>();
    native(x.dot_general(y, &[rank as i64 - 1], &[rank as i64 - 2], &batch, &batch))
}

impl XlaLowering for CholeskyOperation {
    fn lower_xla(
        &self,
        _: &LoweringContext<'_>,
        _: Eqn<'_>,
        inputs: &[XlaOp],
    ) -> io::Result<Vec<XlaOp>> {
        let x = unary("cholesky", inputs)?;
        let x = if self.symmetrize() {
            symmetrize(x)?
        } else {
            x.clone()
        };
        let result = native(x.cholesky(true))?;
        Ok(vec![native(result.lower_triangle())?])
    }
}

impl XlaLowering for CholeskyUpdateOperation {
    fn lower_xla(
        &self,
        _: &LoweringContext<'_>,
        _: Eqn<'_>,
        inputs: &[XlaOp],
    ) -> io::Result<Vec<XlaOp>> {
        let [r, w] = inputs else {
            return Err(invalid("cholesky_update signature"));
        };
        let wt = native(w.array_shape())?;
        let mut shape = wt.dims().to_vec();
        shape.push(1);
        let w = native(w.reshape(&shape))?;
        let cross = matmul(&w, &transpose(&w, false)?)?;
        let base = matmul(&transpose(r, false)?, r)?;
        let result = native(native(base.add_(&cross))?.cholesky(false))?;
        Ok(vec![native(result.upper_triangle())?])
    }
}

impl XlaLowering for TriangularSolveOperation {
    fn lower_xla(
        &self,
        _: &LoweringContext<'_>,
        _: Eqn<'_>,
        inputs: &[XlaOp],
    ) -> io::Result<Vec<XlaOp>> {
        let [a, b] = inputs else {
            return Err(invalid("triangular solve signature"));
        };
        let p = self.options();
        let a = maybe_conjugate(a, p.conjugate && !p.transpose)?;
        let mode = match (p.transpose, p.conjugate) {
            (false, _) => 1,
            (true, false) => 2,
            (true, true) => 3,
        };
        let a_rank = native(a.array_shape())?.dims().len();
        let b_shape = native(b.array_shape())?;
        let vec = b_shape.dims().len() + 1 == a_rank;
        let mut normalized = b_shape.dims().to_vec();
        let rhs = if vec {
            let point = if p.left_side {
                normalized.len()
            } else {
                normalized.len().saturating_sub(1)
            };
            normalized.insert(point, 1);
            native(b.reshape(&normalized))?
        } else {
            b.clone()
        };
        let result = native(a.triangular_solve(&rhs, p.left_side, p.lower, p.unit_diagonal, mode))?;
        if vec {
            Ok(vec![native(result.reshape(b_shape.dims()))?])
        } else {
            Ok(vec![result])
        }
    }
}

impl XlaLowering for HouseholderProductOperation {
    fn lower_xla(
        &self,
        _: &LoweringContext<'_>,
        _: Eqn<'_>,
        inputs: &[XlaOp],
    ) -> io::Result<Vec<XlaOp>> {
        let [a, tau] = inputs else {
            return Err(invalid("householder signature"));
        };
        Ok(vec![native(a.householder_product(tau))?])
    }
}

impl XlaLowering for OrmqrOperation {
    fn lower_xla(
        &self,
        _: &LoweringContext<'_>,
        _: Eqn<'_>,
        inputs: &[XlaOp],
    ) -> io::Result<Vec<XlaOp>> {
        let [a, tau, c] = inputs else {
            return Err(invalid("ormqr signature"));
        };
        let p = self.options();
        let shape = native(a.array_shape())?;
        let rank = shape.dims().len();
        if rank < 2 {
            return Err(invalid("ormqr matrix rank"));
        }
        let rows = shape.dims()[rank - 2];
        let cols = shape.dims()[rank - 1];
        let square = if cols < rows {
            let padding = (0..rank)
                .map(|i| {
                    if i == rank - 1 {
                        (0, rows - cols, 0)
                    } else {
                        (0, 0, 0)
                    }
                })
                .collect::<Vec<_>>();
            native(a.pad(&native(a.builder().zero(shape.ty()))?, &padding))?
        } else {
            native(a.slice_in_dim1(0, rows, rank as i64 - 1))?
        };
        let q = native(square.householder_product(tau))?;
        let q = if p.transpose { transpose(&q, true)? } else { q };
        Ok(vec![if p.left {
            matmul(&q, c)?
        } else {
            matmul(c, &q)?
        }])
    }
}

impl XlaLowering for SymmetricProductOperation {
    fn lower_xla(
        &self,
        _: &LoweringContext<'_>,
        _: Eqn<'_>,
        inputs: &[XlaOp],
    ) -> io::Result<Vec<XlaOp>> {
        let [a, c] = inputs else {
            return Err(invalid("symmetric product signature"));
        };
        let [alpha, beta] = self.coefficients();
        let product = matmul(a, &transpose(a, false)?)?;
        let product = native(product.mul_(&scalar(a, alpha)?))?;
        if beta == 0. {
            return Ok(vec![product]);
        }
        Ok(vec![native(
            product.add_(&native(c.mul_(&scalar(c, beta)?))?),
        )?])
    }
}

impl XlaLowering for EighOperation {
    fn lower_xla(
        &self,
        _: &LoweringContext<'_>,
        _: Eqn<'_>,
        inputs: &[XlaOp],
    ) -> io::Result<Vec<XlaOp>> {
        let x = unary("eigh", inputs)?;
        let x = if self.symmetrize() {
            symmetrize(x)?
        } else {
            x.clone()
        };
        items(&native(x.eigh(self.lower(), self.sort()))?, 2)
    }
}

impl XlaLowering for QrOperation {
    fn lower_xla(
        &self,
        _: &LoweringContext<'_>,
        eqn: Eqn<'_>,
        inputs: &[XlaOp],
    ) -> io::Result<Vec<XlaOp>> {
        if self.pivoting() {
            return lapack(Lapack::PivotedQr, i64::from(self.full()), eqn, inputs);
        }
        items(&native(unary("qr", inputs)?.qr(self.full()))?, 2)
    }
}

crate::register_xla_lowering!(
    CholeskyOperation,
    CholeskyUpdateOperation,
    TriangularSolveOperation,
    HouseholderProductOperation,
    OrmqrOperation,
    SymmetricProductOperation,
    EighOperation,
    QrOperation,
);

use xla::lapack::Kernel as Lapack;

fn lapack(op: Lapack, flags: i64, eqn: Eqn<'_>, inputs: &[XlaOp]) -> io::Result<Vec<XlaOp>> {
    let outputs = eqn
        .outputs()
        .map(|output| super::shape(&super::metadata(output.info_ref())?))
        .collect::<io::Result<Vec<_>>>()?;
    let result = native(op.call(flags, inputs, &outputs))?;
    if outputs.len() == 1 {
        Ok(vec![result])
    } else {
        items(&result, outputs.len())
    }
}

macro_rules! lapack_lowering {
    ($($ty:ty, $op:ident, |$me:ident| $flags:expr;)*) => {$(
        impl XlaLowering for $ty {
            fn lower_xla(&self, _: &LoweringContext<'_>, eqn: Eqn<'_>, inputs: &[XlaOp]) -> io::Result<Vec<XlaOp>> {
                let $me = self;
                lapack(Lapack::$op, $flags, eqn, inputs)
            }
        }
        crate::register_xla_lowering!($ty);
    )*};
}
lapack_lowering! {
    LuOperation, Lu, |_op| 0;
    HessenbergOperation, Hessenberg, |_op| 0;
    LuPivotsOperation, PivotsToPermutation, |op| super::indices(&[op.size()])?[0];
    TridiagonalOperation, Tridiagonal, |op| i64::from(op.lower());
    SchurOperation, Schur, |op| i64::from(op.vectors());
    SvdOperation, Svd, |op| i64::from(op.full()) + 2 * i64::from(op.vectors());
    TridiagonalSolveOperation, TridiagonalSolve, |op| i64::from(op.perturb_singular());
}

impl XlaLowering for EigOperation {
    fn lower_xla(
        &self,
        _: &LoweringContext<'_>,
        eqn: Eqn<'_>,
        inputs: &[XlaOp],
    ) -> io::Result<Vec<XlaOp>> {
        let x = unary("eig", inputs)?;
        let dtype = match native(x.ty())? {
            PrimitiveType::F32 | PrimitiveType::C64 => PrimitiveType::C64,
            PrimitiveType::F64 | PrimitiveType::C128 => PrimitiveType::C128,
            _ => {
                return Err(invalid(
                    "eig requires float32, float64, complex64, or complex128",
                ));
            }
        };
        let x = native(x.convert(dtype))?;
        lapack(
            Lapack::Eig,
            i64::from(self.left()) + 2 * i64::from(self.right()),
            eqn,
            &[x],
        )
    }
}
crate::register_xla_lowering!(EigOperation);

impl XlaLowering for CustomLinearSolveOperation {
    fn lower_xla(
        &self,
        ctx: &LoweringContext<'_>,
        _: Eqn<'_>,
        inputs: &[XlaOp],
    ) -> io::Result<Vec<XlaOp>> {
        ctx.lower_region(1, inputs)
    }
}
crate::register_xla_lowering!(CustomLinearSolveOperation);

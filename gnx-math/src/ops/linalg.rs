use super::{Dialect, Eqn, Op, Operation, Result, arity, fmt, signature};

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct CholeskyOperation {
    symmetrize: bool,
}
impl CholeskyOperation {
    pub fn symmetrize(&self) -> bool {
        self.symmetrize
    }
}
gnx_array::array_operation!(
    CholeskyOperation,
    "cholesky",
    |_op, eqn| arity(eqn, 1),
    symmetrize
);
pub fn cholesky(symmetrize: bool) -> Op {
    Op::new(CholeskyOperation { symmetrize })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TriangularSolveOptions {
    pub left_side: bool,
    pub lower: bool,
    pub transpose: bool,
    pub conjugate: bool,
    pub unit_diagonal: bool,
}
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct TriangularSolveOperation {
    options: TriangularSolveOptions,
}
impl TriangularSolveOperation {
    pub fn options(&self) -> TriangularSolveOptions {
        self.options
    }
}
gnx_array::array_operation!(
    TriangularSolveOperation,
    "triangular_solve",
    |_op, eqn| arity(eqn, 2),
    options
);
pub fn triangular_solve(options: TriangularSolveOptions) -> Op {
    Op::new(TriangularSolveOperation { options })
}

simple_operations! {
    CholeskyUpdateOperation, cholesky_update_op, cholesky_update, 2;
    HouseholderProductOperation, householder_product_op, householder_product, 2;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OrmqrOptions {
    pub left: bool,
    pub transpose: bool,
}
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct OrmqrOperation {
    options: OrmqrOptions,
}
impl OrmqrOperation {
    pub fn options(&self) -> OrmqrOptions {
        self.options
    }
}
gnx_array::array_operation!(OrmqrOperation, "ormqr", |_op, eqn| arity(eqn, 3), options);
pub fn ormqr(options: OrmqrOptions) -> Op {
    Op::new(OrmqrOperation { options })
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct SymmetricProductOperation {
    coefficients: [u64; 2],
}
impl SymmetricProductOperation {
    pub fn coefficients(&self) -> [f64; 2] {
        self.coefficients.map(f64::from_bits)
    }
}
gnx_array::array_operation!(
    SymmetricProductOperation,
    "symmetric_product",
    |_op, eqn| arity(eqn, 2),
    coefficients
);
pub fn symmetric_product(alpha: f64, beta: f64) -> Op {
    Op::new(SymmetricProductOperation {
        coefficients: [alpha.to_bits(), beta.to_bits()],
    })
}

fn signature_count(eqn: &Eqn, inputs: usize, outputs: usize) -> Result<()> {
    if eqn.inputs().len() != inputs || eqn.outputs().len() != outputs || !eqn.subexprs().is_empty()
    {
        return Err(signature("linear algebra operation signature"));
    }
    Ok(())
}

macro_rules! matrix_operation {
    ($ty:ident, $ctor:ident, [$($field:ident: $v:ty),*], $inputs:literal, |$op:ident| $outputs:expr) => {
        #[derive(Clone, PartialEq, Eq, Hash)]
        pub struct $ty { $($field: $v),* }
        impl $ty { $(pub fn $field(&self) -> $v { self.$field })* }
        gnx_array::array_operation!($ty, stringify!($ctor), |$op, eqn| signature_count(eqn, $inputs, $outputs));
        pub fn $ctor($($field: $v),*) -> Op { Op::new($ty { $($field),* }) }
    };
}
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct EigOperation {
    left: bool,
    right: bool,
    differentiate_vectors: bool,
}
impl EigOperation {
    pub fn left(&self) -> bool {
        self.left
    }
    pub fn right(&self) -> bool {
        self.right
    }
    pub fn differentiate_vectors(&self) -> bool {
        self.differentiate_vectors
    }
}
gnx_array::array_operation!(EigOperation, "eig", |op, eqn| signature_count(
    eqn,
    1,
    1 + usize::from(op.left) + usize::from(op.right)
));
pub fn eig(left: bool, right: bool) -> Op {
    eig_with_derivatives(left, right, false)
}
/// Enable the nonsymmetric eigenvector derivative for simple eigenvalues.
pub fn eig_with_derivatives(left: bool, right: bool, differentiate_vectors: bool) -> Op {
    Op::new(EigOperation {
        left,
        right,
        differentiate_vectors,
    })
}
matrix_operation!(EighOperation, eigh, [lower: bool, symmetrize: bool, sort: bool], 1, |_op| 2);
matrix_operation!(HessenbergOperation, hessenberg, [], 1, |_op| 2);
matrix_operation!(LuOperation, lu, [], 1, |_op| 3);
matrix_operation!(LuPivotsOperation, lu_pivots_to_permutation, [size: usize], 1, |_op| 1);
matrix_operation!(QrOperation, qr, [full: bool, pivoting: bool], 1, |op| 2 + usize::from(op.pivoting));
matrix_operation!(SchurOperation, schur, [vectors: bool], 1, |op| 1 + usize::from(op.vectors));
matrix_operation!(SvdOperation, svd, [full: bool, vectors: bool], 1, |op| if op.vectors { 3 } else { 1 });
matrix_operation!(TridiagonalOperation, tridiagonal, [lower: bool], 1, |_op| 4);
matrix_operation!(TridiagonalSolveOperation, tridiagonal_solve, [perturb_singular: bool], 4, |_op| 1);

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct CustomLinearSolveOperation {
    parameters: usize,
    symmetric: bool,
}
impl CustomLinearSolveOperation {
    pub fn parameters(&self) -> usize {
        self.parameters
    }
    pub fn symmetric(&self) -> bool {
        self.symmetric
    }
}

gnx_array::array_operation!(CustomLinearSolveOperation, "custom_linear_solve", |op, eqn| {
    let n = eqn.outputs().len();
    let regions = eqn.subexprs();
    if n == 0 || eqn.inputs().len() != op.parameters + n || !(regions.len() == 2 || regions.len() == 3) {
        return Err(signature("invalid custom_linear_solve arity"));
    }
    for reg in regions {
        let body = reg.expr();
        if !reg.closure().is_empty() || body.inputs().len() != eqn.inputs().len() || body.outputs().len() != n {
            return Err(signature("custom_linear_solve regions need matching parameters and results"));
        }
        for (result, (_, info)) in body.outputs().iter().zip(eqn.outputs()) {
            if body.value_info(result) != Some(info) {
                return Err(signature("custom_linear_solve result metadata mismatch"));
            }
        }
    }
    Ok(())
}, parameters;
    fn lower(&self, eqn: &gnx_expr::HEqn, ctx: &mut gnx_expr::lower::LoweringContext<'_>) -> Result<()> {
        ctx.lower_closed_regions_apply(eqn)
    }
);

/// The read-only regions are ordered matvec, solve, and optional transpose solve.
/// Each takes explicit parameters followed by the right-hand side or vector.
pub fn custom_linear_solve(parameters: usize, symmetric: bool) -> Op {
    Op::new(CustomLinearSolveOperation {
        parameters,
        symmetric,
    })
}

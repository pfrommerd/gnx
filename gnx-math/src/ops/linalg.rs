use super::{Eqn, Op, Result, arity, count, signature};

#[derive(rkyv::Archive, rkyv::Serialize)]
#[rkyv(archived = CholeskyOperation)]
pub struct Cholesky {
    symmetrize: bool,
}
impl CholeskyOperation {
    pub fn symmetrize(&self) -> bool {
        self.symmetrize
    }
}
gnx_base::array_operation!(
    CholeskyOperation,
    "cholesky",
    |_op, eqn| arity(eqn, 1),
    symmetrize
);
pub fn cholesky(symmetrize: bool) -> Op {
    Op::new(Cholesky { symmetrize })
}

#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Hash, rkyv::Archive, rkyv::Serialize, rkyv::Portable,
)]
#[rkyv(as = Self)]
#[repr(C)]
pub struct TriangularSolveOptions {
    pub left_side: bool,
    pub lower: bool,
    pub transpose: bool,
    pub conjugate: bool,
    pub unit_diagonal: bool,
}
#[derive(rkyv::Archive, rkyv::Serialize)]
#[rkyv(archived = TriangularSolveOperation)]
pub struct TriangularSolve {
    options: TriangularSolveOptions,
}
impl TriangularSolveOperation {
    pub fn options(&self) -> TriangularSolveOptions {
        self.options
    }
}
gnx_base::array_operation!(
    TriangularSolveOperation,
    "triangular_solve",
    |_op, eqn| arity(eqn, 2),
    options
);
pub fn triangular_solve(options: TriangularSolveOptions) -> Op {
    Op::new(TriangularSolve { options })
}

simple_operations! {
    CholeskyUpdateOperation, cholesky_update_op, cholesky_update, "cholesky_update", 2;
    HouseholderProductOperation, householder_product_op, householder_product, "householder_product", 2;
}

#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Hash, rkyv::Archive, rkyv::Serialize, rkyv::Portable,
)]
#[rkyv(as = Self)]
#[repr(C)]
pub struct OrmqrOptions {
    pub left: bool,
    pub transpose: bool,
}
#[derive(rkyv::Archive, rkyv::Serialize)]
#[rkyv(archived = OrmqrOperation)]
pub struct Ormqr {
    options: OrmqrOptions,
}
impl OrmqrOperation {
    pub fn options(&self) -> OrmqrOptions {
        self.options
    }
}
gnx_base::array_operation!(OrmqrOperation, "ormqr", |_op, eqn| arity(eqn, 3), options);
pub fn ormqr(options: OrmqrOptions) -> Op {
    Op::new(Ormqr { options })
}

#[derive(rkyv::Archive, rkyv::Serialize)]
#[rkyv(archived = SymmetricProductOperation)]
pub struct SymmetricProduct {
    coefficients: [u64; 2],
}
impl SymmetricProductOperation {
    pub fn coefficients(&self) -> [f64; 2] {
        self.coefficients
            .map(|bits| f64::from_bits(bits.to_native()))
    }
}
impl gnx_expr::Operation for SymmetricProductOperation {
    // Shows the coefficients as their bits.
    fn display(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let bits = self.coefficients.map(|bits| bits.to_native());
        write!(f, "array::symmetric_product<coefficients: {bits:?}>")
    }
    fn check_expr(&self, eqn: Eqn<'_>) -> Result<()> {
        arity(eqn, 2)
    }
}
gnx_expr::dialect! {
    dyn gnx_expr::Operation, "array" => { "symmetric_product" => SymmetricProductOperation }
}
pub fn symmetric_product(alpha: f64, beta: f64) -> Op {
    Op::new(SymmetricProduct {
        coefficients: [alpha.to_bits(), beta.to_bits()],
    })
}

fn signature_count(eqn: Eqn<'_>, inputs: usize, outputs: usize) -> Result<()> {
    if eqn.inputs().len() != inputs || eqn.outputs().len() != outputs || eqn.subexprs().len() != 0 {
        return Err(signature("linear algebra operation signature"));
    }
    Ok(())
}

/// A factorization with boolean parameters: `$builder` is stored as `$ty`.
macro_rules! matrix_operation {
    ($builder:ident => $ty:ident, $name:literal, $ctor:ident, [$($field:ident),*], $inputs:literal, |$op:ident| $outputs:expr) => {
        #[derive(rkyv::Archive, rkyv::Serialize)]
        #[rkyv(archived = $ty)]
        pub struct $builder { $($field: bool),* }
        impl $ty { $(pub fn $field(&self) -> bool { self.$field })* }
        gnx_base::array_operation!($ty, $name, |$op, eqn| signature_count(eqn, $inputs, $outputs));
        pub fn $ctor($($field: bool),*) -> Op { Op::new($builder { $($field),* }) }
    };
}
#[derive(rkyv::Archive, rkyv::Serialize)]
#[rkyv(archived = EigOperation)]
pub struct Eig {
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
gnx_base::array_operation!(EigOperation, "eig", |op, eqn| signature_count(
    eqn,
    1,
    1 + usize::from(op.left) + usize::from(op.right)
));
pub fn eig(left: bool, right: bool) -> Op {
    eig_with_derivatives(left, right, false)
}
/// Enable the nonsymmetric eigenvector derivative for simple eigenvalues.
pub fn eig_with_derivatives(left: bool, right: bool, differentiate_vectors: bool) -> Op {
    Op::new(Eig {
        left,
        right,
        differentiate_vectors,
    })
}
matrix_operation!(Eigh => EighOperation, "eigh", eigh, [lower, symmetrize, sort], 1, |_op| 2);
matrix_operation!(Qr => QrOperation, "qr", qr, [full, pivoting], 1, |op| 2 + usize::from(op.pivoting));
matrix_operation!(Schur => SchurOperation, "schur", schur, [vectors], 1, |op| 1 + usize::from(op.vectors));
matrix_operation!(Svd => SvdOperation, "svd", svd, [full, vectors], 1, |op| if op.vectors { 3 } else { 1 });
matrix_operation!(Tridiagonal => TridiagonalOperation, "tridiagonal", tridiagonal, [lower], 1, |_op| 4);
matrix_operation!(TridiagonalSolve => TridiagonalSolveOperation, "tridiagonal_solve", tridiagonal_solve, [perturb_singular], 4, |_op| 1);

gnx_base::unit_operations! {
    pub struct HessenbergOperation;
    pub struct LuOperation;
}
gnx_base::array_operation!(HessenbergOperation, "hessenberg", |eqn| signature_count(
    eqn, 1, 2
));
gnx_base::array_operation!(LuOperation, "lu", |eqn| signature_count(eqn, 1, 3));
pub fn hessenberg() -> Op {
    Op::new(HessenbergOperation)
}
pub fn lu() -> Op {
    Op::new(LuOperation)
}

#[derive(rkyv::Archive, rkyv::Serialize)]
#[rkyv(archived = LuPivotsOperation)]
pub struct LuPivots {
    size: u64,
}
impl LuPivotsOperation {
    pub fn size(&self) -> usize {
        count(self.size)
    }
}
gnx_base::array_operation!(LuPivotsOperation, "lu_pivots_to_permutation", |eqn| {
    signature_count(eqn, 1, 1)
});
pub fn lu_pivots_to_permutation(size: usize) -> Op {
    Op::new(LuPivots { size: size as u64 })
}

#[derive(rkyv::Archive, rkyv::Serialize)]
#[rkyv(archived = CustomLinearSolveOperation)]
pub struct CustomLinearSolve {
    parameters: u64,
    symmetric: bool,
}
impl CustomLinearSolveOperation {
    pub fn parameters(&self) -> usize {
        count(self.parameters)
    }
    pub fn symmetric(&self) -> bool {
        self.symmetric
    }
}

gnx_base::array_operation!(CustomLinearSolveOperation, "custom_linear_solve", |op, eqn| {
    let n = eqn.outputs().len();
    let regions = eqn.subexprs();
    if n == 0 || eqn.inputs().len() != op.parameters() + n || !(regions.len() == 2 || regions.len() == 3) {
        return Err(signature("invalid custom_linear_solve arity"));
    }
    for reg in regions {
        let body = reg.expr();
        if !reg.provided_vars().is_empty() || body.inputs().len() != eqn.inputs().len() || body.outputs().len() != n {
            return Err(signature("custom_linear_solve regions need matching parameters and results"));
        }
        for (&result, output) in body.outputs().iter().zip(eqn.outputs()) {
            if body.value_info(result).is_none_or(|info| info != output.info_ref()) {
                return Err(signature("custom_linear_solve result metadata mismatch"));
            }
        }
    }
    Ok(())
}, parameters;
    fn lower(&self, eqn: gnx_expr::HEqn<'_>, ctx: &mut gnx_expr::lower::LoweringContext<'_>) -> Result<()> {
        ctx.lower_closed_regions_apply(eqn)
    }
);

/// The read-only regions are ordered matvec, solve, and optional transpose solve.
/// Each takes explicit parameters followed by the right-hand side or vector.
pub fn custom_linear_solve(parameters: usize, symmetric: bool) -> Op {
    Op::new(CustomLinearSolve {
        parameters: parameters as u64,
        symmetric,
    })
}

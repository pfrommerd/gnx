use crate::ops::{
    contraction::{ConvGeneralDilatedOperation, RaggedDotGeneralOperation, ScaledDotOperation},
    elementwise::{ClampOperation, PolynomialOperation},
    indexing::{DynamicSliceOperation, DynamicUpdateSliceOperation, SelectNOperation},
    linalg::*,
    ordering::*,
};
use gnx_expr::Eqn;
use gnx_expr::pass::Builder;

use gnx_transforms::TransformResult;
use gnx_transforms::map::batch::{BatchRule, Batched, batch_region, materialize};
use gnx_transforms::util::{with_axis, without_axis};

// Lower a mapped operation through scan when a single call cannot consume
// vectorized index or shape parameters directly.
pub(super) fn map_lanes(
    b: &mut Builder,
    size: usize,
    eqn: &Eqn,
    inputs: &[Batched],
) -> TransformResult<Vec<Batched>> {
    let vars = inputs
        .iter()
        .map(|&x| materialize(b, x, size, 0))
        .collect::<TransformResult<Vec<_>>>()?;
    let infos = vars
        .iter()
        .map(|&v| without_axis(b.info(v), 0).map(|(_, info)| info))
        .collect::<TransformResult<Vec<_>>>()?;
    let region = b.region(&[], &infos, |child, _, args| {
        Ok::<_, gnx_transforms::TransformError>(child.emit(
            eqn.op().clone(),
            args,
            eqn.outputs().iter().map(|(_, info)| info.clone()),
            eqn.subexprs().to_vec(),
        ))
    })?;
    let outputs = eqn
        .outputs()
        .iter()
        .map(|(_, info)| with_axis(info, 0, size))
        .collect::<TransformResult<Vec<_>>>()?;
    Ok(b.emit(
        gnx_transforms::cf::scan_op(0, size),
        &vars,
        outputs,
        vec![region],
    )
    .into_iter()
    .map(|v| Batched::at(v, 0))
    .collect())
}

macro_rules! map_by_lane {
    ($($ty:ty),+ $(,)?) => {$(
        impl BatchRule for $ty {
            fn batch(&self, b: &mut Builder, size: usize, eqn: &Eqn, inputs: &[Batched], _: &[Vec<Batched>])
                -> TransformResult<Vec<Batched>>
            {
                map_lanes(b, size, eqn, inputs)
            }
        }
        gnx_expr::register_op_interface!(dyn BatchRule: $ty);
    )+};
}
map_by_lane!(
    DynamicSliceOperation,
    DynamicUpdateSliceOperation,
    SelectNOperation,
    ConvGeneralDilatedOperation,
    ClampOperation,
    PolynomialOperation,
    RaggedDotGeneralOperation,
    ScaledDotOperation,
    CumulativeOperation,
    SortOperation,
    TopKOperation,
    ApproxTopKOperation,
);

macro_rules! batch_matrices {
    ($($ty:ty),+ $(,)?) => {$(
        impl BatchRule for $ty {
            fn batch(&self, b: &mut Builder, size: usize, eqn: &Eqn, inputs: &[Batched], _: &[Vec<Batched>])
                -> TransformResult<Vec<Batched>>
            {
                let vars = inputs.iter().map(|&x| materialize(b, x, size, 0))
                    .collect::<TransformResult<Vec<_>>>()?;
                let output = eqn.outputs().iter().map(|(_, info)| with_axis(info, 0, size))
                    .collect::<TransformResult<Vec<_>>>()?;
                Ok(b.emit(eqn.op().clone(), &vars, output, vec![]).into_iter().map(|v| Batched::at(v, 0)).collect())
            }
        }
        gnx_expr::register_op_interface!(dyn BatchRule: $ty);
    )+};
}
batch_matrices!(
    CholeskyOperation,
    CholeskyUpdateOperation,
    EigOperation,
    EighOperation,
    HessenbergOperation,
    HouseholderProductOperation,
    LuOperation,
    LuPivotsOperation,
    OrmqrOperation,
    QrOperation,
    SchurOperation,
    SvdOperation,
    SymmetricProductOperation,
    TriangularSolveOperation,
    TridiagonalOperation,
    TridiagonalSolveOperation,
);

impl BatchRule for CustomLinearSolveOperation {
    fn batch(
        &self,
        b: &mut Builder,
        size: usize,
        eqn: &Eqn,
        inputs: &[Batched],
        _: &[Vec<Batched>],
    ) -> TransformResult<Vec<Batched>> {
        let vars = inputs
            .iter()
            .map(|&x| materialize(b, x, size, 0))
            .collect::<TransformResult<Vec<_>>>()?;
        let output = eqn
            .outputs()
            .iter()
            .map(|(_, info)| with_axis(info, 0, size))
            .collect::<TransformResult<Vec<_>>>()?;
        let inputs = vec![true; vars.len()];
        let outputs = vec![true; output.len()];
        let regions = eqn
            .subexprs()
            .iter()
            .map(|region| {
                batch_region(b, size, region.expr(), &[], &inputs, Some(&outputs))
                    .map(|(body, _)| body)
            })
            .collect::<TransformResult<Vec<_>>>()?;
        Ok(b.emit(eqn.op().clone(), &vars, output, regions)
            .into_iter()
            .map(|v| Batched::at(v, 0))
            .collect())
    }
}
gnx_expr::register_op_interface!(dyn BatchRule: CustomLinearSolveOperation);

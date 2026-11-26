use crate::ops::broadcast_in_dim_op;
use gnx_transforms::map::def::{defbatch, defbroadcasting, leading, shifted};
use gnx_transforms::rules::batch_reduce;
use gnx_transforms::util::with_axis;

use crate::ops::elementwise::*;

use crate::ops::bitwise::*;

use crate::ops::dtype::*;

use crate::ops::special::*;

defbroadcasting!(
    NegOperation,
    AbsOperation,
    SignOperation,
    MinOperation,
    MaxOperation,
    PowOperation,
    ExpOperation,
    Expm1Operation,
    LogOperation,
    Log1pOperation,
    LogisticOperation,
    SqrtOperation,
    RsqrtOperation,
    CbrtOperation,
    SinOperation,
    CosOperation,
    TanhOperation,
    Atan2Operation,
    EqOperation,
    NeOperation,
    LtOperation,
    LeOperation,
    GtOperation,
    GeOperation,
    IsFiniteOperation,
    CeilOperation,
    FloorOperation,
    SquareOperation,
    AndOperation,
    OrOperation,
    XorOperation,
    NotOperation,
    ClzOperation,
    RealOperation,
    ImagOperation,
    ErfOperation,
);

use crate::ops::reduction::{ReduceWindowOperation, reduce_window_op};
use gnx_base::pass::Builder;
use gnx_expr::Eqn;
use gnx_transforms::map::batch::materialize;
use gnx_transforms::map::{BatchRule, Batched};
use gnx_transforms::{TransformError, TransformResult};

impl BatchRule for ReduceWindowOperation {
    fn batch(
        &self,
        b: &mut Builder,
        size: usize,
        eqn: Eqn<'_>,
        inputs: &[Batched],
        _: &[Vec<Batched>],
    ) -> TransformResult<Vec<Batched>> {
        let mut window = self.window().clone();
        window.dimensions.insert(0, 1);
        window.strides.insert(0, 1);
        window.base_dilation.insert(0, 1);
        window.window_dilation.insert(0, 1);
        window.padding.insert(0, (0, 0));
        batch_reduce(b, size, eqn, inputs, reduce_window_op(window))
    }
}

gnx_expr::register_op_interface!(dyn BatchRule: ReduceWindowOperation);

use crate::ops::{
    indexing::{SliceOperation, slice},
    shape::{PadOperation, pad},
};
struct SliceRule;
impl BatchRule for SliceRule {
    fn batch(
        &self,
        b: &mut Builder,
        size: usize,
        eqn: Eqn<'_>,
        inputs: &[Batched],
        _: &[Vec<Batched>],
    ) -> TransformResult<Vec<Batched>> {
        let mut ranges = vec![(0, size, 1)];
        ranges.extend_from_slice(&eqn.op().downcast_ref::<SliceOperation>().unwrap().ranges());
        gnx_transforms::map::def::at_leading_axis(b, size, eqn, inputs, slice(&ranges))
    }
}

impl BatchRule for PadOperation {
    fn batch(
        &self,
        b: &mut Builder,
        size: usize,
        eqn: Eqn<'_>,
        inputs: &[Batched],
        _: &[Vec<Batched>],
    ) -> TransformResult<Vec<Batched>> {
        let [input, fill] = inputs else {
            return Err(TransformError::invalid("pad signature"));
        };
        if fill.axis.is_some() {
            return super::extended::map_lanes(b, size, eqn, inputs);
        }
        let mut padding = vec![(0, 0, 0)];
        padding.extend_from_slice(&self.padding());
        let value = materialize(b, *input, size, 0)?;
        let output = with_axis(&gnx_transforms::util::output_infos(eqn)[0], 0, size)?;
        Ok(vec![Batched::at(
            b.emit1(pad(&padding), &[value, fill.var], output),
            0,
        )])
    }
}

gnx_expr::register_op_interface!(dyn BatchRule: SliceOperation => SliceRule);
gnx_expr::register_op_interface!(dyn BatchRule: PadOperation);

defbroadcasting!(
    PopulationCountOperation,
    ShiftLeftOperation,
    ShiftRightArithmeticOperation,
    ShiftRightLogicalOperation,
    TanOperation,
    AsinOperation,
    AcosOperation,
    AtanOperation,
    SinhOperation,
    CoshOperation,
    AsinhOperation,
    AcoshOperation,
    AtanhOperation,
    Exp2Operation,
    Log2Operation,
    OneMinusSquareOperation,
    NextafterOperation,
    EqToOperation,
    LtToOperation,
    LeToOperation,
    MulhiOperation,
    ConjOperation,
    ComplexOperation,
    ErfcOperation,
    ErfInvOperation,
    LgammaOperation,
    DigammaOperation,
    BesselI0eOperation,
    BesselI1eOperation,
    IgammaOperation,
    IgammacOperation,
    IgammaGradAOperation,
    PolygammaOperation,
    ZetaOperation,
    RegularizedIncompleteBetaOperation,
);

defbroadcasting!(
    RoundOperation,
    IntegerPowOperation,
    ReducePrecisionOperation
);
defbatch!(BitcastConvertTypeOperation, |_| Ok(
    bitcast_convert_type_op()
));

use crate::ops::{
    contraction::{DotGeneralOperation, dot_general},
    fft::{FftOperation, fft},
};

defbatch!(FftOperation, |op| Ok(fft(op.kind(), &op.lengths())));
defbatch!(DotGeneralOperation, |op| {
    let mut dims = op.dimensions().clone();
    dims.lhs_batch = leading(&dims.lhs_batch);
    dims.rhs_batch = leading(&dims.rhs_batch);
    dims.lhs_contracting = shifted(&dims.lhs_contracting);
    dims.rhs_contracting = shifted(&dims.rhs_contracting);
    Ok(dot_general(dims))
});

use crate::ops::{
    creation::FullOperation,
    shape::{ConcatenateOperation, concatenate},
};
defbatch!(ConcatenateOperation, |op| Ok(concatenate(op.axis() + 1)));

impl BatchRule for FullOperation {
    fn batch(
        &self,
        b: &mut Builder,
        size: usize,
        eqn: Eqn<'_>,
        inputs: &[Batched],
        _: &[Vec<Batched>],
    ) -> TransformResult<Vec<Batched>> {
        let [fill] = inputs else {
            return Err(TransformError::invalid("full signature"));
        };
        let value = materialize(b, *fill, size, 0)?;
        let output = with_axis(&gnx_transforms::util::output_infos(eqn)[0], 0, size)?;
        Ok(vec![Batched::at(
            b.emit1(broadcast_in_dim_op(&[0]), &[value], output),
            0,
        )])
    }
}
gnx_expr::register_op_interface!(dyn BatchRule: FullOperation);

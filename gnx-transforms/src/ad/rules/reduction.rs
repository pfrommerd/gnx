use gnx_common::ops::reduction::{ReduceAddOperation, ReduceOperation, reduce_add};
use gnx_common::ops::{AddOperation, broadcast_in_dim_op};
use gnx_expr::pass::Builder;
use gnx_expr::{Eqn, Expr, Var};

use crate::ad::{Dual, JvpRule, Operand, Tangent, TransposeRule, def::deflinear};
use crate::util::{copy_eqn, dims};
use crate::{TransformError, TransformResult};

/// Forward-mode rule for an axis reduction with a non-additive scalar combiner.
///
/// A provider registers this interface for `ReduceOperation`. The transform
/// crate handles additive reductions directly so it can emit sums on its own.
pub trait ReduceJvpRule {
    fn jvp(
        &self,
        b: &mut Builder,
        eqn: &Eqn,
        axes: &[usize],
        inputs: &[Dual],
        body: &Expr,
    ) -> TransformResult<Vec<Dual>>;
}

pub fn reducer_body(eqn: &Eqn) -> TransformResult<&Expr> {
    let [region] = eqn.subexprs() else {
        return Err(TransformError::invalid("expected one reducer region"));
    };
    Ok(region.expr())
}

pub fn additive_reducer(reducer: &Expr) -> bool {
    let [eqn] = reducer.eqns() else {
        return false;
    };
    reducer.inputs().len() == 2
        && eqn.inputs() == [reducer.inputs()[0].0, reducer.inputs()[1].0]
        && eqn.op().downcast_ref::<AddOperation>().is_some()
        && eqn.outputs().len() == 1
        && reducer.outputs() == [eqn.outputs()[0].0]
}

pub fn copy_reduction_primal(b: &mut Builder, eqn: &Eqn, inputs: &[Dual]) -> Vec<Var> {
    copy_eqn(
        b,
        eqn,
        &inputs.iter().map(|dual| dual.primal).collect::<Vec<_>>(),
        &[vec![]],
    )
}

impl JvpRule for ReduceOperation {
    fn jvp(
        &self,
        b: &mut Builder,
        eqn: &Eqn,
        inputs: &[Dual],
        _: &[Vec<Dual>],
    ) -> TransformResult<Vec<Dual>> {
        let (operands, identities) = inputs.split_at(eqn.outputs().len());
        if identities.iter().any(|value| !value.tangent.is_zero()) {
            return Err(TransformError::invalid(
                "reduce does not differentiate its identities",
            ));
        }
        let reducer = reducer_body(eqn)?;
        if additive_reducer(reducer) {
            let primal = copy_reduction_primal(b, eqn, inputs)[0];
            let tangent = match operands[0].tangent {
                Tangent::Zero => Tangent::Zero,
                Tangent::Var(tangent) => Tangent::Var(reduce_add(b, tangent, self.axes())?),
            };
            return Ok(vec![Dual { primal, tangent }]);
        }
        eqn.op()
            .interface::<dyn ReduceJvpRule>()?
            .jvp(b, eqn, self.axes(), inputs, reducer)
    }
}

impl TransposeRule for ReduceOperation {
    fn transpose(
        &self,
        b: &mut Builder,
        eqn: &Eqn,
        operands: &[Operand],
        _: &[Vec<Var>],
        cotangents: &[Tangent],
    ) -> TransformResult<Vec<Tangent>> {
        let ([Operand::Linear(input), Operand::Residual(_)], [Tangent::Var(ct)]) =
            (operands, cotangents)
        else {
            return Err(TransformError::invalid("cannot transpose this reduction"));
        };
        if !additive_reducer(reducer_body(eqn)?) {
            return Err(TransformError::invalid(
                "expected an additive linear reduction",
            ));
        }
        let mapping = retained_axes(dims(input)?.len(), self.axes());
        let result = b.emit1(broadcast_in_dim_op(&mapping), &[*ct], input.clone());
        Ok(vec![Tangent::Var(result), Tangent::Zero])
    }
}

fn retained_axes(rank: usize, reduced: &[usize]) -> Vec<usize> {
    (0..rank).filter(|axis| !reduced.contains(axis)).collect()
}

deflinear!(ReduceAddOperation, |cx, ct, [input]| {
    let axes = cx.operation::<ReduceAddOperation>()?.axes();
    let mapping = retained_axes(input.dims()?.len(), axes);
    Ok(cx.emit(broadcast_in_dim_op(&mapping), &[ct], input.info()))
});

gnx_expr::register_op_interface!(dyn JvpRule: ReduceOperation);
gnx_expr::register_op_interface!(dyn TransposeRule: ReduceOperation);

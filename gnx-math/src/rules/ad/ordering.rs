use crate::ops::{
    add_op,
    creation::iota,
    elementwise::{abs_op, exp_op, is_finite_op, log1p_op, max_op, min_op, neg_op, sub_op},
    indexing::{BoundsMode, GatherDimensions, GatherParameters, gather, slice},
    mul_op,
    ordering::*,
    reshape_op, rev_op,
    shape::concatenate,
};
use gnx_base::DType;
use gnx_base::pass::Builder;
use gnx_expr::{Eqn, Op, Var};
use gnx_transforms::ad::def::deftranspose;
use gnx_transforms::ad::jvp::{instantiate, jvp_expr};
use gnx_transforms::ad::{Dual, JvpRule, Tangent};
use gnx_transforms::util::{copy_eqn, dims, dtype, info, select};
use gnx_transforms::{TransformError, TransformResult};

fn take_along_axis(b: &mut Builder, x: Var, indices: Var, axis: usize) -> TransformResult<Var> {
    let mut sizes = dims(b.info(x))?;
    let mut shape = dims(b.info(indices))?;
    let output = info(&shape, dtype(b.info(x))?);
    shape.push(1);
    let indices = b.emit1(
        reshape_op(),
        &[indices],
        info(&shape, dtype(b.info(indices))?),
    );
    sizes.fill(1);
    let batch: Vec<_> = (0..sizes.len()).filter(|&i| i != axis).collect();
    let parameters = GatherParameters {
        dimensions: GatherDimensions {
            offset: vec![],
            collapsed: vec![axis],
            start_index_map: vec![axis],
            operand_batch: batch.clone(),
            index_batch: batch,
        },
        slice_sizes: sizes,
        mode: BoundsMode::PromiseInBounds,
        sorted: false,
        unique: true,
    };
    Ok(b.emit1(gather(parameters), &[x, indices], output))
}

fn top_k_jvp(
    b: &mut Builder,
    eqn: Eqn<'_>,
    input: Dual,
    axis: usize,
) -> TransformResult<Vec<Dual>> {
    let outputs = copy_eqn(b, eqn, &[input.primal], &[]);
    let [value, indices] = outputs[..] else {
        return Err(TransformError::invalid(
            "top_k must return values and indices",
        ));
    };
    let tangent = match input.tangent {
        Tangent::Zero => Tangent::Zero,
        Tangent::Var(t) => Tangent::Var(take_along_axis(b, t, indices, axis)?),
    };
    Ok(vec![
        Dual {
            primal: value,
            tangent,
        },
        Dual::constant(indices),
    ])
}

impl JvpRule for TopKOperation {
    fn jvp(
        &self,
        b: &mut Builder,
        eqn: Eqn<'_>,
        inputs: &[Dual],
        _: &[Vec<Dual>],
    ) -> TransformResult<Vec<Dual>> {
        let rank = dims(b.info(inputs[0].primal))?.len();
        top_k_jvp(b, eqn, inputs[0], rank - 1)
    }
}
impl JvpRule for ApproxTopKOperation {
    fn jvp(
        &self,
        b: &mut Builder,
        eqn: Eqn<'_>,
        inputs: &[Dual],
        _: &[Vec<Dual>],
    ) -> TransformResult<Vec<Dual>> {
        top_k_jvp(b, eqn, inputs[0], self.axis())
    }
}

impl JvpRule for SortOperation {
    fn jvp(
        &self,
        b: &mut Builder,
        eqn: Eqn<'_>,
        inputs: &[Dual],
        _: &[Vec<Dual>],
    ) -> TransformResult<Vec<Dual>> {
        let mut keys: Vec<_> = inputs.iter().map(|value| value.primal).collect();
        let shape = dims(b.info(keys[0]))?;
        let indices = info(&shape, DType::I32);
        keys.push(b.emit1(iota(self.axis()), &[], indices.clone()));
        let metadata = (gnx_transforms::util::output_infos(eqn).into_iter()).chain([indices]);
        let mut output = b.emit(eqn.op().to_owned(), &keys, metadata, vec![]);
        let permutation = output.pop().expect("sort index result");
        output
            .into_iter()
            .zip(inputs)
            .map(|(primal, dual)| {
                let tangent = match dual.tangent {
                    Tangent::Zero => Tangent::Zero,
                    Tangent::Var(x) => {
                        Tangent::Var(take_along_axis(b, x, permutation, self.axis())?)
                    }
                };
                Ok(Dual { primal, tangent })
            })
            .collect()
    }
}

fn cumulative_pair(
    kind: CumulativeKind,
    metadata: gnx_expr::VarInfo,
) -> TransformResult<std::sync::Arc<gnx_expr::RootExpr>> {
    let mut b = Builder::new();
    let lhs = b.input(metadata.clone());
    let rhs = b.input(metadata.clone());
    let result = match kind {
        CumulativeKind::Sum => b.emit1(add_op(), &[lhs, rhs], metadata),
        CumulativeKind::Product => b.emit1(mul_op(), &[lhs, rhs], metadata),
        CumulativeKind::Min => b.emit1(min_op(), &[lhs, rhs], metadata),
        CumulativeKind::Max => b.emit1(max_op(), &[lhs, rhs], metadata),
        CumulativeKind::LogSumExp => {
            let max = b.emit1(max_op(), &[lhs, rhs], metadata.clone());
            let diff = b.emit1(sub_op(), &[lhs, rhs], metadata.clone());
            let abs = b.emit1(abs_op(), &[diff], metadata.clone());
            let neg = b.emit1(neg_op(), &[abs], metadata.clone());
            let exp = b.emit1(exp_op(), &[neg], metadata.clone());
            let log = b.emit1(log1p_op(), &[exp], metadata.clone());
            let result = b.emit1(add_op(), &[max, log], metadata.clone());
            let pred = b.emit1(is_finite_op(), &[max], info(&dims(&metadata)?, DType::Bool));
            select(&mut b, pred, result, max)
        }
    };
    Ok(b.finish(vec![result]).build()?)
}

fn apply(b: &mut Builder, op: Op, input: Dual, metadata: gnx_expr::VarInfo) -> Dual {
    let primal = b.emit1(op.clone(), &[input.primal], metadata.clone());
    let tangent = match input.tangent {
        Tangent::Zero => Tangent::Zero,
        Tangent::Var(tangent) => Tangent::Var(b.emit1(op, &[tangent], metadata)),
    };
    Dual { primal, tangent }
}

impl JvpRule for CumulativeOperation {
    fn jvp(
        &self,
        b: &mut Builder,
        eqn: Eqn<'_>,
        inputs: &[Dual],
        _: &[Vec<Dual>],
    ) -> TransformResult<Vec<Dual>> {
        let original = inputs[0];
        let metadata = b.info(original.primal).clone();
        let mut shape = dims(&metadata)?;
        let axis = self.axis();
        let extent = shape[axis];
        let dtype = dtype(&metadata)?;
        if self.kind() == CumulativeKind::Sum {
            return Ok(vec![apply(b, eqn.op().to_owned(), original, metadata)]);
        }
        let mut value = if self.reverse() {
            apply(b, rev_op(&[axis]), original, metadata.clone())
        } else {
            original
        };
        let mut shift = 1;
        while shift < extent {
            let mut ranges: Vec<_> = shape.iter().map(|&n| (0, n, 1)).collect();
            ranges[axis] = (0, shift, 1);
            shape[axis] = shift;
            let head = apply(b, slice(&ranges), value, info(&shape, dtype));
            shape[axis] = extent - shift;
            let pair = info(&shape, dtype);
            ranges[axis] = (0, extent - shift, 1);
            let previous = apply(b, slice(&ranges), value, pair.clone());
            ranges[axis] = (shift, extent, 1);
            let next = apply(b, slice(&ranges), value, pair.clone());
            let tail = jvp_expr(
                b,
                cumulative_pair(self.kind(), pair)?.view(),
                &[],
                &[previous, next],
            )?[0];
            let primal = b.emit1(
                concatenate(axis),
                &[head.primal, tail.primal],
                metadata.clone(),
            );
            let head_ty = b.info(head.primal).clone();
            let tail_ty = b.info(tail.primal).clone();
            let a = instantiate(b, head.tangent, &head_ty);
            let z = instantiate(b, tail.tangent, &tail_ty);
            let tangent = Tangent::Var(b.emit1(concatenate(axis), &[a, z], metadata.clone()));
            value = Dual { primal, tangent };
            shape[axis] = extent;
            shift = shift.saturating_mul(2);
        }
        if self.reverse() {
            value = apply(b, rev_op(&[axis]), value, metadata);
        }
        Ok(vec![value])
    }
}

deftranspose!(CumulativeOperation, |cx, ct, _| {
    let op = cx.operation::<CumulativeOperation>()?;
    if op.kind() != CumulativeKind::Sum {
        return Err(TransformError::invalid("expected cumulative sum"));
    }
    Ok(cx.emit(cumsum(op.axis(), !op.reverse()), &[ct], ct.info()))
});
gnx_expr::register_op_interface!(dyn JvpRule: CumulativeOperation, SortOperation, TopKOperation, ApproxTopKOperation);

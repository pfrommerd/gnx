use crate::ops::{
    contraction::{ConvGeneralDilatedOperation, DotDimensions, DotGeneralOperation, dot_general},
    fft::{FftKind, FftOperation, fft},
};
use gnx_transforms::ad::def::{Arg, Ctx, Val, defbilinear, defjvp, deflinear};
use gnx_transforms::util::{dtype, info};
use gnx_transforms::{TransformError, TransformResult};

fn dot_transpose<'a>(
    cx: &'a Ctx<'a>,
    ct: Val<'a>,
    target: Arg<'a>,
    other: Arg<'a>,
    left: bool,
) -> TransformResult<Val<'a>> {
    let d = cx.operation::<DotGeneralOperation>()?.dimensions();
    let (target_shape, other_shape) = (target.dims()?, other.dims()?);
    let (batch, other_batch, contraction, other_contraction) = if left {
        (
            &d.lhs_batch,
            &d.rhs_batch,
            &d.lhs_contracting,
            &d.rhs_contracting,
        )
    } else {
        (
            &d.rhs_batch,
            &d.lhs_batch,
            &d.rhs_contracting,
            &d.lhs_contracting,
        )
    };
    let kept: Vec<_> = (0..target_shape.len())
        .filter(|i| !batch.contains(i) && !contraction.contains(i))
        .collect();
    let other_kept: Vec<_> = (0..other_shape.len())
        .filter(|i| !other_batch.contains(i) && !other_contraction.contains(i))
        .collect();
    let ct_start = batch.len() + if left { kept.len() } else { 0 };
    let contract = DotDimensions {
        lhs_batch: (0..batch.len()).collect(),
        rhs_batch: other_batch.to_vec(),
        lhs_contracting: (ct_start..ct_start + other_kept.len()).collect(),
        rhs_contracting: other_kept,
    };
    let mut natural: Vec<_> = batch.iter().chain(&kept).copied().collect();
    let mut coupled: Vec<_> = other_contraction
        .iter()
        .copied()
        .zip(contraction.iter().copied())
        .collect();
    coupled.sort_unstable();
    natural.extend(coupled.into_iter().map(|(_, axis)| axis));
    let output_shape: Vec<_> = natural.iter().map(|&axis| target_shape[axis]).collect();
    let result = cx.emit(
        dot_general(contract),
        &[ct, other.val()?],
        info(&output_shape, dtype(&target.info())?),
    );
    let permutation: Vec<_> = (0..target_shape.len())
        .map(|axis| {
            natural
                .iter()
                .position(|&i| i == axis)
                .ok_or_else(|| TransformError::invalid("invalid dot transpose dimensions"))
        })
        .collect::<TransformResult<_>>()?;
    result.transpose(&permutation)
}

defbilinear!(
    DotGeneralOperation,
    |cx, ct, [lhs, rhs]| dot_transpose(cx, ct, lhs, rhs, true),
    |cx, ct, [lhs, rhs]| dot_transpose(cx, ct, rhs, lhs, false)
);

defjvp!(
    ConvGeneralDilatedOperation,
    |cx, t, [_, rhs]| Ok(cx.apply(&[t, rhs])),
    |cx, t, [lhs, _]| Ok(cx.apply(&[lhs, t]))
);

fn fft_constant<'a>(cx: &'a Ctx<'a>, v: f64, like: Val<'a>) -> TransformResult<Val<'a>> {
    let scalar = cx.emit(
        crate::ops::creation::constant(v),
        &[],
        info(&[], gnx_base::DType::F64),
    );
    let scalar = cx.emit(
        crate::ops::dtype::convert_element_type_op(),
        &[scalar],
        info(&[], dtype(&like.info())?),
    );
    Ok(cx.emit(crate::ops::creation::full(), &[scalar], like.info()))
}

fn fft_weights<'a>(cx: &'a Ctx<'a>, value: Val<'a>, size: usize) -> TransformResult<Val<'a>> {
    use crate::ops::{
        bitwise::or_op, creation::iota, dtype::convert_element_type_op, elementwise::eq_op,
    };
    let shape = value.dims()?;
    let ordinal = cx.emit(
        iota(shape.len() - 1),
        &[],
        info(&shape, gnx_base::DType::I64),
    );
    let bool_shape = info(&shape, gnx_base::DType::Bool);
    let first = cx.emit(
        eq_op(),
        &[ordinal, ordinal.zeros_like()],
        bool_shape.clone(),
    );
    let edge = if size % 2 == 0 {
        let last = fft_constant(cx, (size / 2) as f64, ordinal)?;
        let last = cx.emit(eq_op(), &[ordinal, last], bool_shape.clone());
        cx.emit(or_op(), &[first, last], bool_shape)
    } else {
        first
    };
    let edge = cx.emit(convert_element_type_op(), &[edge], value.info());
    Ok(cx.emit(
        sub_op(),
        &[fft_constant(cx, 2., value)?, edge],
        value.info(),
    ))
}

deflinear!(FftOperation, |cx, ct, [input]| {
    let op = cx.operation::<FftOperation>()?;
    match op.kind() {
        FftKind::Fft | FftKind::Ifft => {
            Ok(cx.emit(fft(op.kind(), &op.lengths()), &[ct], input.info()))
        }
        FftKind::Rfft | FftKind::Irfft if op.lengths().is_empty() => Ok(ct),
        FftKind::Rfft => {
            use crate::ops::{dtype::conj_op, elementwise::div_op};
            let sz = *op.lengths().last().unwrap();
            let n = op.lengths().iter().map(|&size| size as f64).product();
            let conj = cx.emit(conj_op(), &[ct], ct.info());
            let weight = fft_weights(cx, ct, sz)?;
            let scaled = cx.emit(div_op(), &[conj, weight], ct.info());
            let transformed = cx.emit(fft(FftKind::Irfft, &op.lengths()), &[scaled], input.info());
            Ok(transformed * fft_constant(cx, n, transformed)?)
        }
        FftKind::Irfft => {
            use crate::ops::dtype::conj_op;
            let sz = *op.lengths().last().unwrap();
            let n = op
                .lengths()
                .iter()
                .map(|&size| size as f64)
                .product::<f64>();
            let transformed = cx.emit(fft(FftKind::Rfft, &op.lengths()), &[ct], input.info());
            let weighted = transformed
                * fft_weights(cx, transformed, sz)?
                * fft_constant(cx, 1. / n, transformed)?;
            Ok(cx.emit(conj_op(), &[weighted], input.info()))
        }
    }
});

use crate::ops::{elementwise::sub_op, linalg::CustomLinearSolveOperation};
use gnx_base::pass::Builder;
use gnx_expr::{Eqn, Var};
use gnx_transforms::ad::jvp::{instantiate, jvp_expr};
use gnx_transforms::ad::{Dual, JvpRule, Operand, Tangent, TransposeRule};

impl JvpRule for CustomLinearSolveOperation {
    fn jvp(
        &self,
        b: &mut Builder,
        eqn: Eqn<'_>,
        inputs: &[Dual],
        _: &[Vec<Dual>],
    ) -> TransformResult<Vec<Dual>> {
        let primals: Vec<_> = inputs.iter().map(|value| value.primal).collect();
        let result =
            gnx_transforms::util::copy_eqn(b, eqn, &primals, &vec![vec![]; eqn.subexprs().len()]);
        let (parameters, rhs) = inputs.split_at(self.parameters());
        let matvec: Vec<_> = parameters
            .iter()
            .copied()
            .chain(result.iter().copied().map(Dual::constant))
            .collect();
        let perturbation = jvp_expr(
            b,
            (eqn.subexprs().next())
                .ok_or_else(|| TransformError::invalid("custom_linear_solve requires a matvec"))?
                .expr(),
            &[],
            &matvec,
        )?;
        let mut new_operands = primals[..self.parameters()].to_vec();
        for (&rhs, applied) in rhs.iter().zip(perturbation) {
            let metadata = b.info(rhs.primal).clone();
            let rhs_t = instantiate(b, rhs.tangent, &metadata);
            let lhs_t = instantiate(b, applied.tangent, &metadata);
            new_operands.push(b.emit1(sub_op(), &[rhs_t, lhs_t], metadata));
        }
        let tangents = gnx_transforms::util::copy_eqn(
            b,
            eqn,
            &new_operands,
            &vec![vec![]; eqn.subexprs().len()],
        );
        Ok(result
            .into_iter()
            .zip(tangents)
            .map(|(primal, tangent)| Dual {
                primal,
                tangent: Tangent::Var(tangent),
            })
            .collect())
    }
}

impl TransposeRule for CustomLinearSolveOperation {
    fn transpose(
        &self,
        b: &mut Builder,
        eqn: Eqn<'_>,
        operands: &[Operand],
        _: &[Vec<Var>],
        cotangents: &[Tangent],
    ) -> TransformResult<Vec<Tangent>> {
        let index = if self.symmetric() { 1 } else { 2 };
        let region = eqn.subexprs().nth(index).ok_or_else(|| {
            TransformError::invalid("custom_linear_solve requires a transpose solver")
        })?;
        let mut args = Vec::new();
        for parameter in &operands[..self.parameters()] {
            let Operand::Residual(var) = parameter else {
                return Err(TransformError::invalid(
                    "linear solve parameters must be residuals",
                ));
            };
            args.push(*var);
        }
        for (&cotangent, info) in (cotangents.iter()).zip(gnx_transforms::util::output_infos(eqn)) {
            args.push(instantiate(b, cotangent, &info));
        }
        let out = b.inline(region.expr(), region.provided_symbols(), &[], &args)?;
        let mut result = vec![Tangent::Zero; self.parameters()];
        result.extend(out.into_iter().zip(&operands[self.parameters()..]).map(
            |(value, operand)| {
                if operand.is_linear() {
                    Tangent::Var(value)
                } else {
                    Tangent::Zero
                }
            },
        ));
        Ok(result)
    }
}

gnx_expr::register_op_interface!(dyn JvpRule: CustomLinearSolveOperation);
gnx_expr::register_op_interface!(dyn TransposeRule: CustomLinearSolveOperation);

use crate::ops::contraction::{ConvAxes, ConvKernelAxes, ConvParameters, conv_general_dilated};
use crate::ops::dtype::convert_element_type_op;
use gnx_transforms::ad::def::deftranspose;

fn conv_padding(
    p: &ConvParameters,
    lhs: &[usize],
    rhs: &[usize],
    output: &[usize],
    lhs_gradient: bool,
) -> TransformResult<Vec<(i64, i64)>> {
    let dilated = |size: usize, dilation: usize| {
        if size == 0 {
            return Ok(0);
        }
        (size as i128 - 1)
            .checked_mul(dilation as i128)
            .and_then(|n| n.checked_add(1))
            .and_then(|n| i64::try_from(n).ok())
            .map(i128::from)
            .ok_or_else(|| TransformError::invalid("convolution dilation extent overflow"))
    };
    (0..p.strides.len())
        .map(|axis| {
            let x = dilated(lhs[p.input.spatial[axis]], p.lhs_dilation[axis])?;
            let y = dilated(rhs[p.kernel.spatial[axis]], p.rhs_dilation[axis])?;
            let z = dilated(output[p.output.spatial[axis]], p.strides[axis])?;
            let original = p.padding[axis].0 as i128;
            let (lo, hi) = if lhs_gradient {
                let lo = y - original - 1;
                (lo, x + y - 1 - z - lo)
            } else {
                (original, z - x + y - original - 1)
            };
            Ok((
                i64::try_from(lo)
                    .map_err(|_| TransformError::invalid("convolution padding overflow"))?,
                i64::try_from(hi)
                    .map_err(|_| TransformError::invalid("convolution padding overflow"))?,
            ))
        })
        .collect()
}

fn shift_group<'a>(
    value: Val<'a>,
    group: usize,
    from: usize,
    to: usize,
) -> TransformResult<Val<'a>> {
    let mut shape = value.dims()?;
    if group == 0 || shape[from] % group != 0 {
        return Err(TransformError::invalid(
            "invalid convolution group dimensions",
        ));
    }
    let dtype = dtype(&value.info())?;
    shape[from] /= group;
    shape.insert(from, group);
    let value = value.reshape(info(&shape, dtype));
    let mut perm: Vec<_> = (0..shape.len()).filter(|&axis| axis != from).collect();
    perm.insert(to, from);
    shape.remove(from);
    shape[to] = shape[to]
        .checked_mul(group)
        .ok_or_else(|| TransformError::invalid("convolution group extent overflow"))?;
    Ok(value.transpose(&perm)?.reshape(info(&shape, dtype)))
}

fn conv_cast<'a>(cx: &'a Ctx<'a>, value: Val<'a>, ty: gnx_base::DType) -> TransformResult<Val<'a>> {
    if dtype(&value.info())? == ty {
        return Ok(value);
    }
    Ok(cx.emit(
        convert_element_type_op(),
        &[value],
        info(&value.dims()?, ty),
    ))
}

fn conv_lhs<'a>(
    cx: &'a Ctx<'a>,
    ct: Val<'a>,
    lhs: Arg<'a>,
    rhs: Arg<'a>,
) -> TransformResult<Val<'a>> {
    let p = cx.operation::<ConvGeneralDilatedOperation>()?.parameters();
    let mut shape = lhs.dims()?;
    let pad = conv_padding(&p, &shape, &rhs.dims()?, &ct.dims()?, true)?;
    let groups = p.feature_groups.max(p.batch_groups);
    let mut rhs = rhs.val()?;
    if groups > 1 {
        rhs = shift_group(rhs, groups, p.kernel.output_feature, p.kernel.input_feature)?;
    }
    let rhs = conv_cast(cx, rhs.rev(&p.kernel.spatial)?, dtype(&ct.info())?)?;
    let conv = ConvParameters {
        input: p.output.clone(),
        output: p.input.clone(),
        kernel: ConvKernelAxes {
            output_feature: p.kernel.input_feature,
            input_feature: p.kernel.output_feature,
            spatial: p.kernel.spatial.clone(),
        },
        strides: p.lhs_dilation.clone(),
        padding: pad,
        lhs_dilation: p.strides.clone(),
        rhs_dilation: p.rhs_dilation.clone(),
        feature_groups: groups,
        batch_groups: 1,
    };
    if p.batch_groups > 1 {
        shape[p.input.batch] /= p.batch_groups;
        shape[p.input.feature] *= p.batch_groups;
    }
    let result = cx.emit(
        conv_general_dilated(conv),
        &[ct, rhs],
        info(&shape, dtype(&ct.info())?),
    );
    let result = if p.batch_groups > 1 {
        shift_group(result, p.batch_groups, p.input.feature, p.input.batch)?
    } else {
        result
    };
    conv_cast(cx, result, dtype(&lhs.info())?)
}

fn conv_rhs<'a>(
    cx: &'a Ctx<'a>,
    ct: Val<'a>,
    lhs: Arg<'a>,
    rhs: Arg<'a>,
) -> TransformResult<Val<'a>> {
    let p = cx.operation::<ConvGeneralDilatedOperation>()?.parameters();
    let ct_shape = ct.dims()?;
    if ct_shape.contains(&0) {
        return Ok(cx.zeros(&rhs.info()));
    }
    let rhs_shape = rhs.dims()?;
    let pad = conv_padding(&p, &lhs.dims()?, &rhs_shape, &ct_shape, false)?;
    let ty = dtype(&ct.info())?;
    let lhs = conv_cast(cx, lhs.val()?, ty)?;
    let conv = ConvParameters {
        input: ConvAxes {
            batch: p.input.feature,
            feature: p.input.batch,
            spatial: p.input.spatial.clone(),
        },
        kernel: ConvKernelAxes {
            output_feature: p.output.feature,
            input_feature: p.output.batch,
            spatial: p.output.spatial.clone(),
        },
        output: ConvAxes {
            batch: p.kernel.input_feature,
            feature: p.kernel.output_feature,
            spatial: p.kernel.spatial.clone(),
        },
        strides: p.rhs_dilation.clone(),
        padding: pad,
        lhs_dilation: p.lhs_dilation.clone(),
        rhs_dilation: p.strides.clone(),
        feature_groups: p.batch_groups,
        batch_groups: p.feature_groups,
    };
    let value = cx.emit(conv_general_dilated(conv), &[lhs, ct], info(&rhs_shape, ty));
    conv_cast(cx, value, dtype(&rhs.info())?)
}
deftranspose!(
    ConvGeneralDilatedOperation,
    |cx, ct, [lhs, rhs]| conv_lhs(cx, ct, lhs, rhs),
    |cx, ct, [lhs, rhs]| conv_rhs(cx, ct, lhs, rhs)
);

use crate::ops::contraction::{RaggedDotDimensions, RaggedDotGeneralOperation, ragged_dot_general};

defjvp!(
    RaggedDotGeneralOperation,
    |cx, t, [_, y, groups]| Ok(cx.apply(&[t, y, groups])),
    |cx, t, [x, _, groups]| Ok(cx.apply(&[x, t, groups])),
    None
);

fn ragged_gradient<'a>(
    cx: &'a Ctx<'a>,
    ct: Val<'a>,
    lhs: Arg<'a>,
    rhs: Arg<'a>,
    groups: Arg<'a>,
    left: bool,
) -> TransformResult<Val<'a>> {
    let cfg = cx.operation::<RaggedDotGeneralOperation>()?.dimensions();
    let d = &cfg.dot;
    let group = cfg.rhs_group.ok_or_else(|| {
        TransformError::invalid("ragged dot transpose requires a noncontracting ragged dimension")
    })?;
    let left_shape = lhs.dims()?;
    let right_shape = rhs.dims()?;
    let x_kept: Vec<_> = (0..left_shape.len())
        .filter(|axis| !d.lhs_batch.contains(axis) && !d.lhs_contracting.contains(axis))
        .collect();
    let y_kept: Vec<_> = (0..right_shape.len())
        .filter(|axis| {
            *axis != group && !d.rhs_batch.contains(axis) && !d.rhs_contracting.contains(axis)
        })
        .collect();
    let b = d.lhs_batch.len();
    let batch_out = (0..b).collect::<Vec<_>>();
    let result_lhs = (b..b + x_kept.len()).collect::<Vec<_>>();
    let result_rhs = (b + x_kept.len()..b + x_kept.len() + y_kept.len()).collect::<Vec<_>>();
    let mut contraction: Vec<_> = d
        .lhs_contracting
        .iter()
        .copied()
        .zip(d.rhs_contracting.iter().copied())
        .collect();
    let (parameters, natural, target, variables) = if left {
        let ragged = x_kept
            .iter()
            .position(|&axis| axis == cfg.lhs_ragged)
            .ok_or_else(|| TransformError::invalid("ragged left axis is not a kept dimension"))?;
        contraction.sort_by_key(|&(_, y)| y);
        let order = d
            .lhs_batch
            .iter()
            .chain(&x_kept)
            .copied()
            .chain(contraction.into_iter().map(|(x, _)| x))
            .collect::<Vec<_>>();
        (
            RaggedDotDimensions {
                dot: DotDimensions {
                    lhs_batch: batch_out,
                    rhs_batch: d.rhs_batch.clone(),
                    lhs_contracting: result_rhs,
                    rhs_contracting: y_kept,
                },
                lhs_ragged: b + ragged,
                rhs_group: Some(group),
            },
            order,
            lhs,
            [ct, rhs.val()?, groups.val()?],
        )
    } else {
        contraction.sort_by_key(|&(x, _)| x);
        let order = std::iter::once(group)
            .chain(d.rhs_batch.iter().copied())
            .chain(contraction.into_iter().map(|(_, y)| y))
            .chain(y_kept)
            .collect::<Vec<_>>();
        (
            RaggedDotDimensions {
                dot: DotDimensions {
                    lhs_batch: d.lhs_batch.clone(),
                    rhs_batch: batch_out,
                    lhs_contracting: x_kept,
                    rhs_contracting: result_lhs,
                },
                lhs_ragged: cfg.lhs_ragged,
                rhs_group: None,
            },
            order,
            rhs,
            [lhs.val()?, ct, groups.val()?],
        )
    };
    let source = target.dims()?;
    let output_shape = natural.iter().map(|&axis| source[axis]).collect::<Vec<_>>();
    let ty = dtype(&target.info())?;
    let result = cx.emit(
        ragged_dot_general(parameters),
        &variables,
        info(&output_shape, ty),
    );
    result.transpose(&gnx_transforms::util::inverse(&natural))
}
deftranspose!(
    RaggedDotGeneralOperation,
    |cx, ct, [lhs, rhs, groups]| ragged_gradient(cx, ct, lhs, rhs, groups, true),
    |cx, ct, [lhs, rhs, groups]| ragged_gradient(cx, ct, lhs, rhs, groups, false),
    None
);

use crate::ops::{
    creation::FullOperation,
    indexing::{DynamicSliceOperation, SliceOperation, dynamic_update_slice, slice},
    shape::PadOperation,
    shape::pad,
};

use gnx_transforms::ad::def::{Ctx, Val, defjvp, deflinear, deftranspose};
use gnx_transforms::util::{dtype, info};

deflinear!(SliceOperation, |cx, ct, [input]| {
    let input_shape = input.dims()?;
    let output_shape = ct.dims()?;
    let ranges = cx.operation::<SliceOperation>()?.ranges();
    if output_shape.contains(&0) {
        return Ok(cx.zeros(&input.info()));
    }
    let padding = ranges
        .iter()
        .zip(input_shape.iter().zip(output_shape))
        .map(|(&(start, _, stride), (&size, len))| {
            let end = start + (len - 1) * stride + 1;
            (start as i64, (size - end) as i64, stride - 1)
        })
        .collect::<Vec<_>>();
    let zero = cx.zeros(&info(&[], dtype(&ct.info())?));
    Ok(cx.emit(pad(&padding), &[ct, zero], input.info()))
});

fn unpad<'a>(
    cx: &'a Ctx<'a>,
    ct: Val<'a>,
    shape: &[usize],
    metadata: gnx_expr::ValueInfo,
) -> gnx_transforms::TransformResult<Val<'a>> {
    let padded = ct.dims()?;
    let operation = cx.operation::<PadOperation>()?;
    let mut ranges = Vec::with_capacity(shape.len());
    let mut cropped_shape = Vec::with_capacity(shape.len());
    let mut restore = Vec::with_capacity(shape.len());
    for ((&(low, _, interior), &size), extent) in operation.padding().iter().zip(shape).zip(padded)
    {
        if size == 0 || extent == 0 {
            return Ok(cx.zeros(&metadata));
        }
        let step = interior as i128 + 1;
        let low = low as i128;
        let begin = (((-low).div_euclid(step)) + i128::from((-low).rem_euclid(step) != 0))
            .clamp(0, size as i128) as usize;
        let end = ((extent as i128 - 1 - low).div_euclid(step) + 1).clamp(0, size as i128) as usize;
        if end <= begin {
            return Ok(cx.zeros(&metadata));
        }
        ranges.push((
            (low + begin as i128 * step) as usize,
            (low + (end - 1) as i128 * step + 1) as usize,
            interior + 1,
        ));
        cropped_shape.push(end - begin);
        restore.push((begin as i64, (size - end) as i64, 0));
    }
    let ty = dtype(&metadata)?;
    let cropped = cx.emit(slice(&ranges), &[ct], info(&cropped_shape, ty));
    Ok(cx.emit(
        pad(&restore),
        &[cropped, cx.zeros(&info(&[], ty))],
        metadata,
    ))
}

deflinear!(
    PadOperation,
    |cx, ct, [input, _]| unpad(cx, ct, &input.dims()?, input.info()),
    |cx, ct, [input, _]| {
        let input_shape = input.dims()?;
        let output_shape = ct.dims()?;
        let mask_info = |shape: &[usize]| info(shape, gnx_array::DType::Bool);
        let ones = cx.emit(crate::ops::ones_op(), &[], mask_info(&input_shape));
        let false_value = cx.zeros(&mask_info(&[]));
        let mask = cx.emit(
            pad(cx.operation::<PadOperation>()?.padding()),
            &[ones, false_value],
            mask_info(&output_shape),
        );
        cx.select(mask, ct.zeros_like(), ct)
            .reduce_sum(&(0..output_shape.len()).collect::<Vec<_>>())
    }
);

defjvp!(FullOperation, |cx, t, _| Ok(cx.apply(&[t])));
deftranspose!(FullOperation, |_, ct, _| ct
    .reduce_sum(&(0..ct.dims()?.len()).collect::<Vec<_>>()));

impl gnx_transforms::ad::JvpRule for DynamicSliceOperation {
    fn jvp(
        &self,
        b: &mut gnx_expr::pass::Builder,
        eqn: &gnx_expr::Eqn,
        inputs: &[gnx_transforms::ad::Dual],
        captures: &[Vec<gnx_transforms::ad::Dual>],
    ) -> gnx_transforms::TransformResult<Vec<gnx_transforms::ad::Dual>> {
        let mut vars: Vec<_> = inputs.iter().map(|value| value.primal).collect();
        let primal = gnx_transforms::util::copy_eqn(b, eqn, &vars, &[])[0];
        let tangent = match inputs[0].tangent {
            gnx_transforms::ad::Tangent::Zero => gnx_transforms::ad::Tangent::Zero,
            gnx_transforms::ad::Tangent::Var(value) => {
                vars[0] = value;
                gnx_transforms::ad::Tangent::Var(
                    gnx_transforms::util::copy_eqn(b, eqn, &vars, &[])[0],
                )
            }
        };
        debug_assert!(captures.is_empty());
        Ok(vec![gnx_transforms::ad::Dual { primal, tangent }])
    }
}

impl gnx_transforms::ad::TransposeRule for DynamicSliceOperation {
    fn transpose(
        &self,
        b: &mut gnx_expr::pass::Builder,
        eqn: &gnx_expr::Eqn,
        operands: &[gnx_transforms::ad::Operand],
        _: &[Vec<gnx_expr::Var>],
        cotangents: &[gnx_transforms::ad::Tangent],
    ) -> gnx_transforms::TransformResult<Vec<gnx_transforms::ad::Tangent>> {
        use gnx_transforms::ad::{Operand, Tangent};
        let ([Operand::Linear(input), starts @ ..], [Tangent::Var(ct)]) = (operands, cotangents)
        else {
            return Err(gnx_transforms::TransformError::invalid(
                "dynamic slice transpose signature",
            ));
        };
        let mut args = vec![gnx_transforms::util::zeros(b, input), *ct];
        for start in starts {
            let Operand::Residual(var) = start else {
                return Err(gnx_transforms::TransformError::invalid(
                    "dynamic slice starts must be residuals",
                ));
            };
            args.push(*var);
        }
        let mut result = vec![Tangent::Zero; eqn.inputs().len()];
        result[0] = Tangent::Var(b.emit1(dynamic_update_slice(), &args, input.clone()));
        Ok(result)
    }
}

gnx_expr::register_op_interface!(dyn gnx_transforms::ad::JvpRule: DynamicSliceOperation);
gnx_expr::register_op_interface!(dyn gnx_transforms::ad::TransposeRule: DynamicSliceOperation);

use crate::ops::shape::{ConcatenateOperation, split};
use gnx_expr::pass::Builder;
use gnx_expr::{Eqn, Var};
use gnx_transforms::ad::jvp::instantiate;
use gnx_transforms::ad::{Dual, JvpRule, Operand, Tangent, TransposeRule};
use gnx_transforms::{TransformError, TransformResult};

impl JvpRule for ConcatenateOperation {
    fn jvp(
        &self,
        b: &mut Builder,
        eqn: &Eqn,
        inputs: &[Dual],
        _: &[Vec<Dual>],
    ) -> TransformResult<Vec<Dual>> {
        let primals: Vec<_> = inputs.iter().map(|value| value.primal).collect();
        let result = gnx_transforms::util::copy_eqn(b, eqn, &primals, &[])[0];
        let tangents: Vec<_> = inputs
            .iter()
            .map(|value| {
                let info = b.info(value.primal).clone();
                instantiate(b, value.tangent, &info)
            })
            .collect();
        let tangent = gnx_transforms::util::copy_eqn(b, eqn, &tangents, &[])[0];
        Ok(vec![Dual {
            primal: result,
            tangent: Tangent::Var(tangent),
        }])
    }
}

impl TransposeRule for ConcatenateOperation {
    fn transpose(
        &self,
        b: &mut Builder,
        _: &Eqn,
        operands: &[Operand],
        _: &[Vec<Var>],
        ct: &[Tangent],
    ) -> TransformResult<Vec<Tangent>> {
        let [Tangent::Var(ct)] = ct else {
            return Err(TransformError::invalid("concatenate transpose signature"));
        };
        let sizes = operands
            .iter()
            .map(|value| {
                let info = match value {
                    Operand::Linear(info) => info,
                    Operand::Residual(var) => b.info(*var),
                };
                gnx_transforms::util::dims(info)?
                    .get(self.axis())
                    .copied()
                    .ok_or_else(|| TransformError::invalid("concatenate axis out of range"))
            })
            .collect::<TransformResult<Vec<_>>>()?;
        Ok(split(b, *ct, self.axis(), &sizes)?
            .into_iter()
            .zip(operands)
            .map(|(var, operand)| {
                if operand.is_linear() {
                    Tangent::Var(var)
                } else {
                    Tangent::Zero
                }
            })
            .collect())
    }
}

gnx_expr::register_op_interface!(dyn JvpRule: ConcatenateOperation);
gnx_expr::register_op_interface!(dyn TransposeRule: ConcatenateOperation);

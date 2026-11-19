use std::io;

use gnx_expr::Eqn;
use gnx_math::ops::{creation::*, indexing::*, shape::*};
use xla::{PrimitiveType, XlaOp};

use super::{
    LoweringContext, XlaLowering, dimensions, element, indices, invalid, native, result_info, unary,
};

impl XlaLowering for SliceOperation {
    fn lower_xla(
        &self,
        _: &LoweringContext<'_>,
        _: &Eqn,
        inputs: &[XlaOp],
    ) -> io::Result<Vec<XlaOp>> {
        let mut value = unary("slice", inputs)?.clone();
        for (axis, &(start, end, stride)) in self.ranges().iter().enumerate() {
            let range = indices(&[start, end, stride, axis])?;
            value = native(value.slice_in_dim(range[0], range[1], range[2], range[3]))?;
        }
        Ok(vec![value])
    }
}

impl XlaLowering for ConcatenateOperation {
    fn lower_xla(
        &self,
        _: &LoweringContext<'_>,
        _: &Eqn,
        inputs: &[XlaOp],
    ) -> io::Result<Vec<XlaOp>> {
        let (first, rest) = inputs
            .split_first()
            .ok_or_else(|| invalid("empty concatenate"))?;
        Ok(vec![native(
            first.concat_in_dim(rest, indices(&[self.axis()])?[0]),
        )?])
    }
}

impl XlaLowering for PadOperation {
    fn lower_xla(
        &self,
        _: &LoweringContext<'_>,
        _: &Eqn,
        inputs: &[XlaOp],
    ) -> io::Result<Vec<XlaOp>> {
        let [input, fill] = inputs else {
            return Err(invalid("pad signature"));
        };
        let padding = self
            .padding()
            .iter()
            .map(|&(low, high, interior)| Ok((low, high, indices(&[interior])?[0])))
            .collect::<io::Result<Vec<_>>>()?;
        Ok(vec![native(input.pad(fill, &padding))?])
    }
}

impl XlaLowering for IotaOperation {
    fn lower_xla(
        &self,
        ctx: &LoweringContext<'_>,
        eqn: &Eqn,
        _: &[XlaOp],
    ) -> io::Result<Vec<XlaOp>> {
        let info = result_info(eqn)?;
        Ok(vec![native(ctx.builder().iota(
            element(info.dtype()),
            &dimensions(info)?,
            indices(&[self.axis()])?[0],
        ))?])
    }
}

impl XlaLowering for FullOperation {
    fn lower_xla(
        &self,
        _: &LoweringContext<'_>,
        eqn: &Eqn,
        inputs: &[XlaOp],
    ) -> io::Result<Vec<XlaOp>> {
        let fill = unary("full", inputs)?;
        Ok(vec![native(
            fill.broadcast_in_dim(&dimensions(result_info(eqn)?)?, &[]),
        )?])
    }
}

fn wrap_starts(operand: &XlaOp, starts: &[XlaOp]) -> io::Result<Vec<XlaOp>> {
    let shape = native(operand.array_shape())?;
    if shape.dims().len() != starts.len() {
        return Err(invalid("dynamic index count must equal operand rank"));
    }
    let builder = operand.builder();
    starts
        .iter()
        .zip(shape.dims())
        .map(|(start, &dim)| {
            let ty = native(start.ty())?;
            if matches!(
                ty,
                PrimitiveType::S8 | PrimitiveType::S16 | PrimitiveType::S32 | PrimitiveType::S64
            ) {
                let start = native(start.convert(PrimitiveType::S64))?;
                let zero = native(builder.c0(0_i64))?;
                let dim = native(builder.c0(dim))?;
                let negative = native(start.lt(&zero))?;
                let wrapped = native(start.add_(&dim))?;
                native(negative.select(&wrapped, &start))
            } else {
                Ok(start.clone())
            }
        })
        .collect()
}

impl XlaLowering for DynamicSliceOperation {
    fn lower_xla(
        &self,
        _: &LoweringContext<'_>,
        eqn: &Eqn,
        inputs: &[XlaOp],
    ) -> io::Result<Vec<XlaOp>> {
        let [operand, starts @ ..] = inputs else {
            return Err(invalid("dynamic_slice signature"));
        };
        let starts = wrap_starts(operand, starts)?;
        Ok(vec![native(
            operand.dynamic_slice(&starts, &dimensions(result_info(eqn)?)?),
        )?])
    }
}

impl XlaLowering for DynamicUpdateSliceOperation {
    fn lower_xla(
        &self,
        _: &LoweringContext<'_>,
        _: &Eqn,
        inputs: &[XlaOp],
    ) -> io::Result<Vec<XlaOp>> {
        let [operand, update, starts @ ..] = inputs else {
            return Err(invalid("dynamic_update_slice signature"));
        };
        let starts = wrap_starts(operand, starts)?;
        Ok(vec![native(operand.dynamic_update_slice(update, &starts))?])
    }
}

impl XlaLowering for SelectNOperation {
    fn lower_xla(
        &self,
        ctx: &LoweringContext<'_>,
        _: &Eqn,
        inputs: &[XlaOp],
    ) -> io::Result<Vec<XlaOp>> {
        let [selector, first, rest @ ..] = inputs else {
            return Err(invalid("select_n signature"));
        };
        let ty = native(selector.ty())?;
        if ty == PrimitiveType::Pred {
            return match rest {
                [] => Ok(vec![first.clone()]),
                [when_true] => Ok(vec![native(selector.select(when_true, first))?]),
                _ => Err(invalid(
                    "select_n with a boolean selector has at most two cases",
                )),
            };
        }
        let shape = native(first.shape())?;
        let mut selected = first.clone();
        for (index, case) in rest.iter().enumerate() {
            if native(case.shape())? != shape {
                return Err(invalid("select_n cases must have the same shape and dtype"));
            }
            let threshold = native(native(ctx.builder().c0((index + 1) as i64))?.convert(ty))?;
            let pred = native(selector.ge(&threshold))?;
            selected = native(pred.select(case, &selected))?;
        }
        Ok(vec![selected])
    }
}

crate::register_xla_lowering!(
    SliceOperation,
    ConcatenateOperation,
    PadOperation,
    IotaOperation,
    FullOperation,
    DynamicSliceOperation,
    DynamicUpdateSliceOperation,
    SelectNOperation,
);

impl XlaLowering for ConstantOperation {
    fn lower_xla(
        &self,
        ctx: &LoweringContext<'_>,
        eqn: &Eqn,
        _: &[XlaOp],
    ) -> io::Result<Vec<XlaOp>> {
        let scalar = native(ctx.builder().c0(self.value()))?;
        Ok(vec![native(scalar.convert(
            element(result_info(eqn)?.dtype()).primitive_type(),
        ))?])
    }
}
crate::register_xla_lowering!(ConstantOperation);

impl XlaLowering for ScalarBoundOperation {
    fn lower_xla(
        &self,
        ctx: &LoweringContext<'_>,
        eqn: &Eqn,
        _: &[XlaOp],
    ) -> io::Result<Vec<XlaOp>> {
        let ty = element(result_info(eqn)?.dtype());
        Ok(vec![if self.lower() {
            native(ctx.builder().min_value(ty))?
        } else {
            native(ctx.builder().max_value(ty))?
        }])
    }
}
crate::register_xla_lowering!(ScalarBoundOperation);

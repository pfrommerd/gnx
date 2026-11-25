use std::io;

use super::{LoweringContext, XlaLowering, indices, invalid, items, native, unary};
use gnx_expr::Eqn;
use gnx_math::ops::ordering::*;
use xla::{ElementType, XlaBuilder, XlaOp};

impl XlaLowering for CumulativeOperation {
    fn lower_xla(
        &self,
        ctx: &LoweringContext<'_>,
        _: Eqn<'_>,
        inputs: &[XlaOp],
    ) -> io::Result<Vec<XlaOp>> {
        let x = unary("cumulative", inputs)?;
        let shape = native(x.array_shape())?;
        let axis = self.axis();
        let size = *shape
            .dims()
            .get(axis)
            .ok_or_else(|| invalid("cumulative axis out of range"))?;
        if size < 2 {
            return Ok(vec![x.clone()]);
        }
        let ty = shape.ty();
        let reduction = XlaBuilder::new("cumulative_combiner");
        let lhs = native(reduction.parameter(0, ty, &[], "lhs"))?;
        let rhs = native(reduction.parameter(1, ty, &[], "rhs"))?;
        let b = ctx.builder();
        let (value, identity) = match self.kind() {
            CumulativeKind::Sum if ty == ElementType::Pred => {
                (native(lhs.or(&rhs))?, native(b.zero(ty))?)
            }
            CumulativeKind::Sum => (native(lhs.add_(&rhs))?, native(b.zero(ty))?),
            CumulativeKind::Product if ty == ElementType::Pred => {
                (native(lhs.and(&rhs))?, native(b.one(ty))?)
            }
            CumulativeKind::Product => (native(lhs.mul_(&rhs))?, native(b.one(ty))?),
            CumulativeKind::Min => (native(lhs.min(&rhs))?, native(b.max_value(ty))?),
            CumulativeKind::Max => (native(lhs.max(&rhs))?, native(b.min_value(ty))?),
            CumulativeKind::LogSumExp => {
                let max = native(lhs.max(&rhs))?;
                let diff = native(native(lhs.sub_(&rhs))?.abs())?;
                let finite = native(max.is_finite())?;
                let value = native(max.add_(&native(native(native(diff.neg())?.exp())?.log1p())?))?;
                (
                    native(finite.select(&value, &max))?,
                    native(b.min_value(ty))?,
                )
            }
        };
        let combiner = native(value.build())?;
        let rank = shape.dims().len();
        let mut window = vec![1; rank];
        let mut padding = vec![(0, 0); rank];
        window[axis] = size;
        padding[axis] = if self.reverse() {
            (0, size - 1)
        } else {
            (size - 1, 0)
        };
        let units = vec![1; rank];
        Ok(vec![native(XlaOp::reduce_window(
            inputs,
            &[identity],
            &combiner,
            &window,
            &units,
            &units,
            &units,
            &padding,
        ))?])
    }
}

fn canonical_sort_key(value: &XlaOp) -> io::Result<XlaOp> {
    use xla::PrimitiveType::*;
    let ty = native(value.ty())?;
    if !matches!(ty, F16 | Bf16 | F32 | F64 | F8E4M3FN | F8E5M2 | F8E8M0FNU) {
        return Ok(value.clone());
    }
    let zero = native(
        value
            .builder()
            .zero(ty.element_type().map_err(|e| invalid(e.to_string()))?),
    )?;
    let is_zero = native(value.eq(&zero))?;
    let value = native(is_zero.select(&zero, value))?;
    let nan = native(native(value.builder().c0(f64::NAN))?.convert(ty))?;
    let is_nan = native(value.ne(&value))?;
    native(is_nan.select(&nan, &value))
}

impl XlaLowering for SortOperation {
    fn lower_xla(
        &self,
        _: &LoweringContext<'_>,
        _: Eqn<'_>,
        inputs: &[XlaOp],
    ) -> io::Result<Vec<XlaOp>> {
        let cb = XlaBuilder::new("sort_comparator");
        let params = inputs
            .iter()
            .enumerate()
            .map(|(i, operand)| {
                let ty = native(operand.array_shape())?.ty();
                Ok((
                    native(cb.parameter(2 * i as i64, ty, &[], "left"))?,
                    native(cb.parameter(2 * i as i64 + 1, ty, &[], "right"))?,
                ))
            })
            .collect::<io::Result<Vec<_>>>()?;
        let mut less = native(cb.zero(ElementType::Pred))?;
        for (lhs, rhs) in params.iter().take(self.keys()).rev() {
            let mut extend = |lhs: &XlaOp, rhs: &XlaOp| -> io::Result<()> {
                let lhs = canonical_sort_key(lhs)?;
                let rhs = canonical_sort_key(rhs)?;
                let equal = native(lhs.eq_to(&rhs))?;
                let comparison = native(lhs.lt_to(&rhs))?;
                less = native(comparison.or(&native(equal.and(&less))?))?;
                Ok(())
            };
            if matches!(
                native(lhs.ty())?,
                xla::PrimitiveType::C64 | xla::PrimitiveType::C128
            ) {
                extend(&native(lhs.imag())?, &native(rhs.imag())?)?;
                extend(&native(lhs.real())?, &native(rhs.real())?)?;
            } else {
                extend(lhs, rhs)?;
            }
        }
        let comp = native(less.build())?;
        let sorted = native(XlaOp::sort_variadic(
            inputs,
            &comp,
            indices(&[self.axis()])?[0],
            self.stable(),
        ))?;
        if inputs.len() == 1 {
            Ok(vec![sorted])
        } else {
            items(&sorted, inputs.len())
        }
    }
}

impl XlaLowering for TopKOperation {
    fn lower_xla(
        &self,
        _: &LoweringContext<'_>,
        _: Eqn<'_>,
        inputs: &[XlaOp],
    ) -> io::Result<Vec<XlaOp>> {
        let result = native(unary("top_k", inputs)?.top_k(indices(&[self.k()])?[0], true))?;
        items(&result, 2)
    }
}

impl XlaLowering for ApproxTopKOperation {
    fn lower_xla(
        &self,
        _: &LoweringContext<'_>,
        _: Eqn<'_>,
        inputs: &[XlaOp],
    ) -> io::Result<Vec<XlaOp>> {
        let params = indices(&[self.k(), self.axis()])?;
        let original = self
            .original_size()
            .map(|n| indices(&[n]).map(|n| n[0]))
            .transpose()?
            .unwrap_or(-1);
        let result = native(unary("approx_top_k", inputs)?.approx_top_k_with_options(
            params[0],
            params[1],
            self.recall(),
            self.largest(),
            self.aggregate(),
            original,
        ))?;
        items(&result, 2)
    }
}

crate::register_xla_lowering!(
    CumulativeOperation,
    SortOperation,
    TopKOperation,
    ApproxTopKOperation
);

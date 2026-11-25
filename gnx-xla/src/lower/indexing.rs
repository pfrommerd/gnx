use std::io;

use gnx_expr::Eqn;
use gnx_math::ops::indexing::{BoundsMode, GatherOperation, ScatterOperation, ScatterUpdate};
use xla::{ElementType, PrimitiveType, XlaBuilder, XlaComputation, XlaOp};

use super::{LoweringContext, XlaLowering, indices, invalid, lower_body, native};

fn vector_axis(op: &XlaOp) -> io::Result<i64> {
    native(op.array_shape())?
        .dims()
        .len()
        .checked_sub(1)
        .and_then(|axis| i64::try_from(axis).ok())
        .ok_or_else(|| invalid("indices need a trailing index-vector dimension"))
}

fn index_bounds(
    indices: &XlaOp,
    sizes: &[i64],
    windows: &[i64],
    mapping: &[usize],
) -> io::Result<(XlaOp, XlaOp, XlaOp)> {
    let bounds = mapping
        .iter()
        .map(|&axis| {
            let (&size, &window) = sizes
                .get(axis)
                .zip(windows.get(axis))
                .ok_or_else(|| invalid("index mapping axis out of range"))?;
            size.checked_sub(window)
                .filter(|&n| n >= 0)
                .ok_or_else(|| invalid("indexed window exceeds source dimension"))
        })
        .collect::<io::Result<Vec<_>>>()?;
    use PrimitiveType::*;
    let ty = match native(indices.ty())? {
        U8 | U16 | U32 | U64 => U64,
        S8 | S16 | S32 | S64 => S64,
        _ => return Err(invalid("indices must be integer arrays")),
    };
    let axis = vector_axis(indices)?;
    let shape = native(indices.array_shape())?;
    let builder = indices.builder();
    let lower = native(native(builder.c0(0_i64))?.convert(ty))?;
    let upper = native(native(builder.constant_r1(&bounds))?.convert(ty))?;
    let upper = native(upper.broadcast_in_dim(shape.dims(), &[axis]))?;
    Ok((native(indices.convert(ty))?, lower, upper))
}

impl XlaLowering for GatherOperation {
    fn lower_xla(
        &self,
        _: &LoweringContext<'_>,
        _: Eqn<'_>,
        inputs: &[XlaOp],
    ) -> io::Result<Vec<XlaOp>> {
        let [operand, start, more @ ..] = inputs else {
            return Err(invalid("gather signature"));
        };
        let p = self.parameters();
        let d = &p.dimensions;
        let axis = vector_axis(start)?;
        let windows = indices(&p.slice_sizes)?;
        let result = native(operand.gather_with_batching(
            start,
            &indices(&d.offset)?,
            &indices(&d.collapsed)?,
            &indices(&d.start_index_map)?,
            Some(axis),
            &windows,
            &indices(&d.operand_batch)?,
            &indices(&d.index_batch)?,
            p.sorted,
        ))?;
        if p.mode != BoundsMode::FillOrDrop {
            return Ok(vec![result]);
        }
        let [fill] = more else {
            return Err(invalid("gather fill value required"));
        };
        let shape = native(operand.array_shape())?;
        let (start, lower, upper) =
            index_bounds(start, shape.dims(), &windows, &d.start_index_map)?;
        let mask = native(native(start.ge(&lower))?.and(&native(start.le(&upper))?))?;
        let combiner = XlaBuilder::new("gather_in_bounds");
        let lhs = native(combiner.parameter(0, ElementType::Pred, &[], "lhs"))?;
        let rhs = native(combiner.parameter(1, ElementType::Pred, &[], "rhs"))?;
        let and = native(native(lhs.and(&rhs))?.build())?;
        let init = native(mask.builder().one(ElementType::Pred))?;
        let mask = native(mask.reduce(init, and, &[axis], false))?;
        let output = native(result.array_shape())?;
        let batch: Vec<_> = (0..output.dims().len())
            .filter(|axis| !d.offset.contains(axis))
            .collect();
        let mask = native(mask.broadcast_in_dim(output.dims(), &indices(&batch)?))?;
        let fill = native(fill.broadcast_in_dim(output.dims(), &[]))?;
        Ok(vec![native(mask.select(&result, &fill))?])
    }
}

fn scatter_computation(
    operation: ScatterUpdate,
    eqn: Eqn<'_>,
    ty: ElementType,
) -> io::Result<XlaComputation> {
    let builder = XlaBuilder::new("scatter_update");
    let lhs = native(builder.parameter(0, ty, &[], "original"))?;
    let rhs = native(builder.parameter(1, ty, &[], "update"))?;
    let result = match operation {
        ScatterUpdate::Replace => rhs,
        ScatterUpdate::Add => native(lhs.add_(&rhs))?,
        ScatterUpdate::Subtract => native(lhs.sub_(&rhs))?,
        ScatterUpdate::Multiply => native(lhs.mul_(&rhs))?,
        ScatterUpdate::Min => native(lhs.min(&rhs))?,
        ScatterUpdate::Max => native(lhs.max(&rhs))?,
        ScatterUpdate::Apply => {
            let mut regions = eqn.subexprs();
            let (Some(region), None) = (regions.next(), regions.next()) else {
                return Err(invalid("scatter_apply needs an update region"));
            };
            let results = lower_body(&builder, region.expr(), &[], &[lhs])?;
            let [result] = results
                .try_into()
                .map_err(|_| invalid("scatter_apply reducer must return one scalar"))?;
            result
        }
    };
    native(result.build())
}

impl XlaLowering for ScatterOperation {
    fn lower_xla(
        &self,
        _: &LoweringContext<'_>,
        eqn: Eqn<'_>,
        inputs: &[XlaOp],
    ) -> io::Result<Vec<XlaOp>> {
        let [operand, start, updates] = inputs else {
            return Err(invalid("scatter signature"));
        };
        let p = self.parameters();
        let d = &p.dimensions;
        let axis = vector_axis(start)?;
        let shape = native(operand.array_shape())?;
        let computation = scatter_computation(self.update(), eqn, shape.ty())?;
        let start = if p.mode == BoundsMode::Clip {
            let update = native(updates.array_shape())?;
            let mut axes = d.update_window.iter();
            let windows = (0..shape.dims().len())
                .map(|axis| {
                    if d.inserted_window.contains(&axis) || d.operand_batch.contains(&axis) {
                        Ok(1)
                    } else {
                        axes.next()
                            .and_then(|&axis| update.dims().get(axis))
                            .copied()
                            .ok_or_else(|| invalid("invalid scatter update window dimensions"))
                    }
                })
                .collect::<io::Result<Vec<_>>>()?;
            let (start, lower, upper) = index_bounds(start, shape.dims(), &windows, &d.index_map)?;
            native(start.clamp(&lower, &upper))?
        } else {
            start.clone()
        };
        Ok(vec![native(operand.scatter_with_batching(
            &start,
            updates,
            &computation,
            &indices(&d.update_window)?,
            &indices(&d.inserted_window)?,
            &indices(&d.index_map)?,
            axis,
            &indices(&d.operand_batch)?,
            &indices(&d.index_batch)?,
            p.sorted && p.mode != BoundsMode::Clip,
            p.unique,
        ))?])
    }
}

crate::register_xla_lowering!(GatherOperation, ScatterOperation);

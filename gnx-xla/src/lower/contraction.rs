use std::io;

use super::{LoweringContext, XlaLowering, element, indices, invalid, native, result_info, unary};
use gnx_expr::Eqn;
use gnx_math::ops::{contraction::*, fft::*};
use xla::XlaOp;

impl XlaLowering for DotGeneralOperation {
    fn lower_xla(
        &self,
        _: &LoweringContext<'_>,
        eqn: &Eqn,
        inputs: &[XlaOp],
    ) -> io::Result<Vec<XlaOp>> {
        let [lhs, rhs] = inputs else {
            return Err(invalid("dot_general signature"));
        };
        let d = self.dimensions();
        let output = element(result_info(eqn)?.dtype()).primitive_type();
        Ok(vec![native(lhs.dot_general_with_type(
            rhs,
            &indices(&d.lhs_contracting)?,
            &indices(&d.rhs_contracting)?,
            &indices(&d.lhs_batch)?,
            &indices(&d.rhs_batch)?,
            output,
        ))?])
    }
}

impl XlaLowering for ConvGeneralDilatedOperation {
    fn lower_xla(
        &self,
        _: &LoweringContext<'_>,
        _: &Eqn,
        inputs: &[XlaOp],
    ) -> io::Result<Vec<XlaOp>> {
        let [lhs, rhs] = inputs else {
            return Err(invalid("convolution signature"));
        };
        let p = self.parameters();
        let dim = |axis| -> io::Result<i64> { Ok(indices(&[axis])?[0]) };
        Ok(vec![native(lhs.conv_general_dilated(
            rhs,
            &indices(&p.strides)?,
            &p.padding,
            &indices(&p.lhs_dilation)?,
            &indices(&p.rhs_dilation)?,
            dim(p.input.batch)?,
            dim(p.input.feature)?,
            &indices(&p.input.spatial)?,
            dim(p.output.batch)?,
            dim(p.output.feature)?,
            &indices(&p.output.spatial)?,
            dim(p.kernel.input_feature)?,
            dim(p.kernel.output_feature)?,
            &indices(&p.kernel.spatial)?,
            dim(p.feature_groups)?,
            dim(p.batch_groups)?,
            false,
        ))?])
    }
}

impl XlaLowering for FftOperation {
    fn lower_xla(
        &self,
        _: &LoweringContext<'_>,
        _: &Eqn,
        inputs: &[XlaOp],
    ) -> io::Result<Vec<XlaOp>> {
        let operand = unary("fft", inputs)?;
        if self.lengths().is_empty() {
            return Ok(vec![operand.clone()]);
        }
        let mode = match self.kind() {
            FftKind::Fft => 0,
            FftKind::Ifft => 1,
            FftKind::Rfft => 2,
            FftKind::Irfft => 3,
        };
        Ok(vec![native(operand.fft(mode, &indices(self.lengths())?))?])
    }
}
crate::register_xla_lowering!(
    DotGeneralOperation,
    ConvGeneralDilatedOperation,
    FftOperation
);

fn scale_operand(
    operand: &XlaOp,
    scale: Option<&XlaOp>,
    contracting: &[usize],
) -> io::Result<XlaOp> {
    use xla::PrimitiveType::Bf16;
    let dims = native(operand.array_shape())?;
    let operand = native(operand.convert(Bf16))?;
    let Some(scale) = scale else {
        return Ok(operand);
    };
    let scale_dims = native(scale.array_shape())?;
    if scale_dims.dims().len() != dims.dims().len() {
        return Err(invalid("scale rank must equal operand rank"));
    }
    let mut expanded = Vec::new();
    let mut mapping = Vec::new();
    for (axis, (&operand, &scale)) in dims.dims().iter().zip(scale_dims.dims()).enumerate() {
        mapping.push(expanded.len() as i64);
        expanded.push(scale);
        if contracting.contains(&axis) {
            if scale <= 0 || operand % scale != 0 || operand / scale < 2 {
                return Err(invalid("invalid scale size on contracting dimension"));
            }
            expanded.push(operand / scale);
        } else if operand != scale {
            return Err(invalid("scale must match noncontracting dimensions"));
        }
    }
    let scale = native(scale.convert(Bf16))?;
    let expanded = native(scale.broadcast_in_dim(&expanded, &mapping))?;
    let expanded = native(expanded.reshape(dims.dims()))?;
    native(operand.mul_(&expanded))
}

impl XlaLowering for ScaledDotOperation {
    fn lower_xla(
        &self,
        _: &LoweringContext<'_>,
        eqn: &Eqn,
        inputs: &[XlaOp],
    ) -> io::Result<Vec<XlaOp>> {
        let [lhs, rhs, scales @ ..] = inputs else {
            return Err(invalid("scaled dot signature"));
        };
        let d = self.dimensions();
        let mut scales = scales.iter();
        let lhs = scale_operand(
            lhs,
            if self.lhs_scaled() {
                scales.next()
            } else {
                None
            },
            &d.lhs_contracting,
        )?;
        let rhs = scale_operand(
            rhs,
            if self.rhs_scaled() {
                scales.next()
            } else {
                None
            },
            &d.rhs_contracting,
        )?;
        lower_dot(&lhs, &rhs, d, eqn)
    }
}

fn lower_dot(lhs: &XlaOp, rhs: &XlaOp, d: &DotDimensions, eqn: &Eqn) -> io::Result<Vec<XlaOp>> {
    let output = element(result_info(eqn)?.dtype()).primitive_type();
    Ok(vec![native(lhs.dot_general_with_type(
        rhs,
        &indices(&d.lhs_contracting)?,
        &indices(&d.rhs_contracting)?,
        &indices(&d.lhs_batch)?,
        &indices(&d.rhs_batch)?,
        output,
    ))?])
}

fn sorted_broadcast(input: &XlaOp, output: &[i64], mapping: &[usize]) -> io::Result<XlaOp> {
    let mut perm: Vec<_> = (0..mapping.len()).collect();
    perm.sort_by_key(|&axis| mapping[axis]);
    let input = native(input.transpose(&indices(&perm)?))?;
    let sorted: Vec<_> = perm.iter().map(|&axis| mapping[axis]).collect();
    native(input.broadcast_in_dim(output, &indices(&sorted)?))
}

fn expand_ragged(x: &XlaOp, groups: &XlaOp, axis: usize, prefix: &[usize]) -> io::Result<XlaOp> {
    let x_shape = native(x.array_shape())?;
    let group_shape = native(groups.array_shape())?;
    let &count = group_shape
        .dims()
        .last()
        .ok_or_else(|| invalid("group sizes need a group axis"))?;
    if axis >= x_shape.dims().len() {
        return Err(invalid("ragged axis out of range"));
    }
    let shape: Vec<_> = std::iter::once(count)
        .chain(x_shape.dims().iter().copied())
        .collect();
    let output_map: Vec<_> = (1..shape.len() as i64).collect();
    let expanded = native(x.broadcast_in_dim(&shape, &output_map))?;
    let ty = x_shape.ty();
    let zero = native(native(x.builder().zero(ty))?.broadcast_in_dim(&shape, &[]))?;
    if count == 0 {
        return Ok(zero);
    }
    let group_axis = group_shape.dims().len() - 1;
    let blend = xla::XlaBuilder::new("group_sizes_sum");
    let lhs = native(blend.parameter(0, group_shape.ty(), &[], "lhs"))?;
    let rhs = native(blend.parameter(1, group_shape.ty(), &[], "rhs"))?;
    let reduce = native(native(lhs.add_(&rhs))?.build())?;
    let start = native(x.builder().zero(group_shape.ty()))?;
    let rank = group_shape.dims().len();
    let mut windows = vec![1; rank];
    let unit = vec![1; rank];
    let mut padding = vec![(0, 0); rank];
    windows[group_axis] = count;
    padding[group_axis] = (count - 1, 0);
    let end = native(XlaOp::reduce_window(
        std::slice::from_ref(groups),
        &[start],
        &reduce,
        &windows,
        &unit,
        &unit,
        &unit,
        &padding,
    ))?;
    let start = native(end.sub_(groups))?;
    let mut map = if rank == 1 {
        vec![]
    } else {
        if rank != prefix.len() + 1 {
            return Err(invalid("group sizes prefix rank"));
        }
        prefix.iter().map(|&dim| dim + 1).collect()
    };
    map.push(0);
    let end = sorted_broadcast(&end, &shape, &map)?;
    let start = sorted_broadcast(&start, &shape, &map)?;
    let position = native(
        x.builder()
            .iota(group_shape.ty(), &shape, (axis + 1) as i64),
    )?;
    let mask = native(native(position.ge(&start))?.and(&native(position.lt(&end))?))?;
    native(mask.select(&expanded, &zero))
}

impl XlaLowering for RaggedDotGeneralOperation {
    fn lower_xla(
        &self,
        _: &LoweringContext<'_>,
        eqn: &Eqn,
        inputs: &[XlaOp],
    ) -> io::Result<Vec<XlaOp>> {
        let [lhs, rhs, groups] = inputs else {
            return Err(invalid("ragged dot signature"));
        };
        let p = self.dimensions();
        let d = &p.dot;
        let axis = p.lhs_ragged;
        if d.lhs_batch.contains(&axis) {
            return lower_dot(lhs, rhs, d, eqn);
        }
        let shift = |dims: &[usize]| dims.iter().map(|axis| axis + 1).collect::<Vec<_>>();
        if let Some(contracting) = d.lhs_contracting.iter().position(|&i| i == axis) {
            let rhs_axis = *d
                .rhs_contracting
                .get(contracting)
                .ok_or_else(|| invalid("ragged dot contracting arity"))?;
            let l_prefix: Vec<_> = d
                .lhs_batch
                .iter()
                .chain(&d.lhs_contracting[..contracting])
                .copied()
                .collect();
            let r_prefix: Vec<_> = d
                .rhs_batch
                .iter()
                .chain(&d.rhs_contracting[..contracting])
                .copied()
                .collect();
            let lhs = expand_ragged(lhs, groups, axis, &l_prefix)?;
            let rhs = expand_ragged(rhs, groups, rhs_axis, &r_prefix)?;
            let dimensions = DotDimensions {
                lhs_contracting: shift(&d.lhs_contracting),
                rhs_contracting: shift(&d.rhs_contracting),
                lhs_batch: std::iter::once(0).chain(shift(&d.lhs_batch)).collect(),
                rhs_batch: std::iter::once(0).chain(shift(&d.rhs_batch)).collect(),
            };
            lower_dot(&lhs, &rhs, &dimensions, eqn)
        } else {
            let rhs_group = p
                .rhs_group
                .ok_or_else(|| invalid("missing RHS group axis"))?;
            let l_prefix: Vec<_> = d
                .lhs_batch
                .iter()
                .copied()
                .chain(
                    (0..axis)
                        .filter(|i| !d.lhs_batch.contains(i) && !d.lhs_contracting.contains(i)),
                )
                .collect();
            let lhs = expand_ragged(lhs, groups, axis, &l_prefix)?;
            let dimensions = DotDimensions {
                lhs_contracting: shift(&d.lhs_contracting).into_iter().chain([0]).collect(),
                rhs_contracting: d
                    .rhs_contracting
                    .iter()
                    .copied()
                    .chain([rhs_group])
                    .collect(),
                lhs_batch: shift(&d.lhs_batch),
                rhs_batch: d.rhs_batch.clone(),
            };
            lower_dot(&lhs, rhs, &dimensions, eqn)
        }
    }
}

crate::register_xla_lowering!(ScaledDotOperation, RaggedDotGeneralOperation);

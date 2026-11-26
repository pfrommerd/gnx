use super::*;
pub use ops::contraction::{ConvAxes, ConvKernelAxes, ConvParameters};

primitive!(/// General N-dimensional dilated convolution with explicit layouts and grouping.
    conv_general_dilated, try_conv_general_dilated(x:&Array, kernel:&Array, p:ConvParameters)->Array);
pub fn try_conv_general_dilated(x: &Array, kernel: &Array, p: ConvParameters) -> Result<Array> {
    let d = x.fixed_dims()?;
    let k = kernel.fixed_dims()?;
    let rank = d.len();
    let valid = |spatial: &[usize], a: usize, b: usize| {
        let mut axes = spatial.to_vec();
        axes.extend([a, b]);
        axes.sort_unstable();
        axes.into_iter().eq(0..rank)
    };
    if rank < 3
        || k.len() != rank
        || x.dtype() != kernel.dtype()
        || x.dtype().is_key()
        || !valid(&p.input.spatial, p.input.batch, p.input.feature)
        || !valid(
            &p.kernel.spatial,
            p.kernel.input_feature,
            p.kernel.output_feature,
        )
        || !valid(&p.output.spatial, p.output.batch, p.output.feature)
    {
        return Err(signature("invalid convolution operand layout"));
    }
    let n = rank - 2;
    if [&p.strides, &p.lhs_dilation, &p.rhs_dilation]
        .iter()
        .any(|v| v.len() != n || v.contains(&0))
        || p.padding.len() != n
    {
        return Err(signature("invalid convolution window"));
    }
    let input = d[p.input.feature];
    let batch = d[p.input.batch];
    let output = k[p.kernel.output_feature];
    let fg = p.feature_groups;
    let bg = p.batch_groups;
    if fg == 0
        || bg == 0
        || (bg > 1 && fg > 1)
        || input % fg != 0
        || input / fg != k[p.kernel.input_feature]
        || output % fg != 0
        || output % bg != 0
        || batch % bg != 0
    {
        return Err(signature("invalid convolution groups"));
    }
    let mut shape = vec![0; rank];
    shape[p.output.batch] = batch / bg;
    shape[p.output.feature] = output;
    for i in 0..n {
        let extent = |s: usize, d: usize| -> Result<i128> {
            if s == 0 {
                return Ok(0);
            }
            Ok((s - 1)
                .checked_mul(d)
                .and_then(|v| v.checked_add(1))
                .ok_or_else(|| signature("dilated extent overflow"))? as i128)
        };
        let width = extent(k[p.kernel.spatial[i]], p.rhs_dilation[i])?;
        if width == 0 {
            return Err(signature("empty convolution kernel"));
        }
        let padded = extent(d[p.input.spatial[i]], p.lhs_dilation[i])?
            + i128::from(p.padding[i].0)
            + i128::from(p.padding[i].1);
        shape[p.output.spatial[i]] = if padded < width {
            0
        } else {
            usize::try_from((padded - width) / p.strides[i] as i128 + 1)
                .map_err(|_| signature("convolution extent overflow"))?
        };
    }
    Ok(invoke_array(
        ops::contraction::conv_general_dilated(p),
        &[x, kernel],
        &ArrayInfo::new(shape, x.dtype()),
    ))
}

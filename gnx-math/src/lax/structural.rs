use super::*;
use gnx_expr::{Invocation, VarInfo};
pub use ops::contraction::DotDimensions;

fn fixed(x: &Array) -> Result<Vec<usize>> {
    ops::fixed_dims(&ArrayInfo::new(x.shape(), x.dtype()))
        .map_err(|_| signature("operation requires fixed dimensions"))
}

/// A rank-zero constant, converted from `f64` to `dtype`.
pub fn scalar(value: f64, dtype: DType) -> Array {
    invoke_array(
        ops::creation::constant(value),
        &[],
        &ArrayInfo::new(Shape::fixed([]), dtype),
    )
}
/// A zero-filled array of the specified shape and dtype.
pub fn zeros(shape: impl IntoIterator<Item = usize>, dtype: DType) -> Array {
    invoke_array(
        ops::zeros_op(),
        &[],
        &ArrayInfo::new(Shape::fixed(shape), dtype),
    )
}

primitive!(/// Coordinates along `axis` in a fixed output shape.
    iota, try_iota(shape: &[usize], axis: usize, dtype: DType) -> Array);
/// Checked variant of [`iota`].
pub fn try_iota(shape: &[usize], axis: usize, dtype: DType) -> Result<Array> {
    if axis >= shape.len() || dtype == DType::Bool {
        return Err(signature(
            "iota requires an in-range axis and numeric dtype",
        ));
    }
    Ok(invoke_array(
        ops::creation::iota(axis),
        &[],
        &ArrayInfo::new(Shape::fixed(shape.iter().copied()), dtype),
    ))
}
primitive!(/// Slice one fixed-size axis with a half-open interval.
    slice_axis, try_slice_axis(x: &Array, axis: usize, start: usize, end: usize) -> Array);
/// Checked variant of [`slice_axis`].
pub fn try_slice_axis(x: &Array, axis: usize, start: usize, end: usize) -> Result<Array> {
    let mut dims = fixed(x)?;
    if axis >= dims.len() || start > end || end > dims[axis] {
        return Err(signature("invalid slice interval or axis"));
    }
    let mut ranges: Vec<_> = dims.iter().map(|&n| (0, n, 1)).collect();
    ranges[axis] = (start, end, 1);
    dims[axis] = end - start;
    Ok(invoke_array(
        ops::indexing::slice(&ranges),
        &[x],
        &ArrayInfo::new(Shape::fixed(dims), x.dtype()),
    ))
}
primitive!(/// Join fixed-shape arrays along an existing axis.
    concatenate, try_concatenate(xs: &[&Array], axis: usize) -> Array);
/// Checked variant of [`concatenate`].
pub fn try_concatenate(xs: &[&Array], axis: usize) -> Result<Array> {
    let Some(x) = xs.first() else {
        return Err(signature("concatenate needs an operand"));
    };
    let mut dims = fixed(x)?;
    if axis >= dims.len() {
        return Err(signature("concatenate axis out of range"));
    }
    for y in &xs[1..] {
        let other = fixed(y)?;
        if y.dtype() != x.dtype()
            || other.len() != dims.len()
            || other
                .iter()
                .zip(&dims)
                .enumerate()
                .any(|(i, (a, b))| i != axis && a != b)
        {
            return Err(signature("incompatible concatenate operands"));
        }
        dims[axis] = dims[axis]
            .checked_add(other[axis])
            .ok_or_else(|| signature("concatenate size overflow"))?;
    }
    Ok(invoke_array(
        ops::shape::concatenate(axis),
        xs,
        &ArrayInfo::new(Shape::fixed(dims), x.dtype()),
    ))
}
fn check_slice(dims: &[usize], sizes: &[usize], starts: &[&Array]) -> Result<()> {
    if dims.len() != sizes.len()
        || dims.len() != starts.len()
        || dims.iter().zip(sizes).any(|(d, s)| s > d)
    {
        return Err(signature("slice sizes must fit the operand shape"));
    }
    if starts.iter().any(|x| {
        !x.shape().dims().is_empty()
            || !elementwise::integer(x.dtype())
            || x.dtype() != starts[0].dtype()
    }) {
        return Err(signature(
            "slice starts must be integer scalars of one dtype",
        ));
    }
    Ok(())
}
primitive!(/// Extract a fixed-size slice starting at runtime scalar indices.
    dynamic_slice, try_dynamic_slice(x: &Array, starts: &[&Array], sizes: &[usize]) -> Array);
/// Checked variant of [`dynamic_slice`].
pub fn try_dynamic_slice(x: &Array, starts: &[&Array], sizes: &[usize]) -> Result<Array> {
    check_slice(&fixed(x)?, sizes, starts)?;
    let mut inputs = vec![x];
    inputs.extend_from_slice(starts);
    Ok(invoke_array(
        ops::indexing::dynamic_slice(),
        &inputs,
        &ArrayInfo::new(Shape::fixed(sizes.iter().copied()), x.dtype()),
    ))
}
primitive!(/// Replace a slice starting at runtime scalar indices and return the updated array.
    dynamic_update_slice, try_dynamic_update_slice(x: &Array, update: &Array, starts: &[&Array]) -> Array);
/// Checked variant of [`dynamic_update_slice`].
pub fn try_dynamic_update_slice(x: &Array, update: &Array, starts: &[&Array]) -> Result<Array> {
    if x.dtype() != update.dtype() {
        return Err(signature("slice update dtype must match operand"));
    }
    check_slice(&fixed(x)?, &fixed(update)?, starts)?;
    let mut inputs = vec![x, update];
    inputs.extend_from_slice(starts);
    Ok(invoke_array(
        ops::indexing::dynamic_update_slice(),
        &inputs,
        &ArrayInfo::new(x.shape(), x.dtype()),
    ))
}
primitive!(/// Generalized dot product. Output axes are batch, free lhs, then free rhs axes.
    dot_general, try_dot_general(lhs: &Array, rhs: &Array, axes: DotDimensions) -> Array);
/// Checked variant of [`dot_general`].
pub fn try_dot_general(lhs: &Array, rhs: &Array, axes: DotDimensions) -> Result<Array> {
    let (a, b) = (fixed(lhs)?, fixed(rhs)?);
    if lhs.dtype() != rhs.dtype()
        || axes.lhs_batch.len() != axes.rhs_batch.len()
        || axes.lhs_contracting.len() != axes.rhs_contracting.len()
    {
        return Err(signature(
            "incompatible dot_general operands or axis counts",
        ));
    }
    let mut used_a = vec![false; a.len()];
    let mut used_b = vec![false; b.len()];
    for (&i, &j) in axes
        .lhs_batch
        .iter()
        .zip(&axes.rhs_batch)
        .chain(axes.lhs_contracting.iter().zip(&axes.rhs_contracting))
    {
        if i >= a.len() || j >= b.len() || used_a[i] || used_b[j] || a[i] != b[j] {
            return Err(signature("invalid or incompatible dot_general axes"));
        }
        used_a[i] = true;
        used_b[j] = true;
    }
    let dims = axes
        .lhs_batch
        .iter()
        .map(|&i| a[i])
        .chain(
            a.iter()
                .zip(used_a)
                .filter_map(|(&n, used)| (!used).then_some(n)),
        )
        .chain(
            b.iter()
                .zip(used_b)
                .filter_map(|(&n, used)| (!used).then_some(n)),
        );
    let info = ArrayInfo::new(Shape::fixed(dims), lhs.dtype());
    Ok(invoke_array(
        ops::contraction::dot_general(axes),
        &[lhs, rhs],
        &info,
    ))
}
primitive!(/// Largest `k` values along the last axis, and their `i32` indices.
    top_k, try_top_k(x: &Array, k: usize) -> (Array, Array));
/// Checked variant of [`top_k`].
pub fn try_top_k(x: &Array, k: usize) -> Result<(Array, Array)> {
    let mut dims = fixed(x)?;
    let last = dims
        .last_mut()
        .ok_or_else(|| signature("top_k needs a non-scalar operand"))?;
    if k > *last || matches!(x.dtype(), DType::C64 | DType::C128) {
        return Err(signature("invalid top_k extent or dtype"));
    }
    *last = k;
    let shape = Shape::fixed(dims);
    let info = |dtype| VarInfo::new(ArrayInfo::new(shape.clone(), dtype));
    let outputs = Invocation::invoke(
        ops::ordering::top_k(k),
        vec![],
        vec![x.tracer().clone()],
        vec![info(x.dtype()), info(DType::I32)],
    );
    let mut arrays = outputs
        .into_iter()
        .map(|v| v.try_cast::<Array>().expect("top_k produces arrays").into());
    Ok((arrays.next().unwrap(), arrays.next().unwrap()))
}

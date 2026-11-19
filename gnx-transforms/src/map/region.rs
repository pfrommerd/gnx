//! Record `map.map` around an already-traced body. The jit-based [`map`]
//! wrapper is built on this.
//!
//! [`map`]: super::map

use gnx_expr::{Kind, LoweringError, Trace};

use super::MapParameters;
use crate::body::{Body, BodySignature, record, reject_cell_writes, signature, trace_infos};
use crate::cf::expect_inputs;
use crate::util;

/// Map `body` over axis `in_axes[i]` of each input, stacking result `j` along
/// `out_axes[j]`. Captured values are shared by every instance. The size is
/// inferred from the inputs; it is required when there are none.
pub fn map(
    body: impl Into<Body>,
    inputs: &[Trace],
    in_axes: &[usize],
    out_axes: &[usize],
    size: Option<usize>,
) -> Result<Vec<Trace>, LoweringError> {
    let body = body.into();
    reject_cell_writes(&body)?;
    let signature_ = BodySignature::of_hexpr(&body.expr)?;
    if in_axes.len() != inputs.len() || out_axes.len() != signature_.outputs.len() {
        return Err(signature("map arity"));
    }
    let mut size = size;
    for (input, &axis) in inputs.iter().zip(in_axes) {
        let (n, _) = util::without_axis(input.info(), axis)
            .map_err(|_| signature("map axis out of range"))?;
        if *size.get_or_insert(n) != n {
            return Err(signature("mapped axes must have the same size"));
        }
    }
    let parameters = MapParameters {
        size: size.ok_or_else(|| signature("map without mapped inputs requires a size"))?,
        in_axes: in_axes.to_vec(),
        out_axes: out_axes.to_vec(),
    };
    expect_inputs(
        &body.expr,
        signature_.inputs.iter().cloned().map(Kind::Value),
    )?;
    let outputs = parameters.mapped(&signature_.outputs, out_axes)?;
    parameters.check(
        &signature_,
        Some(&trace_infos(inputs)),
        inputs.len(),
        &outputs,
    )?;
    Ok(record(parameters.to_op()?, &body, inputs, outputs))
}

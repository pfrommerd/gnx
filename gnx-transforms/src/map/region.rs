//! Record `map.map` around an already-traced body. The jit-based [`map`]
//! wrapper is built on this.
//!
//! [`map`]: super::map

use gnx_base::ops::signature;
use gnx_expr::{ApplyRegion, Invocation, Kind, LoweringError, Trace};

use super::MapParameters;
use crate::cf::expect_inputs;
use crate::util;

/// Map `body` over axis `in_axes[i]` of each input, stacking result `j` along
/// `out_axes[j]`. Captured values are shared by every instance. The size is
/// inferred from the inputs; it is required when there are none.
pub fn map(
    body: impl Into<ApplyRegion>,
    inputs: &[Trace],
    in_axes: &[usize],
    out_axes: &[usize],
    size: Option<usize>,
) -> Result<Vec<Trace>, LoweringError> {
    let body = body.into();
    if body.body.view().writes_parameters() {
        return Err(signature(
            "transformed functions cannot write captured cells",
        ));
    }
    let (body_inputs, body_outputs) = util::value_signature(body.body.view())?;
    if in_axes.len() != inputs.len() || out_axes.len() != body_outputs.len() {
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
        body.body.view(),
        (body_inputs.iter().cloned()).map(|info| (Kind::Value, info)),
    )?;
    let outputs = parameters.mapped(&body_outputs, out_axes)?;
    let input_infos: Vec<_> = inputs.iter().map(|trace| trace.info().clone()).collect();
    parameters.check(
        &body_inputs,
        &body_outputs,
        Some(&input_infos),
        inputs.len(),
        &outputs,
    )?;
    Ok(Invocation::apply(
        parameters.to_op(),
        vec![body],
        inputs.iter().cloned().map(Into::into).collect(),
        outputs,
    ))
}

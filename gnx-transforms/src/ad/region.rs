//! Record `ad.jvp` / `ad.vjp` around an already-traced body. The jit-based
//! wrappers ([`jvp`](super::jvp), [`vjp`](super::vjp), [`grad`](super::grad),
//! ...) are built on these.

use gnx_base::ops::{signature, strictly_increasing};
use gnx_expr::{ApplyRegion, Invocation, LoweringError, Trace};

use super::DiffParameters;
use crate::util;

fn differentiate(
    body: ApplyRegion,
    primals: &[Trace],
    extra: &[Trace],
    parameters: DiffParameters,
    forward: bool,
) -> Result<(Vec<Trace>, Vec<Trace>), LoweringError> {
    if body.body.view().writes_parameters() {
        return Err(signature(
            "transformed functions cannot write captured cells",
        ));
    }
    let (body_inputs, body_outputs) = util::value_signature(body.body.view())?;
    strictly_increasing(&parameters.wrt, body_inputs.len())?;
    strictly_increasing(&parameters.of, body_outputs.len())?;
    let operands: Vec<_> = primals.iter().chain(extra).cloned().collect();
    let extra_outputs: Vec<_> = if forward {
        (parameters.of.iter())
            .map(|&i| body_outputs[i].clone())
            .collect()
    } else {
        (parameters.wrt.iter())
            .map(|&i| body_inputs[i].clone())
            .collect()
    };
    let outputs: Vec<_> = (body_outputs.iter().cloned())
        .chain(extra_outputs)
        .collect();
    let operand_infos: Vec<_> = operands.iter().map(|trace| trace.info().clone()).collect();
    parameters.check(
        &body_inputs,
        &body_outputs,
        Some(&operand_infos),
        operands.len(),
        &outputs,
        forward,
    )?;
    let op = if forward {
        parameters.jvp_op()
    } else {
        parameters.vjp_op()
    };
    let mut results = Invocation::apply(
        op,
        vec![body],
        operands.into_iter().map(Into::into).collect(),
        outputs,
    );
    let derivatives = results.split_off(body_outputs.len());
    Ok((results, derivatives))
}

/// Forward mode: the body's results and the tangents of results `of`, given
/// tangents for inputs `wrt` (in order).
pub fn jvp(
    body: impl Into<ApplyRegion>,
    primals: &[Trace],
    tangents: &[Trace],
    wrt: &[usize],
    of: &[usize],
) -> Result<(Vec<Trace>, Vec<Trace>), LoweringError> {
    let parameters = DiffParameters {
        wrt: wrt.to_vec(),
        of: of.to_vec(),
    };
    differentiate(body.into(), primals, tangents, parameters, true)
}

/// Reverse mode: the body's results and the cotangents of inputs `wrt`, given
/// cotangents for results `of` (in order).
pub fn vjp(
    body: impl Into<ApplyRegion>,
    primals: &[Trace],
    cotangents: &[Trace],
    wrt: &[usize],
    of: &[usize],
) -> Result<(Vec<Trace>, Vec<Trace>), LoweringError> {
    let parameters = DiffParameters {
        wrt: wrt.to_vec(),
        of: of.to_vec(),
    };
    differentiate(body.into(), primals, cotangents, parameters, false)
}

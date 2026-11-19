//! Record `ad.jvp` / `ad.vjp` around an already-traced body. The jit-based
//! wrappers ([`jvp`](super::jvp), [`vjp`](super::vjp), [`grad`](super::grad),
//! ...) are built on these.

use gnx_expr::{LoweringError, Trace};

use super::DiffParameters;
use crate::body::{
    Body, BodySignature, record, reject_cell_writes, strictly_increasing, trace_infos,
};

fn differentiate(
    body: Body,
    primals: &[Trace],
    extra: &[Trace],
    parameters: DiffParameters,
    forward: bool,
) -> Result<(Vec<Trace>, Vec<Trace>), LoweringError> {
    reject_cell_writes(&body)?;
    let signature_ = BodySignature::of_hexpr(&body.expr)?;
    strictly_increasing(&parameters.wrt, signature_.inputs.len())?;
    strictly_increasing(&parameters.of, signature_.outputs.len())?;
    let operands: Vec<_> = primals.iter().chain(extra).cloned().collect();
    let extra_outputs: Vec<_> = if forward {
        parameters
            .of
            .iter()
            .map(|&i| signature_.outputs[i].clone())
            .collect()
    } else {
        parameters
            .wrt
            .iter()
            .map(|&i| signature_.inputs[i].clone())
            .collect()
    };
    let outputs: Vec<_> = signature_
        .outputs
        .iter()
        .cloned()
        .chain(extra_outputs)
        .collect();
    parameters.check(
        &signature_,
        Some(&trace_infos(&operands)),
        operands.len(),
        &outputs,
        forward,
    )?;
    let op = if forward {
        parameters.jvp_op()
    } else {
        parameters.vjp_op()
    };
    let mut results = record(op, &body, &operands, outputs);
    let derivatives = results.split_off(signature_.outputs.len());
    Ok((results, derivatives))
}

/// Forward mode: the body's results and the tangents of results `of`, given
/// tangents for inputs `wrt` (in order).
pub fn jvp(
    body: impl Into<Body>,
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
    body: impl Into<Body>,
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

//! Vectorization (`map.map`) by per-operation batching rules.

use std::collections::HashMap;

use gnx_expr::pass::Builder;
use gnx_expr::{Eqn, EqnSource, Expr, SubExprBuilder, Var};

use super::MapParameters;
use crate::util::{broadcast_axis, copy_eqn, input_infos, move_axis, region_body, with_axis};
use crate::{TransformError, TransformResult};

/// A value in the batched program and the axis holding its instances, if any.
/// An unbatched value is shared by every instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Batched {
    pub var: Var,
    pub axis: Option<usize>,
}

impl Batched {
    pub fn unbatched(var: Var) -> Self {
        Self { var, axis: None }
    }
    pub fn at(var: Var, axis: usize) -> Self {
        Self {
            var,
            axis: Some(axis),
        }
    }
}

/// Batching rule for an operation, registered with
/// `register_op_interface!(dyn BatchRule: MyOperation)`.
pub trait BatchRule: Sync {
    /// Emit the batched equivalent of `eqn` into `b`. `inputs` and the region
    /// `captures` are bound in `b`; at least one of them is batched. Return one
    /// value per result of `eqn`, whose metadata has `size` inserted at its axis.
    fn batch(
        &self,
        b: &mut Builder,
        size: usize,
        eqn: Eqn<'_>,
        inputs: &[Batched],
        captures: &[Vec<Batched>],
    ) -> TransformResult<Vec<Batched>>;
}

/// Move a value's batch axis to `axis`, broadcasting it if unbatched.
pub fn materialize(
    b: &mut Builder,
    value: Batched,
    size: usize,
    axis: usize,
) -> TransformResult<Var> {
    match value.axis {
        Some(from) => move_axis(b, value.var, from, axis),
        None => broadcast_axis(b, value.var, axis, size),
    }
}

fn vars(values: &[Batched]) -> Vec<Var> {
    values.iter().map(|value| value.var).collect()
}

/// Emit a batched copy of `expr` with the given parameter bindings.
pub fn batch_expr(
    b: &mut Builder,
    size: usize,
    expr: Expr<'_>,
    closure: &[Batched],
    inputs: &[Batched],
) -> TransformResult<Vec<Batched>> {
    let mut env: HashMap<Var, Batched> = expr
        .closure_inputs()
        .chain(expr.inputs())
        .map(|param| param.var())
        .zip(closure.iter().chain(inputs).copied())
        .collect();
    for (index, eqn) in expr.eqns().enumerate() {
        let operands: Vec<_> = eqn.inputs().iter().map(|var| env[var]).collect();
        let captures: Vec<Vec<_>> = eqn
            .subexprs()
            .map(|sub| sub.provided_vars().iter().map(|var| env[var]).collect())
            .collect();
        let results = (|| {
            let results = if operands
                .iter()
                .chain(captures.iter().flatten())
                .all(|value| value.axis.is_none())
            {
                let captures: Vec<_> = captures.iter().map(|values| vars(values)).collect();
                copy_eqn(b, eqn, &vars(&operands), &captures)
                    .into_iter()
                    .map(Batched::unbatched)
                    .collect()
            } else {
                eqn.op()
                    .interface::<dyn BatchRule>()?
                    .batch(b, size, eqn, &operands, &captures)?
            };
            if results.len() != eqn.outputs().len() {
                return Err(TransformError::invalid(
                    "batching rule returned the wrong arity",
                ));
            }
            for (value, param) in results.iter().zip(eqn.outputs()) {
                let info = param.info_ref().to_owned();
                let expected = match value.axis {
                    Some(axis) => with_axis(&info, axis, size)?,
                    None => info,
                };
                if *b.info(value.var) != expected {
                    return Err(TransformError::invalid(format!(
                        "batching rule produced {}, expected {expected}",
                        b.info(value.var)
                    )));
                }
            }
            Ok(results)
        })()
        .map_err(|error: TransformError| error.at(index, eqn))?;
        for (param, value) in eqn.outputs().zip(results) {
            env.insert(param.var(), value);
        }
    }
    Ok(expr.outputs().iter().map(|var| env[var]).collect())
}

/// Batch a closed region of a control-flow operation. Its explicit input `i`
/// is batched along axis 0 if `inputs[i]`; its captures are parent values.
/// Batched results are moved to axis 0. When `outputs` is given, exactly those
/// results are batched (unbatched ones are broadcast); otherwise the batched
/// results are whatever the body produces. Returns the region and the flags.
pub fn batch_region(
    b: &Builder,
    size: usize,
    body: Expr<'_>,
    captures: &[Batched],
    inputs: &[bool],
    outputs: Option<&[bool]>,
) -> TransformResult<(SubExprBuilder, Vec<bool>)> {
    if inputs.len() != body.inputs().len() || captures.len() != body.closure_inputs().len() {
        return Err(TransformError::invalid("region binding arity"));
    }
    let infos = input_infos(body)
        .into_iter()
        .zip(inputs)
        .map(|(info, &batched)| {
            if batched {
                with_axis(&info, 0, size)
            } else {
                Ok(info)
            }
        })
        .collect::<TransformResult<Vec<_>>>()?;
    let mut flags = Vec::new();
    let region = b.region(&vars(captures), &infos, |rb, closure, params| {
        let closure: Vec<_> = closure
            .iter()
            .zip(captures)
            .map(|(&var, capture)| Batched {
                var,
                axis: capture.axis,
            })
            .collect();
        let params: Vec<_> = params
            .iter()
            .zip(inputs)
            .map(|(&var, &batched)| Batched {
                var,
                axis: batched.then_some(0),
            })
            .collect();
        let results = batch_expr(rb, size, body, &closure, &params)?;
        flags = match outputs {
            Some(flags) => flags.to_vec(),
            None => results.iter().map(|value| value.axis.is_some()).collect(),
        };
        results
            .into_iter()
            .zip(&flags)
            .map(|(value, &batched)| {
                if batched {
                    materialize(rb, value, size, 0)
                } else if value.axis.is_some() {
                    Err(TransformError::invalid(
                        "region result is unexpectedly batched",
                    ))
                } else {
                    Ok(value.var)
                }
            })
            .collect()
    })?;
    Ok((region, flags))
}

/// Expand `ad.map` by batching its body.
pub(super) fn expand_map(
    b: &mut Builder,
    eqn: &EqnSource,
    parameters: &MapParameters,
) -> TransformResult<Vec<Var>> {
    let [sub] = eqn.subexprs() else {
        return Err(TransformError::invalid("map takes one body"));
    };
    let body = region_body(sub)?;
    let closure: Vec<_> = sub
        .provided_vars()
        .iter()
        .copied()
        .map(Batched::unbatched)
        .collect();
    let inputs: Vec<_> = eqn
        .inputs()
        .iter()
        .zip(&parameters.in_axes)
        .map(|(&var, &axis)| Batched::at(var, axis))
        .collect();
    let results = batch_expr(b, parameters.size, body.view(), &closure, &inputs)?;
    results
        .into_iter()
        .zip(&parameters.out_axes)
        .map(|(value, &axis)| materialize(b, value, parameters.size, axis))
        .collect()
}

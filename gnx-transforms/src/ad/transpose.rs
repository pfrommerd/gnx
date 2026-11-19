//! Reverse-mode differentiation (`ad.vjp`) by transposing linearized programs.

use std::collections::{HashMap, HashSet};

use gnx_expr::pass::Builder;
use gnx_expr::{Eqn, Expr, ValueInfo, Var};

use super::DiffParameters;
use super::jvp::{Dual, Tangent, accumulate, instantiate, jvp_expr};
use crate::util::copy_eqn;
use crate::{TransformError, TransformResult};

/// An operand of a linear equation: a residual (constant) bound in the output
/// builder, or a linear value whose cotangent the rule must produce.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Operand {
    Residual(Var),
    Linear(ValueInfo),
}

impl Operand {
    pub fn is_linear(&self) -> bool {
        matches!(self, Operand::Linear(_))
    }
}

/// Transposition rule for an operation that is linear in its `Linear`
/// operands, registered with `register_op_interface!(dyn TransposeRule: MyOperation)`.
pub trait TransposeRule: Sync {
    /// Given one cotangent per result of `eqn` (not all zero), return one
    /// cotangent per operand (`Tangent::Zero` for residual operands). Region
    /// `captures` are residuals bound in `b`.
    fn transpose(
        &self,
        b: &mut Builder,
        eqn: &Eqn,
        operands: &[Operand],
        captures: &[Vec<Var>],
        cotangents: &[Tangent],
    ) -> TransformResult<Vec<Tangent>>;
}

/// Evaluate the residual part of `expr` forward and transpose its linear part.
///
/// `bindings` covers the closure followed by the inputs: `Some(var)` binds a
/// residual parameter, `None` marks a linear one. `cotangents` has one entry
/// per result. Returns each result's value if it is not linear, and one
/// cotangent per binding (zero for residuals).
pub fn transpose_expr(
    b: &mut Builder,
    expr: &Expr,
    bindings: &[Option<Var>],
    cotangents: &[Tangent],
) -> TransformResult<(Vec<Option<Var>>, Vec<Tangent>)> {
    let params: Vec<_> = expr.closure().iter().chain(expr.inputs()).collect();
    if params.len() != bindings.len() || cotangents.len() != expr.outputs().len() {
        return Err(TransformError::invalid("transpose binding arity"));
    }
    let mut env = HashMap::new();
    let mut linear = HashSet::new();
    for ((var, _), binding) in params.iter().zip(bindings) {
        match binding {
            Some(value) => {
                env.insert(*var, *value);
            }
            None => {
                linear.insert(*var);
            }
        }
    }
    let mut linear_eqns = Vec::new();
    for (index, eqn) in expr.eqns().iter().enumerate() {
        let is_linear = eqn
            .inputs()
            .iter()
            .chain(eqn.subexprs().iter().flat_map(|sub| sub.closure()))
            .any(|var| linear.contains(var));
        if is_linear {
            linear.extend(eqn.outputs().iter().map(|(var, _)| *var));
            linear_eqns.push((index, eqn));
        } else {
            let inputs: Vec<_> = eqn.inputs().iter().map(|var| env[var]).collect();
            let captures: Vec<Vec<_>> = eqn
                .subexprs()
                .iter()
                .map(|sub| sub.closure().iter().map(|var| env[var]).collect())
                .collect();
            let outputs = copy_eqn(b, eqn, &inputs, &captures);
            for ((var, _), output) in eqn.outputs().iter().zip(outputs) {
                env.insert(*var, output);
            }
        }
    }
    let mut cts: HashMap<Var, Tangent> = HashMap::new();
    for (var, &ct) in expr.outputs().iter().zip(cotangents) {
        if linear.contains(var) {
            let previous = cts.remove(var).unwrap_or(Tangent::Zero);
            cts.insert(*var, accumulate(b, previous, ct));
        }
    }
    for (index, eqn) in linear_eqns.into_iter().rev() {
        let output_cts: Vec<_> = eqn
            .outputs()
            .iter()
            .map(|(var, _)| cts.remove(var).unwrap_or(Tangent::Zero))
            .collect();
        if output_cts.iter().all(|ct| ct.is_zero()) {
            continue;
        }
        let input_cts = (|| {
            let operands: Vec<_> = eqn
                .inputs()
                .iter()
                .map(|var| {
                    if linear.contains(var) {
                        Operand::Linear(expr.value_info(var).expect("defined operand").clone())
                    } else {
                        Operand::Residual(env[var])
                    }
                })
                .collect();
            let captures = eqn
                .subexprs()
                .iter()
                .map(|sub| {
                    sub.closure()
                        .iter()
                        .map(|var| {
                            env.get(var).copied().ok_or_else(|| {
                                TransformError::invalid("a region captures a linear value")
                            })
                        })
                        .collect::<TransformResult<Vec<_>>>()
                })
                .collect::<TransformResult<Vec<_>>>()?;
            let input_cts = eqn.op().interface::<dyn TransposeRule>()?.transpose(
                b,
                eqn,
                &operands,
                &captures,
                &output_cts,
            )?;
            if input_cts.len() != eqn.inputs().len() {
                return Err(TransformError::invalid(
                    "transpose rule returned the wrong arity",
                ));
            }
            for ((operand, ct), var) in operands.iter().zip(&input_cts).zip(eqn.inputs()) {
                match (operand, ct) {
                    (_, Tangent::Zero) => {}
                    (Operand::Linear(info), Tangent::Var(ct)) if b.info(*ct) == info => {}
                    _ => {
                        return Err(TransformError::invalid(format!(
                            "transpose rule produced an invalid cotangent for {var:?}"
                        )));
                    }
                }
            }
            Ok(input_cts)
        })()
        .map_err(|error: TransformError| error.at(index, eqn))?;
        for (var, ct) in eqn.inputs().iter().zip(input_cts) {
            if linear.contains(var) && !ct.is_zero() {
                let previous = cts.remove(var).unwrap_or(Tangent::Zero);
                cts.insert(*var, accumulate(b, previous, ct));
            }
        }
    }
    let values = expr
        .outputs()
        .iter()
        .map(|var| env.get(var).copied())
        .collect();
    let param_cts = params
        .iter()
        .zip(bindings)
        .map(|((var, _), binding)| match binding {
            Some(_) => Tangent::Zero,
            None => cts.remove(var).unwrap_or(Tangent::Zero),
        })
        .collect();
    Ok((values, param_cts))
}

/// Expand `ad.vjp`: linearize the body with forward mode, evaluate the primal
/// part, and transpose the tangent part.
pub(super) fn expand_vjp(b: &mut Builder, eqn: &Eqn) -> TransformResult<Vec<Var>> {
    let parameters = DiffParameters::from_op(eqn.op())?;
    let [sub] = eqn.subexprs() else {
        return Err(TransformError::invalid("vjp takes one body"));
    };
    let body = sub.expr();
    let (primals, cotangents) = eqn.inputs().split_at(body.inputs().len());

    let mut joint = Builder::new();
    let closure: Vec<_> = body
        .closure()
        .iter()
        .map(|(_, info)| Dual::constant(joint.closure(info.clone())))
        .collect();
    let mut inputs: Vec<_> = body
        .inputs()
        .iter()
        .map(|(_, info)| Dual::constant(joint.input(info.clone())))
        .collect();
    for &index in &parameters.wrt {
        let info = body.inputs()[index].1.clone();
        inputs[index].tangent = Tangent::Var(joint.input(info));
    }
    let results = jvp_expr(&mut joint, body, &closure, &inputs)?;
    let mut outputs: Vec<_> = results.iter().map(|dual| dual.primal).collect();
    for &index in &parameters.of {
        let info = joint.info(results[index].primal).clone();
        outputs.push(instantiate(&mut joint, results[index].tangent, &info));
    }
    let joint = joint.finish(outputs);

    let bindings: Vec<_> = sub
        .closure()
        .iter()
        .chain(primals)
        .map(|&var| Some(var))
        .chain(parameters.wrt.iter().map(|_| None))
        .collect();
    let output_cts: Vec<_> = std::iter::repeat_n(Tangent::Zero, results.len())
        .chain(cotangents.iter().map(|&var| Tangent::Var(var)))
        .collect();
    let (values, cts) = transpose_expr(b, &joint, &bindings, &output_cts)?;
    let mut outputs = values[..results.len()]
        .iter()
        .map(|value| {
            value.ok_or_else(|| {
                TransformError::invalid(
                    "a primal result depends on a tangent; the body mixes primal and tangent values",
                )
            })
        })
        .collect::<TransformResult<Vec<_>>>()?;
    let first_tangent = body.closure().len() + body.inputs().len();
    for (k, &index) in parameters.wrt.iter().enumerate() {
        let info = body.inputs()[index].1.clone();
        outputs.push(instantiate(b, cts[first_tangent + k], &info));
    }
    Ok(outputs)
}

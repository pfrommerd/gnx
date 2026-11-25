//! Forward-mode differentiation (`ad.jvp`) and linearization.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use gnx_expr::pass::Builder;
use gnx_expr::{Eqn, EqnSource, Expr, ExprBuilder, RootExpr, Var, VarInfo};

use super::DiffParameters;
use crate::util::{add, copy_eqn, region_body, zeros};
use crate::{TransformError, TransformResult};

/// A tangent, or a symbolic zero that is never materialized unless needed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tangent {
    Zero,
    Var(Var),
}

impl Tangent {
    pub fn is_zero(self) -> bool {
        self == Tangent::Zero
    }
}

/// A primal value and its tangent, both bound in the output builder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dual {
    pub primal: Var,
    pub tangent: Tangent,
}

impl Dual {
    pub fn constant(primal: Var) -> Self {
        Self {
            primal,
            tangent: Tangent::Zero,
        }
    }
}

/// Forward-mode rule for an operation, registered with
/// `register_op_interface!(dyn JvpRule: MyOperation)`.
///
/// To support reverse mode, a rule must keep primal and tangent computations
/// in separate equations: tangent equations consume tangents and primal
/// values (residuals) and are linear in the tangents, while primal equations
/// never consume tangents. Each equation of the tangent part then needs a
/// [`TransposeRule`](super::TransposeRule).
pub trait JvpRule: Sync {
    /// `inputs` and region `captures` are bound in `b`; at least one tangent is
    /// nonzero. Return one dual per result of `eqn`.
    fn jvp(
        &self,
        b: &mut Builder,
        eqn: Eqn<'_>,
        inputs: &[Dual],
        captures: &[Vec<Dual>],
    ) -> TransformResult<Vec<Dual>>;
}

/// The tangent as a variable, materializing a zero with `info`.
pub fn instantiate(b: &mut Builder, tangent: Tangent, info: &VarInfo) -> Var {
    match tangent {
        Tangent::Var(var) => var,
        Tangent::Zero => zeros(b, info),
    }
}

/// Sum two tangents, skipping symbolic zeros.
pub fn accumulate(b: &mut Builder, lhs: Tangent, rhs: Tangent) -> Tangent {
    match (lhs, rhs) {
        (Tangent::Zero, other) | (other, Tangent::Zero) => other,
        (Tangent::Var(lhs), Tangent::Var(rhs)) => Tangent::Var(add(b, lhs, rhs)),
    }
}

/// Emit primal and tangent equations for `expr` with the given bindings.
pub fn jvp_expr(
    b: &mut Builder,
    expr: Expr<'_>,
    closure: &[Dual],
    inputs: &[Dual],
) -> TransformResult<Vec<Dual>> {
    let mut env: HashMap<Var, Dual> = expr
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
            if operands
                .iter()
                .chain(captures.iter().flatten())
                .all(|dual| dual.tangent.is_zero())
            {
                let primals = |duals: &[Dual]| duals.iter().map(|dual| dual.primal).collect();
                let captures: Vec<Vec<_>> = captures.iter().map(|duals| primals(duals)).collect();
                return Ok(copy_eqn(b, eqn, &primals(&operands), &captures)
                    .into_iter()
                    .map(Dual::constant)
                    .collect());
            }
            let results = eqn
                .op()
                .interface::<dyn JvpRule>()?
                .jvp(b, eqn, &operands, &captures)?;
            if results.len() != eqn.outputs().len() {
                return Err(TransformError::invalid("jvp rule returned the wrong arity"));
            }
            for (dual, param) in results.iter().zip(eqn.outputs()) {
                let info = param.info_ref();
                let tangent = match dual.tangent {
                    Tangent::Var(var) => Some(b.info(var)),
                    Tangent::Zero => None,
                };
                if info != *b.info(dual.primal) || tangent.is_some_and(|actual| info != *actual) {
                    return Err(TransformError::invalid("jvp rule changed result metadata"));
                }
            }
            Ok(results)
        })()
        .map_err(|error: TransformError| error.at(index, eqn))?;
        for (param, dual) in eqn.outputs().zip(results) {
            env.insert(param.var(), dual);
        }
    }
    Ok(expr.outputs().iter().map(|var| env[var]).collect())
}

/// A body split into a primal program and a tangent program that is linear in
/// its tangent inputs.
pub struct Linearized {
    /// Same closure and inputs as the body; returns the body's results followed
    /// by the residuals.
    pub primal: Arc<RootExpr>,
    /// No closure; takes the residuals, then the tangents of the flagged
    /// captures, then the tangents of the flagged inputs. Returns the tangents
    /// of the flagged results (materialized).
    pub tangent: Arc<RootExpr>,
    pub residuals: Vec<VarInfo>,
    pub outputs: Vec<bool>,
}

/// Linearize `body` given which captures and inputs have nonzero tangents.
/// If `force` is given, exactly those results get (possibly zero) tangents;
/// a nonzero tangent for another result is an error.
pub fn linearize(
    body: Expr<'_>,
    closure: &[bool],
    inputs: &[bool],
    force: Option<&[bool]>,
) -> TransformResult<Linearized> {
    let mut joint = Builder::new();
    let closure_infos: Vec<VarInfo> = body
        .closure_inputs()
        .map(|param| param.info_ref().to_owned())
        .collect();
    let input_infos: Vec<VarInfo> = body
        .inputs()
        .map(|param| param.info_ref().to_owned())
        .collect();
    let primal_closure: Vec<_> = closure_infos
        .iter()
        .map(|info| joint.closure(info.clone()))
        .collect();
    let primal_inputs: Vec<_> = input_infos
        .iter()
        .map(|info| joint.input(info.clone()))
        .collect();
    let mut tangent_inputs = Vec::new();
    let mut duals = |params: &[VarInfo], primals: &[Var], flags: &[bool], joint: &mut Builder| {
        params
            .iter()
            .zip(primals)
            .zip(flags)
            .map(|((info, &primal), &flag)| Dual {
                primal,
                tangent: if flag {
                    let var = joint.input(info.clone());
                    tangent_inputs.push(var);
                    Tangent::Var(var)
                } else {
                    Tangent::Zero
                },
            })
            .collect::<Vec<_>>()
    };
    if closure.len() != closure_infos.len() || inputs.len() != input_infos.len() {
        return Err(TransformError::invalid("linearize binding arity"));
    }
    let closure_duals = duals(&closure_infos, &primal_closure, closure, &mut joint);
    let input_duals = duals(&input_infos, &primal_inputs, inputs, &mut joint);
    let results = jvp_expr(&mut joint, body, &closure_duals, &input_duals)?;
    let flags: Vec<_> = match force {
        Some(force) => {
            if results
                .iter()
                .zip(force)
                .any(|(dual, &forced)| !forced && !dual.tangent.is_zero())
            {
                return Err(TransformError::invalid("unexpected nonzero tangent"));
            }
            force.to_vec()
        }
        None => results.iter().map(|dual| !dual.tangent.is_zero()).collect(),
    };
    let primal_outputs: Vec<_> = results.iter().map(|dual| dual.primal).collect();
    let mut tangent_outputs = Vec::new();
    for (dual, &flag) in results.iter().zip(&flags) {
        if flag {
            let info = joint.info(dual.primal).clone();
            tangent_outputs.push(instantiate(&mut joint, dual.tangent, &info));
        }
    }
    let joint_outputs = primal_outputs
        .iter()
        .chain(&tangent_outputs)
        .copied()
        .collect();
    let joint = joint.finish(joint_outputs).build()?;
    let (primal, tangent, residuals) =
        split(joint.view(), tangent_inputs.len(), primal_outputs.len())?;
    Ok(Linearized {
        primal,
        tangent,
        residuals,
        outputs: flags,
    })
}

/// Split a joint program whose last `tangents` inputs are tangents and whose
/// first `primals` results are primal into a primal program (returning the
/// primal results and residuals) and a linear tangent program (taking the
/// residuals and tangents).
pub fn split(
    joint: Expr<'_>,
    tangents: usize,
    primals: usize,
) -> TransformResult<(Arc<RootExpr>, Arc<RootExpr>, Vec<VarInfo>)> {
    let params = |params: &mut dyn Iterator<Item = gnx_expr::Param<'_>>| -> Vec<(Var, VarInfo)> {
        params
            .map(|param| (param.var(), param.info_ref().to_owned()))
            .collect()
    };
    let inputs = params(&mut joint.inputs());
    let split_at = inputs.len() - tangents;
    let (primal_inputs, tangent_inputs) = inputs.split_at(split_at);
    let mut linear: HashSet<Var> = tangent_inputs.iter().map(|(var, _)| *var).collect();
    let mut primal_eqns = Vec::new();
    let mut linear_eqns = Vec::new();
    let mut residuals = Vec::new();
    let mut seen = HashSet::new();
    let mut residual = |var: Var, linear: &HashSet<Var>, residuals: &mut Vec<Var>| {
        if !linear.contains(&var) && seen.insert(var) {
            residuals.push(var);
        }
    };
    for eqn in joint.eqns() {
        let operands: Vec<Var> = eqn
            .inputs()
            .iter()
            .chain(eqn.subexprs().flat_map(|sub| sub.provided_vars()))
            .copied()
            .collect();
        if operands.iter().any(|var| linear.contains(var)) {
            for &var in &operands {
                residual(var, &linear, &mut residuals);
            }
            linear.extend(eqn.outputs().map(|param| param.var()));
            linear_eqns.push(eqn);
        } else {
            primal_eqns.push(eqn.to_source());
        }
    }
    let (primal_results, tangent_results) = joint.outputs().split_at(primals);
    if primal_results.iter().any(|var| linear.contains(var)) {
        return Err(TransformError::invalid(
            "an operation mixes primal and tangent values (such as a while_loop) and cannot be linearized",
        ));
    }
    for &var in tangent_results {
        residual(var, &linear, &mut residuals);
    }
    let infos: Vec<_> = residuals
        .iter()
        .map(|&var| joint.value_info(var).expect("defined residual").to_owned())
        .collect();
    let primal = ExprBuilder::from_parts(
        joint.closure_symbols().to_vec(),
        params(&mut joint.closure_inputs()),
        primal_inputs.to_vec(),
        primal_eqns,
        primal_results.iter().chain(&residuals).copied().collect(),
    )
    .build()?;
    let mut tb = Builder::new();
    let mut map: HashMap<Var, Var> = residuals
        .iter()
        .zip(&infos)
        .map(|(&var, info)| (var, tb.input(info.clone())))
        .collect();
    for (var, info) in tangent_inputs {
        map.insert(*var, tb.input(info.clone()));
    }
    for eqn in linear_eqns {
        let inputs: Vec<_> = eqn.inputs().iter().map(|var| map[var]).collect();
        let captures: Vec<Vec<_>> = eqn
            .subexprs()
            .map(|sub| sub.provided_vars().iter().map(|var| map[var]).collect())
            .collect();
        let outputs = copy_eqn(&mut tb, eqn, &inputs, &captures);
        for (param, output) in eqn.outputs().zip(outputs) {
            map.insert(param.var(), output);
        }
    }
    let tangent = tb
        .finish(tangent_results.iter().map(|var| map[var]).collect())
        .build()?;
    Ok((primal, tangent, infos))
}

/// Expand `ad.jvp` by running the forward-mode interpreter on its body.
pub(super) fn expand_jvp(
    b: &mut Builder,
    eqn: &EqnSource,
    parameters: &DiffParameters,
) -> TransformResult<Vec<Var>> {
    let [sub] = eqn.subexprs() else {
        return Err(TransformError::invalid("jvp takes one body"));
    };
    let body = region_body(sub)?;
    let body = body.view();
    let (primals, tangents) = eqn.inputs().split_at(body.inputs().len());
    let closure: Vec<_> = (sub.provided_vars().iter().copied())
        .map(Dual::constant)
        .collect();
    let mut inputs: Vec<_> = primals.iter().copied().map(Dual::constant).collect();
    for (&index, &tangent) in parameters.wrt.iter().zip(tangents) {
        inputs[index].tangent = Tangent::Var(tangent);
    }
    let results = jvp_expr(b, body, &closure, &inputs)?;
    let mut outputs: Vec<_> = results.iter().map(|dual| dual.primal).collect();
    for &index in &parameters.of {
        let info = b.info(results[index].primal).clone();
        outputs.push(instantiate(b, results[index].tangent, &info));
    }
    Ok(outputs)
}

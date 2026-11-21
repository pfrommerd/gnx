use std::collections::HashSet;
use std::sync::Arc;

use super::{Pass, PassError};
use crate::{Eqn, Expr, SubExpr, Var};

/// Dead-code elimination. Expr is pure, so an equation none of whose results
/// are used is removed. Regions are pruned recursively and their unused
/// captures dropped. Region-operation results are never removed individually,
/// since that would change the operation's signature.
pub struct Dce;

impl Pass for Dce {
    fn name(&self) -> &'static str {
        "dce"
    }

    fn run(&self, expr: Expr) -> Result<Expr, PassError> {
        Ok(dce(&expr))
    }
}

pub fn dce(expr: &Expr) -> Expr {
    let mut live: HashSet<Var> = expr.outputs.iter().copied().collect();
    let mut eqns = Vec::new();
    for eqn in expr.eqns.iter().rev() {
        if !eqn.outputs.iter().any(|(var, _)| live.contains(var)) {
            continue;
        }
        let eqn = Eqn::new_with_subexprs(
            eqn.op.clone(),
            eqn.subexprs.iter().map(prune_region).collect(),
            eqn.inputs.clone(),
            eqn.outputs.clone(),
        );
        live.extend(&eqn.inputs);
        live.extend(eqn.subexprs.iter().flat_map(|sub| &sub.captures));
        eqns.push(eqn);
    }
    eqns.reverse();
    Expr {
        symbol_closure: expr.symbol_closure.clone(),
        symbol_inputs: expr.symbol_inputs.clone(),
        closure: expr.closure.clone(),
        inputs: expr.inputs.clone(),
        eqns,
        outputs: expr.outputs.clone(),
        name_hint: expr.name_hint.clone(),
    }
}

fn prune_region(sub: &SubExpr) -> SubExpr {
    let mut body = dce(&sub.expr);
    let used: HashSet<Var> = body
        .eqns
        .iter()
        .flat_map(|eqn| {
            eqn.inputs
                .iter()
                .chain(eqn.subexprs.iter().flat_map(|sub| &sub.captures))
        })
        .chain(&body.outputs)
        .copied()
        .collect();
    let (closure, captures) = std::mem::take(&mut body.closure)
        .into_iter()
        .zip(&sub.captures)
        .filter(|((var, _), _)| used.contains(var))
        .unzip();
    body.closure = closure;
    SubExpr::new(Arc::new(body), captures).with_symbol_captures(sub.symbol_captures.clone())
}

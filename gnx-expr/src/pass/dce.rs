use std::collections::HashSet;
use std::sync::Arc;

use super::{Pass, PassError};
use crate::{EqnSource, Expr, ExprBuilder, RootExpr, SubExpr, SubExprBuilder, Var};

/// Dead-code elimination. Expr is pure, so an equation none of whose results
/// are used is removed. Regions are pruned recursively and their unused
/// captures dropped. Region-operation results are never removed individually,
/// since that would change the operation's signature.
pub struct Dce;

impl Pass for Dce {
    fn name(&self) -> &'static str {
        "dce"
    }

    fn run(&self, expr: Arc<RootExpr>) -> Result<Arc<RootExpr>, PassError> {
        dce(expr.view())
            .build()
            .map_err(|error| PassError::new(self.name(), error))
    }
}

pub fn dce(expr: Expr<'_>) -> ExprBuilder {
    let mut live: HashSet<Var> = expr.outputs().iter().copied().collect();
    let mut eqns = Vec::new();
    for eqn in expr.eqns().rev() {
        if !eqn.outputs().any(|param| live.contains(&param.var())) {
            continue;
        }
        let eqn = EqnSource::new(
            eqn.op().to_owned(),
            eqn.subexprs().map(prune_region).collect(),
            eqn.inputs().to_vec(),
            (eqn.outputs())
                .map(|param| (param.var(), param.info_ref().to_owned()))
                .collect(),
        );
        live.extend(eqn.inputs());
        live.extend(eqn.subexprs().iter().flat_map(|sub| sub.provided_vars()));
        eqns.push(eqn);
    }
    eqns.reverse();
    let params = |params: &mut dyn Iterator<Item = crate::Param<'_>>| {
        params
            .map(|param| (param.var(), param.info_ref().to_owned()))
            .collect()
    };
    let mut pruned = ExprBuilder::from_parts(
        expr.closure_symbols().to_vec(),
        params(&mut expr.closure_inputs()),
        params(&mut expr.inputs()),
        eqns,
        expr.outputs().to_vec(),
    );
    pruned.set_name_hint(expr.name_hint().map(str::to_owned));
    pruned
}

fn prune_region(sub: SubExpr<'_>) -> SubExprBuilder {
    let mut body = dce(sub.expr());
    let used: HashSet<Var> = (body.eqns.iter())
        .flat_map(|eqn| {
            (eqn.inputs().iter()).chain(eqn.subexprs().iter().flat_map(|sub| sub.provided_vars()))
        })
        .chain(&body.outputs)
        .copied()
        .collect();
    let (closure, captures): (Vec<_>, Vec<_>) = std::mem::take(&mut body.closure_inputs)
        .into_iter()
        .zip(sub.provided_vars().iter().copied())
        .filter(|((var, _), _)| used.contains(var))
        .unzip();
    body.closure_inputs = closure;
    SubExprBuilder::new(body)
        .with_vars(captures)
        .with_symbols(sub.provided_symbols().iter().copied())
}

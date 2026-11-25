use std::collections::HashMap;

use super::Builder;
use gnx_expr::{EqnSource, Expr, ExprBuilder, SubExprBuilder, Var};

/// Copy `expr` into a fresh namespace, letting `hook` replace equations.
///
/// Regions are rewritten first (bottom-up). The hook then sees each equation
/// already translated into the new builder: its inputs and region captures
/// are builder variables, its regions are rewritten, and its results are fresh.
/// Returning `Ok(None)` keeps the equation; `Ok(Some(results))` replaces it
/// with whatever the hook emitted, one result per original output.
pub fn rewrite<E, F>(expr: Expr<'_>, hook: &mut F) -> Result<ExprBuilder, E>
where
    F: FnMut(&mut Builder, &EqnSource) -> Result<Option<Vec<Var>>, E>,
{
    let mut builder = Builder::new();
    builder.inherit_symbols(expr);
    builder.set_name_hint(expr.name_hint().map(str::to_owned));
    let mut map = HashMap::new();
    for param in expr.closure_inputs() {
        map.insert(param.var(), builder.closure(param.info_ref().to_owned()));
    }
    for param in expr.inputs() {
        map.insert(param.var(), builder.input(param.info_ref().to_owned()));
    }
    let lookup = |map: &HashMap<Var, Var>, vars: &[Var]| -> Vec<Var> {
        vars.iter().map(|var| map[var]).collect()
    };
    for eqn in expr.eqns() {
        let subexprs = eqn
            .subexprs()
            .map(|sub| {
                Ok(SubExprBuilder::new(rewrite(sub.expr(), hook)?)
                    .with_vars(lookup(&map, sub.provided_vars()))
                    .with_symbols(sub.provided_symbols().iter().copied()))
            })
            .collect::<Result<Vec<_>, E>>()?;
        let outputs = eqn
            .outputs()
            .map(|param| {
                let info = param.info_ref().to_owned();
                if param.var().is_hole() {
                    return (Var::hole(), info);
                }
                (builder.fresh(info.clone()), info)
            })
            .collect();
        let translated = EqnSource::new(
            eqn.op().to_owned(),
            subexprs,
            lookup(&map, eqn.inputs()),
            outputs,
        );
        let results = match hook(&mut builder, &translated)? {
            Some(results) => {
                assert_eq!(
                    results.len(),
                    eqn.outputs().len(),
                    "rewrite of {} returned the wrong number of results",
                    eqn.op()
                );
                for (expected, &var) in eqn.outputs().zip(&results) {
                    debug_assert!(
                        var.is_hole() || expected.info_ref() == *builder.info(var),
                        "rewrite of {} changed result metadata",
                        eqn.op()
                    );
                }
                results
            }
            None => builder.push(translated),
        };
        for (param, result) in eqn.outputs().zip(results) {
            if !param.var().is_hole() {
                map.insert(param.var(), result);
            }
        }
    }
    Ok(builder.finish(lookup(&map, expr.outputs())))
}

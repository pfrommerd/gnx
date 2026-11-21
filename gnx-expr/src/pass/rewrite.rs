use std::collections::HashMap;
use std::sync::Arc;

use super::Builder;
use crate::{Eqn, Expr, SubExpr, Var};

/// Copy `expr` into a fresh namespace, letting `hook` replace equations.
///
/// Regions are rewritten first (bottom-up). The hook then sees each equation
/// already translated into the new builder: its inputs and region captures
/// are builder variables, its regions are rewritten, and its results are fresh.
/// Returning `Ok(None)` keeps the equation; `Ok(Some(results))` replaces it
/// with whatever the hook emitted, one result per original output.
pub fn rewrite<E, F>(expr: &Expr, hook: &mut F) -> Result<Expr, E>
where
    F: FnMut(&mut Builder, &Eqn) -> Result<Option<Vec<Var>>, E>,
{
    let mut builder = Builder::new();
    builder.inherit_symbols(expr);
    builder.set_name_hint(expr.name_hint.clone());
    let mut map = HashMap::new();
    for (var, info) in &expr.closure {
        map.insert(*var, builder.closure_like(*var, info.clone()));
    }
    for (var, info) in &expr.inputs {
        map.insert(*var, builder.input_like(*var, info.clone()));
    }
    let lookup = |map: &HashMap<Var, Var>, vars: &[Var]| -> Vec<Var> {
        vars.iter().map(|var| map[var]).collect()
    };
    for eqn in &expr.eqns {
        let subexprs = eqn
            .subexprs
            .iter()
            .map(|sub| {
                Ok(SubExpr::new(
                    Arc::new(rewrite(&sub.expr, hook)?),
                    lookup(&map, &sub.captures),
                )
                .with_symbol_captures(sub.symbol_captures.clone()))
            })
            .collect::<Result<Vec<_>, E>>()?;
        let outputs = eqn
            .outputs
            .iter()
            .map(|(var, info)| (builder.fresh_like(Some(*var), info.clone()), info.clone()))
            .collect();
        let translated =
            Eqn::new_with_subexprs(eqn.op.clone(), subexprs, lookup(&map, &eqn.inputs), outputs);
        let results = match hook(&mut builder, &translated)? {
            Some(results) => {
                assert_eq!(
                    results.len(),
                    eqn.outputs.len(),
                    "rewrite of {} returned the wrong number of results",
                    eqn.op
                );
                for ((_, expected), &var) in eqn.outputs.iter().zip(&results) {
                    debug_assert_eq!(
                        builder.info(var),
                        expected,
                        "rewrite of {} changed result metadata",
                        eqn.op
                    );
                }
                results
            }
            None => builder.push(translated),
        };
        for ((var, _), result) in eqn.outputs.iter().zip(results) {
            map.insert(*var, result);
        }
    }
    Ok(builder.finish(lookup(&map, &expr.outputs)))
}

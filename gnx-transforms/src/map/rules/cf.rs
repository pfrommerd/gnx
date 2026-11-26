//! Batching rules for the `cf` dialect.
//!
//! Once an expression is lowered, private cell state is part of the carry.
//! New region operations start with a private state count of zero.

use gnx_base::pass::Builder;
use gnx_expr::{Eqn, Expr, ExprBuilder, Ref, SubExprBuilder, Var, register_op_interface};

use crate::cf::{
    CallOperation, CondOperation, ScanOperation, ScanParameters, SwitchOperation, WhileOperation,
    branch_op, call_op, while_op,
};
use crate::map::batch::{BatchRule, Batched, batch_expr, batch_region, materialize};
use crate::util::{
    broadcast_in_dim, dims, infos, output_infos, reduce_sum, select, single_region, union,
    with_axis,
};
use crate::{TransformError, TransformResult};

fn vars(values: &[Batched]) -> Vec<Var> {
    values.iter().map(|value| value.var).collect()
}

impl BatchRule for CallOperation {
    fn batch(
        &self,
        b: &mut Builder,
        size: usize,
        eqn: Eqn<'_>,
        inputs: &[Batched],
        captures: &[Vec<Batched>],
    ) -> TransformResult<Vec<Batched>> {
        // As with JAX's `pjit`, emit a call to the batched body rather than
        // inlining it. Batched operands enter the body on axis 0.
        let (Some(sub), [captures]) = (single_region(eqn), captures) else {
            return Err(TransformError::invalid("call takes one body"));
        };
        let operands = inputs
            .iter()
            .map(|&value| match value.axis {
                Some(_) => Ok(materialize(b, value, size, 0)?),
                None => Ok(value.var),
            })
            .collect::<TransformResult<Vec<_>>>()?;
        let flags: Vec<_> = inputs.iter().map(|value| value.axis.is_some()).collect();
        let (region, outputs) = batch_region(b, size, sub.expr(), captures, &flags, None)?;
        let region = match (sub.expr().name_hint(), region.expr()) {
            (Some(name), Ref::Internal(expr)) => {
                SubExprBuilder::new(ExprBuilder::clone(expr).with_name_hint(format!("map({name})")))
                    .with_vars(region.provided_vars().iter().copied())
                    .with_symbols(region.provided_symbols().iter().copied())
            }
            _ => region,
        };
        let result_infos = output_infos(eqn)
            .iter()
            .zip(&outputs)
            .map(|(info, &batched)| {
                if batched {
                    with_axis(info, 0, size)
                } else {
                    Ok(info.clone())
                }
            })
            .collect::<TransformResult<Vec<_>>>()?;
        let results = b.emit(call_op(), &operands, result_infos, vec![region]);
        Ok(results
            .into_iter()
            .zip(outputs)
            .map(|(var, batched)| Batched {
                var,
                axis: batched.then_some(0),
            })
            .collect())
    }
}

fn batch_branches(
    b: &mut Builder,
    size: usize,
    eqn: Eqn<'_>,
    inputs: &[Batched],
    captures: &[Vec<Batched>],
) -> TransformResult<Vec<Batched>> {
    let (selector, operands) = inputs.split_first().expect("branch selector");
    let infos = output_infos(eqn);
    if selector.axis.is_some() {
        if eqn.op().downcast_ref::<CondOperation>().is_none() {
            return Err(TransformError::invalid(
                "batching a switch over a batched index is not supported",
            ));
        }
        // Evaluate both branches and select per instance.
        let predicate = materialize(b, *selector, size, 0)?;
        let mut branches = Vec::new();
        for (sub, captures) in eqn.subexprs().zip(captures) {
            let results = batch_expr(b, size, sub.expr(), captures, operands)?;
            branches.push(
                results
                    .into_iter()
                    .map(|value| materialize(b, value, size, 0))
                    .collect::<TransformResult<Vec<_>>>()?,
            );
        }
        let [on_true, on_false] = branches.as_slice() else {
            return Err(TransformError::invalid("cond has two branches"));
        };
        return on_true
            .iter()
            .zip(on_false)
            .map(|(&t, &f)| {
                let shape = dims(b.info(t))?;
                let mask = broadcast_in_dim(b, predicate, &shape, &[0])?;
                Ok(Batched::at(select(b, mask, t, f), 0))
            })
            .collect();
    }
    let operands = operands
        .iter()
        .map(|&value| {
            Ok(match value.axis {
                Some(_) => Batched::at(materialize(b, value, size, 0)?, 0),
                None => value,
            })
        })
        .collect::<TransformResult<Vec<_>>>()?;
    let flags: Vec<_> = operands.iter().map(|value| value.axis.is_some()).collect();
    let mut outputs = vec![false; infos.len()];
    for (sub, captures) in eqn.subexprs().zip(captures) {
        let (_, batched) = batch_region(b, size, sub.expr(), captures, &flags, None)?;
        union(&mut outputs, &batched);
    }
    let regions = eqn
        .subexprs()
        .zip(captures)
        .map(|(sub, captures)| {
            Ok(batch_region(b, size, sub.expr(), captures, &flags, Some(&outputs))?.0)
        })
        .collect::<TransformResult<Vec<_>>>()?;
    let result_infos = infos
        .iter()
        .zip(&outputs)
        .map(|(info, &batched)| {
            if batched {
                with_axis(info, 0, size)
            } else {
                Ok(info.clone())
            }
        })
        .collect::<TransformResult<Vec<_>>>()?;
    let mut args = vec![selector.var];
    args.extend(vars(&operands));
    let results = b.emit(branch_op(eqn), &args, result_infos, regions);
    Ok(results
        .into_iter()
        .zip(outputs)
        .map(|(var, batched)| Batched {
            var,
            axis: batched.then_some(0),
        })
        .collect())
}

macro_rules! branch_rules {
    ($($ty:ty),+) => {$(
        impl BatchRule for $ty {
            fn batch(
                &self,
                b: &mut Builder,
                size: usize,
                eqn: Eqn<'_>,
                inputs: &[Batched],
                captures: &[Vec<Batched>],
            ) -> TransformResult<Vec<Batched>> {
                batch_branches(b, size, eqn, inputs, captures)
            }
        }
    )+};
}
branch_rules!(CondOperation, SwitchOperation);

impl BatchRule for WhileOperation {
    fn batch(
        &self,
        b: &mut Builder,
        size: usize,
        eqn: Eqn<'_>,
        inputs: &[Batched],
        captures: &[Vec<Batched>],
    ) -> TransformResult<Vec<Batched>> {
        let regions: Vec<_> = eqn.subexprs().collect();
        let [predicate, body] = regions[..] else {
            return Err(TransformError::invalid("while needs a predicate and body"));
        };
        let mut flags: Vec<_> = inputs.iter().map(|value| value.axis.is_some()).collect();
        loop {
            let (_, batched) = batch_region(b, size, body.expr(), &captures[1], &flags, None)?;
            if !union(&mut flags, &batched) {
                break;
            }
        }
        let (_, predicate_batched) =
            batch_region(b, size, predicate.expr(), &captures[0], &flags, None)?;
        if predicate_batched[0] {
            // Instances may stop at different iterations: run while any
            // continues and keep the others' carries fixed.
            flags.iter_mut().for_each(|flag| *flag = true);
        }
        let carry = inputs
            .iter()
            .zip(&flags)
            .map(|(&value, &batched)| {
                if batched {
                    materialize(b, value, size, 0)
                } else {
                    Ok(value.var)
                }
            })
            .collect::<TransformResult<Vec<_>>>()?;
        let carry_infos = infos(b, &carry);
        let (predicate_region, body_region) = if predicate_batched[0] {
            let predicate_captures = &captures[0];
            let body_captures = &captures[1];
            let any = b.region(
                &vars(predicate_captures),
                &carry_infos,
                |rb, closure, params| {
                    let flags = mask(
                        rb,
                        size,
                        predicate.expr(),
                        predicate_captures,
                        closure,
                        params,
                    )?;
                    Ok::<_, TransformError>(vec![reduce_sum(rb, flags, &[0])?])
                },
            )?;
            let all_captures: Vec<_> = predicate_captures
                .iter()
                .chain(body_captures)
                .copied()
                .collect();
            let step = b.region(&vars(&all_captures), &carry_infos, |rb, closure, params| {
                let (predicate_closure, body_closure) = closure.split_at(predicate_captures.len());
                let active = mask(
                    rb,
                    size,
                    predicate.expr(),
                    predicate_captures,
                    predicate_closure,
                    params,
                )?;
                let closure: Vec<_> = body_closure
                    .iter()
                    .zip(body_captures)
                    .map(|(&var, capture)| Batched {
                        var,
                        axis: capture.axis,
                    })
                    .collect();
                let params_batched: Vec<_> =
                    params.iter().map(|&var| Batched::at(var, 0)).collect();
                let next = batch_expr(rb, size, body.expr(), &closure, &params_batched)?;
                next.into_iter()
                    .zip(params)
                    .map(|(value, &old)| {
                        let new = materialize(rb, value, size, 0)?;
                        let shape = dims(rb.info(new))?;
                        let active = broadcast_in_dim(rb, active, &shape, &[0])?;
                        Ok(select(rb, active, new, old))
                    })
                    .collect::<TransformResult<Vec<_>>>()
            })?;
            (any, step)
        } else {
            (
                batch_region(
                    b,
                    size,
                    predicate.expr(),
                    &captures[0],
                    &flags,
                    Some(&[false]),
                )?
                .0,
                batch_region(b, size, body.expr(), &captures[1], &flags, Some(&flags))?.0,
            )
        };
        let results = b.emit(
            while_op(),
            &carry,
            carry_infos,
            vec![predicate_region, body_region],
        );
        Ok(results
            .into_iter()
            .zip(flags)
            .map(|(var, batched)| Batched {
                var,
                axis: batched.then_some(0),
            })
            .collect())
    }
}

/// Evaluate a batched predicate region on carries batched at axis 0, as a
/// boolean vector over the instances.
fn mask(
    rb: &mut Builder,
    size: usize,
    predicate: Expr<'_>,
    captures: &[Batched],
    closure: &[Var],
    params: &[Var],
) -> TransformResult<Var> {
    let closure: Vec<_> = closure
        .iter()
        .zip(captures)
        .map(|(&var, capture)| Batched {
            var,
            axis: capture.axis,
        })
        .collect();
    let params: Vec<_> = params.iter().map(|&var| Batched::at(var, 0)).collect();
    let [flag] = batch_expr(rb, size, predicate, &closure, &params)?[..] else {
        return Err(TransformError::invalid("while predicate arity"));
    };
    materialize(rb, flag, size, 0)
}

impl BatchRule for ScanOperation {
    fn batch(
        &self,
        b: &mut Builder,
        size: usize,
        eqn: Eqn<'_>,
        inputs: &[Batched],
        captures: &[Vec<Batched>],
    ) -> TransformResult<Vec<Batched>> {
        let (carry_len, _) = ScanParameters::layout(eqn)?;
        let body = single_region(eqn)
            .ok_or_else(|| TransformError::invalid("scan needs one body"))?
            .expr();
        let captures = &captures[0];
        // Stacked inputs keep the scanned axis first; their batch axis goes to
        // 1 so that each slice is batched along axis 0.
        let (carry, xs) = inputs.split_at(carry_len);
        let xs: Vec<_> = xs
            .iter()
            .map(|&value| {
                Ok(match value.axis {
                    Some(_) => Batched::at(materialize(b, value, size, 1)?, 1),
                    None => value,
                })
            })
            .collect::<TransformResult<_>>()?;
        let mut carry_flags: Vec<_> = carry.iter().map(|value| value.axis.is_some()).collect();
        let x_flags: Vec<_> = xs.iter().map(|value| value.axis.is_some()).collect();
        let flags_for = |carry_flags: &[bool]| -> Vec<bool> {
            carry_flags.iter().chain(&x_flags).copied().collect()
        };
        let mut outputs = loop {
            let (_, out) = batch_region(b, size, body, captures, &flags_for(&carry_flags), None)?;
            if !union(&mut carry_flags, &out[..carry_len]) {
                break out;
            }
        };
        outputs[..carry_len].copy_from_slice(&carry_flags);
        let (region, _) = batch_region(
            b,
            size,
            body,
            captures,
            &flags_for(&carry_flags),
            Some(&outputs),
        )?;
        let mut args = carry
            .iter()
            .zip(&carry_flags)
            .map(|(&value, &batched)| {
                if batched {
                    materialize(b, value, size, 0)
                } else {
                    Ok(value.var)
                }
            })
            .collect::<TransformResult<Vec<_>>>()?;
        args.extend(vars(&xs));
        let result_infos = output_infos(eqn)
            .iter()
            .zip(&outputs)
            .enumerate()
            .map(
                |(index, (info, &batched))| match (batched, index < carry_len) {
                    (false, _) => Ok(info.clone()),
                    (true, true) => with_axis(info, 0, size),
                    (true, false) => with_axis(info, 1, size),
                },
            )
            .collect::<TransformResult<Vec<_>>>()?;
        let results = b.emit(eqn.op().to_owned(), &args, result_infos, vec![region]);
        Ok(results
            .into_iter()
            .zip(outputs)
            .enumerate()
            .map(|(index, (var, batched))| Batched {
                var,
                axis: batched.then_some(if index < carry_len { 0 } else { 1 }),
            })
            .collect())
    }
}

register_op_interface!(dyn BatchRule:
    CallOperation, CondOperation, SwitchOperation, WhileOperation, ScanOperation);

//! Forward-mode and transposition rules for the `cf` dialect.
//!
//! New region operations are emitted without the `state` attribute: once an
//! expression is lowered, private cell state is just more carry.

use gnx_expr::pass::Builder;
use gnx_expr::{Eqn, Expr, SubExpr, ValueInfo, Var, register_op_interface};

use crate::ad::jvp::{Dual, JvpRule, Linearized, Tangent, instantiate, jvp_expr, linearize};
use crate::ad::transpose::{Operand, TransposeRule, transpose_expr};
use crate::cf::{
    CallOperation, CondOperation, ScanOperation, ScanParameters, SwitchOperation, WhileOperation,
    branch_op, call_op, scan_op, while_op,
};
use crate::util::{infos, output_infos, rev, union, with_axis, without_axis, zeros};
use crate::{TransformError, TransformResult};

fn primals(duals: &[Dual]) -> Vec<Var> {
    duals.iter().map(|dual| dual.primal).collect()
}

fn nonzero(duals: &[Dual]) -> Vec<bool> {
    duals.iter().map(|dual| !dual.tangent.is_zero()).collect()
}

/// The flagged tangents, materialized.
fn flagged_tangents(b: &mut Builder, duals: &[Dual], flags: &[bool]) -> Vec<Var> {
    duals
        .iter()
        .zip(flags)
        .filter(|(_, flag)| **flag)
        .map(|(dual, _)| {
            let info = b.info(dual.primal).clone();
            instantiate(b, dual.tangent, &info)
        })
        .collect()
}

// ----- call -------------------------------------------------------------------
//
// As with JAX's `pjit`, the call boundary is preserved: forward mode emits a
// primal call (results and residuals) followed by a call that is linear in the
// tangents, so reverse mode sees a residual-only call to evaluate and a linear
// call to transpose into a third call.

/// Name a derived body after the original, e.g. `jvp(f)` or `transpose(f)`.
fn name_after(rb: &mut Builder, transform: &str, body: &Expr) {
    if let Some(name) = body.name_hint() {
        rb.set_name_hint(Some(format!("{transform}({name})").into()));
    }
}

impl JvpRule for CallOperation {
    fn jvp(
        &self,
        b: &mut Builder,
        eqn: &Eqn,
        inputs: &[Dual],
        captures: &[Vec<Dual>],
    ) -> TransformResult<Vec<Dual>> {
        let ([sub], [captures]) = (eqn.subexprs(), captures) else {
            return Err(TransformError::invalid("call takes one body"));
        };
        let body = sub.expr();
        let capture_flags = nonzero(captures);
        let input_flags = nonzero(inputs);
        let linear = linearize(body, &capture_flags, &input_flags, None)?;

        // Primal call: the original bindings; returns the results and residuals.
        let operands = primals(inputs);
        let region = b.region(
            &primals(captures),
            &infos(b, &operands),
            |rb, closure, params| {
                name_after(rb, "jvp", body);
                rb.inline(&linear.primal, &[], closure, params)
                    .map_err(TransformError::from)
            },
        )?;
        let mut primal_results = b.emit(
            call_op(),
            &operands,
            output_infos(eqn)
                .into_iter()
                .chain(linear.residuals.iter().cloned()),
            vec![region],
        );
        let residuals = primal_results.split_off(eqn.outputs().len());
        if !linear.outputs.iter().any(|&flag| flag) {
            return Ok(primal_results.into_iter().map(Dual::constant).collect());
        }

        // Linear call: residuals, then capture tangents, then input tangents.
        let mut args = residuals;
        args.extend(flagged_tangents(b, captures, &capture_flags));
        args.extend(flagged_tangents(b, inputs, &input_flags));
        let region = b.region(&[], &infos(b, &args), |rb, _, params| {
            name_after(rb, "lin", body);
            rb.inline(&linear.tangent, &[], &[], params)
                .map_err(TransformError::from)
        })?;
        let tangent_infos: Vec<_> = output_infos(eqn)
            .into_iter()
            .zip(&linear.outputs)
            .filter(|(_, flag)| **flag)
            .map(|(info, _)| info)
            .collect();
        let mut tangents = b
            .emit(call_op(), &args, tangent_infos, vec![region])
            .into_iter();
        Ok(primal_results
            .into_iter()
            .zip(&linear.outputs)
            .map(|(primal, &flag)| Dual {
                primal,
                tangent: if flag {
                    Tangent::Var(tangents.next().expect("tangent result"))
                } else {
                    Tangent::Zero
                },
            })
            .collect())
    }
}

impl TransposeRule for CallOperation {
    fn transpose(
        &self,
        b: &mut Builder,
        eqn: &Eqn,
        operands: &[Operand],
        captures: &[Vec<Var>],
        cotangents: &[Tangent],
    ) -> TransformResult<Vec<Tangent>> {
        let ([sub], [captures]) = (eqn.subexprs(), captures) else {
            return Err(TransformError::invalid("call takes one body"));
        };
        let body = sub.expr();
        let residuals: Vec<Var> = operands
            .iter()
            .filter_map(|operand| match operand {
                Operand::Residual(var) => Some(*var),
                Operand::Linear(_) => None,
            })
            .collect();
        let linear_infos: Vec<ValueInfo> = operands
            .iter()
            .filter_map(|operand| match operand {
                Operand::Linear(info) => Some(info.clone()),
                Operand::Residual(_) => None,
            })
            .collect();
        // Only nonzero cotangents cross the boundary.
        let cts: Vec<Var> = cotangents
            .iter()
            .filter_map(|ct| match ct {
                Tangent::Var(var) => Some(*var),
                Tangent::Zero => None,
            })
            .collect();
        let mut args = residuals.clone();
        args.extend(&cts);
        let region = b.region(captures, &infos(b, &args), |rb, closure, params| {
            name_after(rb, "transpose", body);
            let (residual_params, mut ct_params) = params.split_at(residuals.len());
            let mut residual_params = residual_params.iter();
            let bindings: Vec<_> = closure
                .iter()
                .map(|&var| Some(var))
                .chain(operands.iter().map(|operand| match operand {
                    Operand::Residual(_) => residual_params.next().copied(),
                    Operand::Linear(_) => None,
                }))
                .collect();
            let cotangents: Vec<_> = cotangents
                .iter()
                .map(|ct| match ct {
                    Tangent::Zero => Tangent::Zero,
                    Tangent::Var(_) => {
                        let (&param, rest) = ct_params.split_first().expect("cotangent");
                        ct_params = rest;
                        Tangent::Var(param)
                    }
                })
                .collect();
            let (_, param_cts) = transpose_expr(rb, body, &bindings, &cotangents)?;
            let operand_cts = &param_cts[closure.len()..];
            Ok::<_, TransformError>(
                operands
                    .iter()
                    .zip(operand_cts)
                    .filter_map(|(operand, &ct)| match operand {
                        Operand::Linear(info) => Some(instantiate(rb, ct, info)),
                        Operand::Residual(_) => None,
                    })
                    .collect(),
            )
        })?;
        let mut results = b
            .emit(call_op(), &args, linear_infos, vec![region])
            .into_iter();
        Ok(operands
            .iter()
            .map(|operand| match operand {
                Operand::Linear(_) => Tangent::Var(results.next().expect("cotangent result")),
                Operand::Residual(_) => Tangent::Zero,
            })
            .collect())
    }
}

// ----- cond / switch ----------------------------------------------------------

/// Linearize every branch with a common output tangent layout.
fn linearize_branches(
    eqn: &Eqn,
    operands: &[bool],
    captures: &[Vec<Dual>],
) -> TransformResult<Vec<Linearized>> {
    let mut outputs = vec![false; eqn.outputs().len()];
    for (sub, captures) in eqn.subexprs().iter().zip(captures) {
        let linear = linearize(sub.expr(), &nonzero(captures), operands, None)?;
        union(&mut outputs, &linear.outputs);
    }
    eqn.subexprs()
        .iter()
        .zip(captures)
        .map(|(sub, captures)| linearize(sub.expr(), &nonzero(captures), operands, Some(&outputs)))
        .collect()
}

fn jvp_branches(
    b: &mut Builder,
    eqn: &Eqn,
    inputs: &[Dual],
    captures: &[Vec<Dual>],
) -> TransformResult<Vec<Dual>> {
    let (selector, operands) = inputs.split_first().expect("branch selector");
    let operand_flags = nonzero(operands);
    let branches = linearize_branches(eqn, &operand_flags, captures)?;
    let outputs = branches[0].outputs.clone();
    let residual_infos: Vec<ValueInfo> = branches
        .iter()
        .flat_map(|linear| linear.residuals.iter().cloned())
        .collect();
    let offsets: Vec<usize> = branches
        .iter()
        .scan(0, |offset, linear| {
            let start = *offset;
            *offset += linear.residuals.len();
            Some(start)
        })
        .collect();

    // Primal branches return their results and every branch's residual slots.
    let operand_infos = infos(b, &primals(operands));
    let mut regions = Vec::new();
    for (index, (linear, captures)) in branches.iter().zip(captures).enumerate() {
        let region = b.region(&primals(captures), &operand_infos, |rb, closure, params| {
            let mut results = rb.inline(&linear.primal, &[], closure, params)?;
            let residuals = results.split_off(eqn.outputs().len());
            let mut slots = Vec::new();
            for (other, branch) in branches.iter().enumerate() {
                if other == index {
                    slots.extend(residuals.iter().copied());
                } else {
                    for info in &branch.residuals {
                        slots.push(zeros(rb, info));
                    }
                }
            }
            results.extend(slots);
            Ok::<_, TransformError>(results)
        })?;
        regions.push(region);
    }
    let mut args = vec![selector.primal];
    args.extend(primals(operands));
    let mut primal_results = b.emit(
        branch_op(eqn),
        &args,
        output_infos(eqn)
            .into_iter()
            .chain(residual_infos.iter().cloned()),
        regions,
    );
    let residuals = primal_results.split_off(eqn.outputs().len());

    if !outputs.iter().any(|&flag| flag) {
        return Ok(primal_results.into_iter().map(Dual::constant).collect());
    }

    // Tangent branches take every residual, the operand tangents, and every
    // branch's captured tangents; each uses its own.
    let operand_tangents = flagged_tangents(b, operands, &operand_flags);
    let capture_flags: Vec<_> = captures.iter().map(|captures| nonzero(captures)).collect();
    let capture_tangents: Vec<Vec<Var>> = captures
        .iter()
        .zip(&capture_flags)
        .map(|(captures, flags)| flagged_tangents(b, captures, flags))
        .collect();
    let mut tangent_args = residuals.clone();
    tangent_args.extend(&operand_tangents);
    tangent_args.extend(capture_tangents.iter().flatten());
    let tangent_infos = infos(b, &tangent_args);
    let mut regions = Vec::new();
    for (index, linear) in branches.iter().enumerate() {
        let region = b.region(&[], &tangent_infos, |rb, _, params| {
            let (all_residuals, rest) = params.split_at(residuals.len());
            let (operand_tangents, mut rest) = rest.split_at(operand_tangents.len());
            let mut own_captures = &[][..];
            for (other, tangents) in capture_tangents.iter().enumerate() {
                let (these, tail) = rest.split_at(tangents.len());
                if other == index {
                    own_captures = these;
                }
                rest = tail;
            }
            let start = offsets[index];
            let mut inputs = all_residuals[start..start + linear.residuals.len()].to_vec();
            inputs.extend(own_captures);
            inputs.extend(operand_tangents);
            rb.inline(&linear.tangent, &[], &[], &inputs)
                .map_err(TransformError::from)
        })?;
        regions.push(region);
    }
    let tangent_out_infos: Vec<_> = output_infos(eqn)
        .into_iter()
        .zip(&outputs)
        .filter(|(_, flag)| **flag)
        .map(|(info, _)| info)
        .collect();
    let mut args = vec![selector.primal];
    args.extend(tangent_args);
    let mut tangents = b
        .emit(branch_op(eqn), &args, tangent_out_infos, regions)
        .into_iter();
    Ok(primal_results
        .into_iter()
        .zip(outputs)
        .map(|(primal, flag)| Dual {
            primal,
            tangent: if flag {
                Tangent::Var(tangents.next().expect("tangent result"))
            } else {
                Tangent::Zero
            },
        })
        .collect())
}

fn transpose_branches(
    b: &mut Builder,
    eqn: &Eqn,
    operands: &[Operand],
    captures: &[Vec<Var>],
    cotangents: &[Tangent],
) -> TransformResult<Vec<Tangent>> {
    let [Operand::Residual(selector), operands @ ..] = operands else {
        return Err(TransformError::invalid(
            "a branch selector must be a constant",
        ));
    };
    let residuals: Vec<Var> = operands
        .iter()
        .filter_map(|operand| match operand {
            Operand::Residual(var) => Some(*var),
            Operand::Linear(_) => None,
        })
        .collect();
    let linear_infos: Vec<ValueInfo> = operands
        .iter()
        .filter_map(|operand| match operand {
            Operand::Linear(info) => Some(info.clone()),
            Operand::Residual(_) => None,
        })
        .collect();
    let cts: Vec<Var> = cotangents
        .iter()
        .zip(output_infos(eqn))
        .map(|(&ct, info)| instantiate(b, ct, &info))
        .collect();
    let mut region_inputs = infos(b, &residuals);
    region_inputs.extend(infos(b, &cts));
    let mut regions = Vec::new();
    for (sub, captures) in eqn.subexprs().iter().zip(captures) {
        let region = b.region(captures, &region_inputs, |rb, closure, params| {
            let (residual_params, ct_params) = params.split_at(residuals.len());
            let mut residual_params = residual_params.iter();
            let bindings: Vec<_> = closure
                .iter()
                .map(|&var| Some(var))
                .chain(operands.iter().map(|operand| match operand {
                    Operand::Residual(_) => residual_params.next().copied(),
                    Operand::Linear(_) => None,
                }))
                .collect();
            let cotangents: Vec<_> = ct_params.iter().map(|&var| Tangent::Var(var)).collect();
            let (_, param_cts) = transpose_expr(rb, sub.expr(), &bindings, &cotangents)?;
            let operand_cts = &param_cts[closure.len()..];
            Ok::<_, TransformError>(
                operands
                    .iter()
                    .zip(operand_cts)
                    .filter_map(|(operand, &ct)| match operand {
                        Operand::Linear(info) => Some(instantiate(rb, ct, info)),
                        Operand::Residual(_) => None,
                    })
                    .collect(),
            )
        })?;
        regions.push(region);
    }
    let mut args = vec![*selector];
    args.extend(&residuals);
    args.extend(&cts);
    let mut results = b
        .emit(branch_op(eqn), &args, linear_infos, regions)
        .into_iter();
    Ok(std::iter::once(Tangent::Zero)
        .chain(operands.iter().map(|operand| match operand {
            Operand::Linear(_) => Tangent::Var(results.next().expect("cotangent result")),
            Operand::Residual(_) => Tangent::Zero,
        }))
        .collect())
}

macro_rules! branch_rules {
    ($($ty:ty),+) => {$(
        impl JvpRule for $ty {
            fn jvp(
                &self,
                b: &mut Builder,
                eqn: &Eqn,
                inputs: &[Dual],
                captures: &[Vec<Dual>],
            ) -> TransformResult<Vec<Dual>> {
                jvp_branches(b, eqn, inputs, captures)
            }
        }
        impl TransposeRule for $ty {
            fn transpose(
                &self,
                b: &mut Builder,
                eqn: &Eqn,
                operands: &[Operand],
                captures: &[Vec<Var>],
                cotangents: &[Tangent],
            ) -> TransformResult<Vec<Tangent>> {
                transpose_branches(b, eqn, operands, captures, cotangents)
            }
        }
    )+};
}
branch_rules!(CondOperation, SwitchOperation);

// ----- while_loop -------------------------------------------------------------

/// Forward mode for a region whose explicit inputs are primals followed by the
/// flagged tangents, returning primals followed by the flagged result tangents.
fn jvp_region(
    b: &Builder,
    body: &Expr,
    captures: &[Dual],
    inputs: &[bool],
    outputs: Option<&[bool]>,
) -> TransformResult<(SubExpr, Vec<bool>)> {
    let capture_tangents: Vec<Var> = captures
        .iter()
        .filter_map(|dual| match dual.tangent {
            Tangent::Var(var) => Some(var),
            Tangent::Zero => None,
        })
        .collect();
    let region_captures: Vec<_> = primals(captures)
        .into_iter()
        .chain(capture_tangents)
        .collect();
    let primal_infos: Vec<_> = body.inputs().iter().map(|(_, info)| info.clone()).collect();
    let mut region_inputs = primal_infos.clone();
    region_inputs.extend(
        primal_infos
            .iter()
            .zip(inputs)
            .filter(|(_, flag)| **flag)
            .map(|(info, _)| info.clone()),
    );
    let mut flags = Vec::new();
    let region = b.region(&region_captures, &region_inputs, |rb, closure, params| {
        let (closure_primals, mut closure_tangents) = closure.split_at(captures.len());
        let closure: Vec<_> = closure_primals
            .iter()
            .zip(captures)
            .map(|(&primal, capture)| Dual {
                primal,
                tangent: if capture.tangent.is_zero() {
                    Tangent::Zero
                } else {
                    let (&tangent, rest) = closure_tangents.split_first().expect("tangent");
                    closure_tangents = rest;
                    Tangent::Var(tangent)
                },
            })
            .collect();
        let (params_primal, mut params_tangent) = params.split_at(primal_infos.len());
        let duals: Vec<_> = params_primal
            .iter()
            .zip(inputs)
            .map(|(&primal, &flag)| Dual {
                primal,
                tangent: if flag {
                    let (&tangent, rest) = params_tangent.split_first().expect("tangent");
                    params_tangent = rest;
                    Tangent::Var(tangent)
                } else {
                    Tangent::Zero
                },
            })
            .collect();
        let results = jvp_expr(rb, body, &closure, &duals)?;
        flags = match outputs {
            Some(flags) => flags.to_vec(),
            None => nonzero(&results),
        };
        if results
            .iter()
            .zip(&flags)
            .any(|(dual, &flag)| !flag && !dual.tangent.is_zero())
        {
            return Err(TransformError::invalid("unexpected nonzero tangent"));
        }
        let mut vars = primals(&results);
        vars.extend(flagged_tangents(rb, &results, &flags));
        Ok(vars)
    })?;
    Ok((region, flags))
}

impl JvpRule for WhileOperation {
    fn jvp(
        &self,
        b: &mut Builder,
        eqn: &Eqn,
        inputs: &[Dual],
        captures: &[Vec<Dual>],
    ) -> TransformResult<Vec<Dual>> {
        let [predicate, body] = eqn.subexprs() else {
            return Err(TransformError::invalid("while needs a predicate and body"));
        };
        let mut flags = nonzero(inputs);
        loop {
            let (_, out) = jvp_region(b, body.expr(), &captures[1], &flags, None)?;
            if !union(&mut flags, &out) {
                break;
            }
        }
        let (body_region, _) = jvp_region(b, body.expr(), &captures[1], &flags, Some(&flags))?;
        // The predicate ignores the tangent carries.
        let tangent_count = flags.iter().filter(|flag| **flag).count();
        let carry_primals = primals(inputs);
        let mut carry = carry_primals.clone();
        carry.extend(flagged_tangents(b, inputs, &flags));
        let carry_infos = infos(b, &carry);
        let predicate_captures = primals(&captures[0]);
        let predicate_region =
            b.region(&predicate_captures, &carry_infos, |rb, closure, params| {
                rb.inline(
                    predicate.expr(),
                    &predicate.symbol_captures,
                    closure,
                    &params[..params.len() - tangent_count],
                )
                .map_err(TransformError::from)
            })?;
        let results = b.emit(
            while_op(),
            &carry,
            carry_infos,
            vec![predicate_region, body_region],
        );
        let (primal_results, tangent_results) = results.split_at(inputs.len());
        let mut tangent_results = tangent_results.iter();
        Ok(primal_results
            .iter()
            .zip(&flags)
            .map(|(&primal, &flag)| Dual {
                primal,
                tangent: if flag {
                    Tangent::Var(*tangent_results.next().expect("tangent carry"))
                } else {
                    Tangent::Zero
                },
            })
            .collect())
    }
}

// ----- scan -------------------------------------------------------------------

impl JvpRule for ScanOperation {
    fn jvp(
        &self,
        b: &mut Builder,
        eqn: &Eqn,
        inputs: &[Dual],
        captures: &[Vec<Dual>],
    ) -> TransformResult<Vec<Dual>> {
        let (carry_len, length) = ScanParameters::layout(eqn)?;
        let body = eqn.subexprs()[0].expr();
        let captures = &captures[0];
        let capture_flags = nonzero(captures);
        let mut flags = nonzero(inputs);
        let mut outputs = loop {
            let linear = linearize(body, &capture_flags, &flags, None)?;
            if !union(&mut flags[..carry_len], &linear.outputs[..carry_len]) {
                break linear.outputs;
            }
        };
        outputs[..carry_len].copy_from_slice(&flags[..carry_len]);
        let linear = linearize(body, &capture_flags, &flags, Some(&outputs))?;
        let ys = eqn.outputs().len() - carry_len;

        // Primal scan: stacks the residuals of each step as extra outputs.
        let body_inputs: Vec<_> = body.inputs().iter().map(|(_, info)| info.clone()).collect();
        let primal_region = b.region(&primals(captures), &body_inputs, |rb, closure, params| {
            rb.inline(&linear.primal, &[], closure, params)
                .map_err(TransformError::from)
        })?;
        let stacked = linear
            .residuals
            .iter()
            .map(|info| with_axis(info, 0, length))
            .collect::<TransformResult<Vec<_>>>()?;
        let mut primal_results = b.emit(
            scan_op(carry_len, length),
            &primals(inputs),
            output_infos(eqn).into_iter().chain(stacked),
            vec![primal_region],
        );
        let residuals = primal_results.split_off(carry_len + ys);
        if !outputs.iter().any(|&flag| flag) {
            return Ok(primal_results.into_iter().map(Dual::constant).collect());
        }

        // Tangent scan: carries the captured tangents unchanged, then the
        // tangent carries; scans the residuals and the input tangents.
        let capture_tangents = flagged_tangents(b, captures, &capture_flags);
        let (carry, xs) = inputs.split_at(carry_len);
        let carry_tangents = flagged_tangents(b, carry, &flags[..carry_len]);
        let x_tangents = flagged_tangents(b, xs, &flags[carry_len..]);
        let mut args = capture_tangents.clone();
        args.extend(&carry_tangents);
        args.extend(&residuals);
        args.extend(&x_tangents);
        let slice_infos = args
            .iter()
            .enumerate()
            .map(|(index, &var)| {
                if index < capture_tangents.len() + carry_tangents.len() {
                    Ok(b.info(var).clone())
                } else {
                    Ok(without_axis(b.info(var), 0)?.1)
                }
            })
            .collect::<TransformResult<Vec<_>>>()?;
        let region = b.region(&[], &slice_infos, |rb, _, params| {
            let (passthrough, rest) = params.split_at(capture_tangents.len());
            let (carries, rest) = rest.split_at(carry_tangents.len());
            let (residuals, x_tangents) = rest.split_at(linear.residuals.len());
            let mut inputs = residuals.to_vec();
            inputs.extend(passthrough);
            inputs.extend(carries);
            inputs.extend(x_tangents);
            let mut results = passthrough.to_vec();
            results.extend(rb.inline(&linear.tangent, &[], &[], &inputs)?);
            Ok::<_, TransformError>(results)
        })?;
        let infos_out: Vec<_> = capture_tangents
            .iter()
            .map(|&var| b.info(var).clone())
            .chain(
                output_infos(eqn)
                    .into_iter()
                    .zip(&outputs)
                    .filter(|(_, flag)| **flag)
                    .map(|(info, _)| info),
            )
            .collect();
        let tangent_carry_len = capture_tangents.len() + carry_tangents.len();
        let tangents = b.emit(
            scan_op(tangent_carry_len, length),
            &args,
            infos_out,
            vec![region],
        );
        let mut tangents = tangents[capture_tangents.len()..].iter();
        Ok(primal_results
            .into_iter()
            .zip(outputs)
            .map(|(primal, flag)| Dual {
                primal,
                tangent: if flag {
                    Tangent::Var(*tangents.next().expect("tangent result"))
                } else {
                    Tangent::Zero
                },
            })
            .collect())
    }
}

impl TransposeRule for ScanOperation {
    fn transpose(
        &self,
        b: &mut Builder,
        eqn: &Eqn,
        operands: &[Operand],
        captures: &[Vec<Var>],
        cotangents: &[Tangent],
    ) -> TransformResult<Vec<Tangent>> {
        let (carry_len, length) = ScanParameters::layout(eqn)?;
        let body = eqn.subexprs()[0].expr();
        // Every carry is treated as linear: a constant initial carry here is a
        // materialized zero tangent, and its cotangent is discarded.
        let (carry, xs) = operands.split_at(carry_len);
        let infos_out = output_infos(eqn);
        let cts: Vec<Var> = cotangents
            .iter()
            .zip(&infos_out)
            .map(|(&ct, info)| instantiate(b, ct, info))
            .collect();
        let (carry_cts, y_cts) = cts.split_at(carry_len);
        let residual_xs: Vec<Var> = xs
            .iter()
            .filter_map(|operand| match operand {
                Operand::Residual(var) => Some(*var),
                Operand::Linear(_) => None,
            })
            .collect();
        // Iterate in reverse by reversing the stacked inputs and outputs.
        let mut args = carry_cts.to_vec();
        for &var in residual_xs.iter().chain(y_cts) {
            args.push(rev(b, var, &[0]));
        }
        let slice_infos = args
            .iter()
            .enumerate()
            .map(|(index, &var)| {
                if index < carry_len {
                    Ok(b.info(var).clone())
                } else {
                    Ok(without_axis(b.info(var), 0)?.1)
                }
            })
            .collect::<TransformResult<Vec<_>>>()?;
        let region = b.region(&captures[0], &slice_infos, |rb, closure, params| {
            let (carry_params, rest) = params.split_at(carry_len);
            let (residual_params, y_params) = rest.split_at(residual_xs.len());
            let mut residual_params = residual_params.iter();
            let bindings: Vec<_> = closure
                .iter()
                .map(|&var| Some(var))
                .chain(carry.iter().map(|_| None))
                .chain(xs.iter().map(|operand| match operand {
                    Operand::Residual(_) => residual_params.next().copied(),
                    Operand::Linear(_) => None,
                }))
                .collect();
            let cotangents: Vec<_> = carry_params
                .iter()
                .chain(y_params)
                .map(|&var| Tangent::Var(var))
                .collect();
            let (_, param_cts) = transpose_expr(rb, body, &bindings, &cotangents)?;
            let param_cts = &param_cts[closure.len()..];
            let mut results = Vec::new();
            for ((_, info), &ct) in body.inputs().iter().zip(param_cts).take(carry_len) {
                results.push(instantiate(rb, ct, info));
            }
            for ((operand, (_, info)), &ct) in xs
                .iter()
                .zip(&body.inputs()[carry_len..])
                .zip(&param_cts[carry_len..])
            {
                if operand.is_linear() {
                    results.push(instantiate(rb, ct, info));
                }
            }
            Ok::<_, TransformError>(results)
        })?;
        let carry_infos = infos_out[..carry_len].to_vec();
        let stacked_infos: Vec<_> = xs
            .iter()
            .filter_map(|operand| match operand {
                Operand::Linear(info) => Some(info.clone()),
                Operand::Residual(_) => None,
            })
            .collect();
        let results = b.emit(
            scan_op(carry_len, length),
            &args,
            carry_infos.into_iter().chain(stacked_infos),
            vec![region],
        );
        let (carry_results, x_results) = results.split_at(carry_len);
        let mut x_results = x_results.iter();
        let mut output: Vec<_> = carry_results
            .iter()
            .zip(carry)
            .map(|(&var, operand)| {
                if operand.is_linear() {
                    Tangent::Var(var)
                } else {
                    Tangent::Zero
                }
            })
            .collect();
        for operand in xs {
            output.push(match operand {
                Operand::Linear(_) => {
                    Tangent::Var(rev(b, *x_results.next().expect("stacked cotangent"), &[0]))
                }
                Operand::Residual(_) => Tangent::Zero,
            });
        }
        Ok(output)
    }
}

register_op_interface!(dyn JvpRule:
    CallOperation, CondOperation, SwitchOperation, WhileOperation, ScanOperation);
register_op_interface!(dyn TransposeRule:
    CallOperation, CondOperation, SwitchOperation, ScanOperation);

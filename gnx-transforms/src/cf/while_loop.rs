//! `cf.while_loop`: iterate a captured body while a predicate holds.

use std::fmt;

use gnx_base::DType;
use gnx_expr::lower::{LoweringContext, RegionArgument};
use gnx_expr::{Capture, Eqn, HEqn, Invocation, Kind, ObjectTracer, Op, Operation, Trace};
use rkyv::{Archive, Serialize};

use super::{
    RegionResult, StateLayout, add_state, check_child, check_state_len, count, expect_inputs,
    expr_output_infos, fmt_control_flow, scalar, signature, stored, value_infos, value_results,
};

/// Builds a `cf::while_loop`, stored as a [`WhileOperation`].
#[derive(Archive, Serialize)]
#[rkyv(archived = WhileOperation)]
pub struct While {
    state_len: u64,
}

impl WhileOperation {
    /// Number of private cell-state slots threaded through the operation.
    pub fn state_len(&self) -> usize {
        count(self.state_len)
    }
}

pub fn while_op() -> Op {
    Op::new(While { state_len: 0 })
}

/// Trace `carry = body(&carry)` while `predicate(&carry)` is true.
/// Each closure is traced once; the predicate must return a scalar boolean array.
/// The carry may be an array, tuple, or graph struct. Its structure, static fields,
/// and array shapes and dtypes must stay fixed. Capture mutable cells in the
/// closures instead of passing them in the carry; only the body may write them.
pub fn while_loop<C>(
    carry: &C,
    predicate: impl FnOnce(&C) -> gnx_base::Array,
    body: impl FnOnce(&C) -> C,
) -> crate::TransformResult<C>
where
    C: gnx_graph::Graph<Owned = C> + gnx_graph::GraphEq,
{
    use crate::cache::{rebuild_graph, split_graph};
    use crate::traced::{trace_region, value};

    let (builder, leaves) = split_graph(carry);
    let values = leaves
        .iter()
        .map(|v| value(v, "while carry"))
        .collect::<crate::TransformResult<Vec<_>>>()?;
    let infos: Vec<_> = values.iter().map(|v| v.info().clone()).collect();
    let rebuild =
        |args: &[Trace]| rebuild_graph(&builder, args.iter().cloned().map(ObjectTracer::from));
    let predicate = trace_region(&infos, |args| {
        Ok(vec![
            predicate(&rebuild(args)).into_tracer().into_trace_ref(),
        ])
    })?;
    let body = trace_region(&infos, |args| {
        let (next, leaves) = split_graph(&body(&rebuild(args)));
        if !C::builders_eq::<ObjectTracer>(&builder, &next, &mut Default::default()) {
            return Err(crate::TransformError::invalid(
                "while carry structure changed",
            ));
        }
        leaves
            .iter()
            .map(|v| value(v, "while body result"))
            .collect()
    })?;
    Ok(rebuild(&while_loop_captured(&predicate, &body, &values)?))
}

/// Loop over immutable carry values. `predicate` returns a single bool scalar;
/// it may read captured cells but cannot write them. `body` returns the original
/// carry signature and may read and write captured cells.
pub fn while_loop_captured(
    predicate: &Capture,
    body: &Capture,
    carry: &[Trace],
) -> RegionResult<Vec<Trace>> {
    let infos: Vec<_> = carry.iter().map(|trace| trace.info().clone()).collect();
    for region in [predicate, body] {
        expect_inputs(
            region.expr(),
            infos.iter().cloned().map(|info| (Kind::Value, info)),
        )?;
    }
    if value_results(predicate.expr())? != [scalar(DType::Bool)]
        || value_results(body.expr())? != infos
    {
        return Err(signature("while predicate or carry result signature"));
    }
    if predicate.expr().writes_parameters() {
        return Err(signature("while predicate cannot write a captured cell"));
    }
    Ok(Invocation::apply(
        while_op(),
        vec![predicate.into(), body.into()],
        carry.iter().cloned().map(ObjectTracer::from).collect(),
        infos,
    ))
}

impl Operation for WhileOperation {
    fn display(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_control_flow(f, "while_loop", self.state_len())
    }

    fn lower(&self, eqn: HEqn<'_>, ctx: &mut LoweringContext<'_>) -> RegionResult<()> {
        let mut subexprs = eqn.subexprs();
        let (Some(predicate), Some(body), None) =
            (subexprs.next(), subexprs.next(), subexprs.next())
        else {
            return Err(signature("while needs a predicate and body"));
        };
        let (values, infos): (Vec<_>, Vec<_>) = eqn
            .inputs()
            .iter()
            .map(|&var| ctx.value_operand(var))
            .collect::<RegionResult<Vec<_>>>()?
            .into_iter()
            .unzip();
        if value_infos(eqn.outputs())? != infos {
            return Err(signature("while carry result signature"));
        }
        check_state_len(self.state_len(), infos.len(), infos.len())?;
        let args = vec![RegionArgument::Value; infos.len()];
        let regions = [
            ctx.lower_structured_region(predicate, &args, &infos)?,
            ctx.lower_structured_region(body, &args, &infos)?,
        ];
        if regions[0].writes().next().is_some() {
            return Err(signature("while predicate cannot write a captured cell"));
        }
        let state = StateLayout::new(ctx, &regions)?;
        let size = infos.len();
        let fixed = [
            regions[0].fix(ctx, &state.slots, size, None, &[scalar(DType::Bool)])?,
            regions[1].fix(ctx, &state.slots, size, Some(size), &infos)?,
        ];
        let op = Op::new(While {
            state_len: stored(add_state(self.state_len(), state.cells.len())?),
        });
        state.emit(ctx, eqn, op, values, (size, size), &fixed)
    }

    fn check_expr(&self, eqn: Eqn<'_>) -> RegionResult<()> {
        let mut subexprs = eqn.subexprs();
        let (Some(predicate), Some(body), None) =
            (subexprs.next(), subexprs.next(), subexprs.next())
        else {
            return Err(signature("while needs a predicate and body"));
        };
        let carry = expr_output_infos(eqn);
        if carry.len() != eqn.inputs().len() {
            return Err(signature("while carry count"));
        }
        check_state_len(self.state_len(), carry.len(), carry.len())?;
        check_child(
            predicate.expr(),
            carry.iter().cloned(),
            &[scalar(DType::Bool)],
        )?;
        check_child(body.expr(), carry.iter().cloned(), &carry)
    }
}

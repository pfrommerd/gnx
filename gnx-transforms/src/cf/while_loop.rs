//! `cf.while_loop`: iterate a captured body while a predicate holds.

use std::fmt;

use gnx_common::DType;
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

/// Trace a while loop with one array carry. The body can also update captured
/// `ArrayRef`s; these become loop state just as in the capture-level API.
pub fn while_array(
    carry: &gnx_common::Array,
    predicate: impl FnOnce(&gnx_common::Array) -> gnx_common::Array,
    body: impl FnOnce(&gnx_common::Array) -> gnx_common::Array,
) -> crate::TransformResult<gnx_common::Array> {
    fn array(t: Trace) -> gnx_common::Array {
        gnx_expr::Tracer::<gnx_expr::Generic>::from(t)
            .try_cast::<gnx_common::Array>()
            .expect("array carry")
            .into()
    }
    fn region(
        carry: &Trace,
        f: impl FnOnce(&gnx_common::Array) -> gnx_common::Array,
    ) -> crate::TransformResult<Capture> {
        crate::traced::trace_region(std::slice::from_ref(carry.info()), |inputs| {
            Ok(vec![
                f(&array(inputs[0].clone())).into_tracer().into_trace_ref(),
            ])
        })
    }
    let carry = carry.tracer().trace_ref();
    let pred = region(carry, predicate)?;
    let body = region(carry, body)?;
    Ok(array(
        while_loop(&pred, &body, std::slice::from_ref(carry))?.remove(0),
    ))
}

/// Loop over immutable carry values. `predicate` returns a single bool scalar;
/// it may read captured cells but cannot write them. `body` returns the original
/// carry signature and may read and write captured cells.
pub fn while_loop(
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

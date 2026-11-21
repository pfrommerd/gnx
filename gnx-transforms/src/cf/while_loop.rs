//! `cf.while_loop`: iterate a captured body while a predicate holds.

use std::fmt;

use gnx_common::DType;
use gnx_expr::lower::{LoweringContext, RegionArgument};
use gnx_expr::{Capture, Dialect, Eqn, HEqn, Kind, ObjectTracer, Op, Operation, Trace};

use super::{
    BoundRegion, CONTROL_FLOW, RegionResult, StateLayout, add_state, check_child, check_state_len,
    defer_regions, expect_inputs, expr_output_infos, fmt_control_flow, scalar, signature,
    value_infos, value_results,
};

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct WhileOperation {
    state_len: usize,
}

pub fn while_op() -> Op {
    Op::new(WhileOperation { state_len: 0 })
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
    defer_regions(
        while_op(),
        vec![
            BoundRegion {
                capture: predicate,
                inputs: &[],
                read_only: true,
            },
            BoundRegion {
                capture: body,
                inputs: &[],
                read_only: false,
            },
        ],
        carry.iter().cloned().map(ObjectTracer::from).collect(),
        infos,
    )
}

impl fmt::Display for WhileOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_control_flow(f, "while_loop", self.state_len)
    }
}

impl Operation for WhileOperation {
    fn name(&self) -> &'static str {
        "while_loop"
    }
    fn dialect(&self) -> &'static dyn Dialect {
        &CONTROL_FLOW
    }

    fn lower(&self, eqn: &HEqn, ctx: &mut LoweringContext<'_>) -> RegionResult<()> {
        let [predicate, body] = eqn.subexprs() else {
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
        check_state_len(self.state_len, infos.len(), infos.len())?;
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
        let op = Op::new(Self {
            state_len: add_state(self.state_len, state.cells.len())?,
        });
        state.emit(ctx, eqn, op, values, (size, size), &fixed)
    }

    fn check_expr(&self, eqn: &Eqn) -> RegionResult<()> {
        let [predicate, body] = eqn.subexprs() else {
            return Err(signature("while needs a predicate and body"));
        };
        let carry = expr_output_infos(eqn);
        if carry.len() != eqn.inputs().len() {
            return Err(signature("while carry count"));
        }
        check_state_len(self.state_len, carry.len(), carry.len())?;
        check_child(
            predicate.expr(),
            carry.iter().cloned(),
            &[scalar(DType::Bool)],
        )?;
        check_child(body.expr(), carry.iter().cloned(), &carry)
    }
}

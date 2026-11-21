//! `cf.cond` and `cf.switch`: select one of several captured branches.

use std::fmt;

use gnx_common::DType;
use gnx_expr::lower::{LoweringContext, RegionArgument};
use gnx_expr::{
    Capture, CellAccess, Dialect, Eqn, HEqn, Kind, ObjectTracer, Op, Operation, Trace, Var,
};

use super::{
    BoundRegion, CONTROL_FLOW, RegionResult, StateLayout, add_state, check_child, check_state_len,
    defer_regions, expect_inputs, expr_output_infos, fmt_control_flow, object_kind, parameters,
    scalar, signature, value_infos, value_results,
};

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct CondOperation {
    state_len: usize,
}
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct SwitchOperation {
    state_len: usize,
}

pub fn cond_op() -> Op {
    Op::new(CondOperation { state_len: 0 })
}

pub fn switch_op() -> Op {
    Op::new(SwitchOperation { state_len: 0 })
}

/// A fresh `cond` or `switch` (without private state) of the same kind as
/// `eqn`, for transformation rules that emit a new branch operation.
pub(crate) fn branch_op(eqn: &Eqn) -> Op {
    if eqn.op().downcast_ref::<CondOperation>().is_some() {
        cond_op()
    } else {
        switch_op()
    }
}

/// Select one of two captured branches using a scalar boolean array. Both
/// branches take the same arguments and must return matching immutable values.
/// They may read or write cells in their arguments or captures.
pub fn cond(
    predicate: Trace,
    on_true: &Capture,
    on_false: &Capture,
    inputs: &[ObjectTracer],
) -> RegionResult<Vec<Trace>> {
    record_branch(
        cond_op(),
        DType::Bool,
        predicate,
        &[on_true, on_false],
        inputs,
    )
}

/// Select a captured branch using a scalar i32 array. Indices are clamped to
/// the valid range. All branches take the same arguments and return the same
/// value signature.
pub fn switch(
    index: Trace,
    branches: &[&Capture],
    inputs: &[ObjectTracer],
) -> RegionResult<Vec<Trace>> {
    record_branch(switch_op(), DType::I32, index, branches, inputs)
}

fn record_branch(
    op: Op,
    dtype: DType,
    selector: Trace,
    branches: &[&Capture],
    inputs: &[ObjectTracer],
) -> RegionResult<Vec<Trace>> {
    let Some(first) = branches.first() else {
        return Err(signature("switch needs at least one branch"));
    };
    if selector.info() != &scalar(dtype) {
        return Err(signature("branch selector type"));
    }
    let infos = value_results(first.expr())?;
    for branch in branches {
        expect_inputs(branch.expr(), inputs.iter().map(object_kind))?;
        if value_results(branch.expr())? != infos {
            return Err(signature("branch result signatures must match"));
        }
    }
    defer_regions(
        op,
        branches
            .iter()
            .map(|branch| BoundRegion {
                capture: branch,
                inputs,
                read_only: false,
            })
            .collect(),
        std::iter::once(selector.into())
            .chain(inputs.iter().cloned())
            .collect(),
        infos,
    )
}

fn lower_branch(
    eqn: &HEqn,
    ctx: &mut LoweringContext<'_>,
    dtype: DType,
    state_len: usize,
    make_op: impl FnOnce(usize) -> Op,
) -> RegionResult<()> {
    let [selector, operands @ ..] = eqn.inputs() else {
        return Err(signature("missing branch selector"));
    };
    let (selector, info) = ctx.value_operand(*selector)?;
    if info != scalar(dtype) {
        return Err(signature("branch selector type"));
    }
    check_state_len(state_len, operands.len(), eqn.outputs().len())?;
    let first = eqn
        .subexprs()
        .first()
        .ok_or_else(|| signature("missing branch"))?;
    if first.expr().inputs().len() != operands.len() {
        return Err(signature("branch operand arity"));
    }
    let mut values = vec![selector];
    let mut infos = Vec::new();
    let mut args = Vec::new();
    for (&operand, (_, kind, _)) in operands.iter().zip(first.expr().inputs()) {
        match kind {
            Kind::Value => {
                let (value, info) = ctx.value_operand(operand)?;
                values.push(value);
                infos.push(info);
                args.push(RegionArgument::Value);
            }
            Kind::Cell => args.push(RegionArgument::Cell(operand)),
        }
    }
    let regions = eqn
        .subexprs()
        .iter()
        .map(|sub| ctx.lower_structured_region(sub, &args, &infos))
        .collect::<RegionResult<Vec<_>>>()?;
    let state = StateLayout::new(ctx, &regions)?;
    let outputs = value_infos(eqn.outputs())?;
    let fixed = regions
        .iter()
        .map(|region| {
            region.fix(
                ctx,
                &state.slots,
                infos.len(),
                Some(outputs.len()),
                &outputs,
            )
        })
        .collect::<RegionResult<Vec<_>>>()?;
    let input_len = values.len();
    let op = make_op(add_state(state_len, state.cells.len())?);
    state.emit(ctx, eqn, op, values, (input_len, outputs.len()), &fixed)
}

fn check_branches(eqn: &Eqn, state_len: usize) -> RegionResult<()> {
    let first = eqn
        .subexprs()
        .first()
        .ok_or_else(|| signature("missing branch"))?
        .expr();
    let outputs = expr_output_infos(eqn);
    if eqn.inputs().len().checked_sub(1) != Some(first.inputs().len()) {
        return Err(signature("branch operand signature"));
    }
    check_state_len(state_len, first.inputs().len(), outputs.len())?;
    for sub in eqn.subexprs() {
        check_child(sub.expr(), parameters(first).cloned(), &outputs)?;
    }
    Ok(())
}

macro_rules! branch_op {
    ($ty:ident, $name:literal, $dtype:expr, $arity:expr) => {
        impl fmt::Display for $ty {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt_control_flow(f, $name, self.state_len)
            }
        }

        impl Operation for $ty {
            fn name(&self) -> &'static str {
                $name
            }
            fn dialect(&self) -> &'static dyn Dialect {
                &CONTROL_FLOW
            }

            fn cell_access(&self, eqn: &HEqn, cell: Var) -> CellAccess {
                let operands = eqn.inputs().get(1..).unwrap_or_default();
                eqn.subexprs()
                    .iter()
                    .fold(eqn.captured_cell_access(cell), |access, body| {
                        access.combine(body.input_cell_access(operands, cell))
                    })
            }
            fn lower(&self, eqn: &HEqn, ctx: &mut LoweringContext<'_>) -> RegionResult<()> {
                if !($arity)(eqn.subexprs().len()) {
                    return Err(signature("branch count"));
                }
                lower_branch(eqn, ctx, $dtype, self.state_len, |state_len| {
                    Op::new($ty { state_len })
                })
            }
            fn check_expr(&self, eqn: &Eqn) -> RegionResult<()> {
                if !($arity)(eqn.subexprs().len()) {
                    return Err(signature("branch count"));
                }
                check_branches(eqn, self.state_len)
            }
        }
    };
}
branch_op!(CondOperation, "cond", DType::Bool, |n| n == 2);
branch_op!(SwitchOperation, "switch", DType::I32, |n| n > 0);

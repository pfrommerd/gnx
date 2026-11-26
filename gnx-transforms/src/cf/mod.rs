//! Trace and lower the `cf` dialect's control-flow operations.
//!
//! Each operation lives in its own submodule; this module holds the dialect
//! registration and the region/state helpers shared between them.

use std::collections::BTreeSet;
use std::fmt;

use gnx_common::{ArrayInfo, DType, Shape};
use gnx_expr::lower::{FixedRegion, LoweringContext, RegionCell, StructuredRegion};
use gnx_expr::{
    Eqn, Expr, HEqn, HExpr, HParam, Kind, LoweringError, ObjectTracer, Op, Operation, Var, VarInfo,
};

mod branch;
mod call;
mod scan;
mod scan_graph;
pub use scan_graph::{GraphScan, scan_graph};
mod while_loop;

pub(crate) use branch::branch_op;
pub use branch::{CondOperation, SwitchOperation, cond, cond_op, switch, switch_op};
pub use call::{
    CallOperation, call, call_immutable, call_lowered, call_op, defer_call, defer_call_with_symbols,
};
pub use scan::{ScanOperation, ScanParameters, scan, scan_op};
pub use while_loop::{WhileOperation, while_array, while_loop, while_op};

gnx_expr::dialect! {
    dyn Operation, "cf" => {
        "call" => CallOperation,
        "cond" => CondOperation,
        "switch" => SwitchOperation,
        "scan" => ScanOperation,
        "while_loop" => WhileOperation,
    }
}

/// A count in the form it is stored in an operation.
fn stored(count: usize) -> u64 {
    count as u64
}

/// A count stored in an operation.
fn count(stored: rkyv::rend::u64_le) -> usize {
    usize::try_from(stored.to_native()).expect("count exceeds usize::MAX")
}

type RegionResult<T> = std::result::Result<T, LoweringError>;

fn signature(message: &'static str) -> LoweringError {
    LoweringError::RegionSignature(message)
}

fn add_state(current: usize, new_slots: usize) -> RegionResult<usize> {
    current
        .checked_add(new_slots)
        .ok_or_else(|| signature("state count overflow"))
}

fn check_state_len(state_len: usize, inputs: usize, outputs: usize) -> RegionResult<()> {
    if state_len > inputs.min(outputs) {
        return Err(signature("invalid control-flow state layout"));
    }
    Ok(())
}

fn fmt_control_flow(f: &mut fmt::Formatter<'_>, name: &str, state_len: usize) -> fmt::Result {
    write!(f, "cf::{name}")?;
    if state_len == 0 {
        Ok(())
    } else {
        write!(f, "<state: {state_len}>")
    }
}

// ---------------------------------------------------------------------------
// Tracing helpers
// ---------------------------------------------------------------------------

fn scalar(dtype: DType) -> VarInfo {
    VarInfo::new(ArrayInfo::new(Shape::fixed([]), dtype))
}

pub(crate) fn value_results(body: HExpr<'_>) -> RegionResult<Vec<VarInfo>> {
    body.outputs()
        .iter()
        .map(|&var| match body.definition(var) {
            Some(param) if param.kind() == Kind::Value => Ok(param.info_ref().to_owned()),
            _ => Err(signature(
                "control flow must return values, not cell handles",
            )),
        })
        .collect()
}

pub(crate) fn expect_inputs(
    body: HExpr<'_>,
    expected: impl IntoIterator<Item = (Kind, VarInfo)>,
) -> RegionResult<()> {
    if body
        .inputs()
        .map(|param| (param.kind(), param.info_ref().to_owned()))
        .eq(expected)
    {
        Ok(())
    } else {
        Err(signature("region input signature"))
    }
}

fn object_kind(object: &ObjectTracer) -> (Kind, VarInfo) {
    match object {
        ObjectTracer::Ref(trace) => (Kind::Value, trace.info().clone()),
        ObjectTracer::Cell(cell) => (Kind::Cell, cell.type_info().clone()),
    }
}

// ---------------------------------------------------------------------------
// Lowering helpers
// ---------------------------------------------------------------------------

pub(crate) fn value_infos<'a>(
    outputs: impl Iterator<Item = HParam<'a>>,
) -> RegionResult<Vec<VarInfo>> {
    outputs
        .map(|param| match param.kind() {
            Kind::Value => Ok(param.info_ref().to_owned()),
            Kind::Cell => Err(LoweringError::ExpectedValue(param.var())),
        })
        .collect()
}

/// Cells written by any region, threaded through the lowered op as extra
/// carried state slots.
struct StateLayout {
    cells: Vec<RegionCell>,
    slots: Vec<(RegionCell, VarInfo)>,
    initial: Vec<Var>,
}

impl StateLayout {
    fn new(ctx: &LoweringContext<'_>, regions: &[StructuredRegion]) -> RegionResult<Self> {
        let cells: Vec<_> = regions
            .iter()
            .flat_map(StructuredRegion::writes)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let mut slots = Vec::new();
        let mut initial = Vec::new();
        for &cell in &cells {
            let (value, info) = ctx.cell_state(cell)?;
            initial.push(value);
            slots.push((cell, info));
        }
        Ok(Self {
            cells,
            slots,
            initial,
        })
    }

    fn emit(
        self,
        ctx: &mut LoweringContext<'_>,
        eqn: HEqn<'_>,
        op: Op,
        mut inputs: Vec<Var>,
        positions: (usize, usize),
        regions: &[FixedRegion],
    ) -> RegionResult<()> {
        let (input_position, output_position) = positions;
        inputs.splice(input_position..input_position, self.initial);
        let mut infos = value_infos(eqn.outputs())?;
        infos.splice(
            output_position..output_position,
            self.slots.into_iter().map(|(_, info)| info),
        );
        let output = ctx.allocate_results(&infos);
        ctx.bind_structured_results(eqn.outputs(), &self.cells, output_position, &output)?;
        ctx.emit(
            op,
            inputs,
            output,
            regions.iter().map(FixedRegion::subexpr).collect(),
        );
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Expression checking helpers
// ---------------------------------------------------------------------------

fn parameters<'a>(expr: Expr<'a>) -> impl Iterator<Item = VarInfo> + 'a {
    expr.inputs().map(|param| param.info_ref().to_owned())
}

fn check_child(
    expr: Expr<'_>,
    inputs: impl IntoIterator<Item = VarInfo>,
    outputs: &[VarInfo],
) -> RegionResult<()> {
    if !parameters(expr).eq(inputs)
        || !expr
            .outputs()
            .iter()
            .map(|&var| expr.value_info(var).map(|info| info.to_owned()))
            .eq(outputs.iter().cloned().map(Some))
    {
        return Err(signature("fixed region signature"));
    }
    Ok(())
}

fn expr_output_infos(eqn: Eqn<'_>) -> Vec<VarInfo> {
    eqn.outputs()
        .map(|param| param.info_ref().to_owned())
        .collect()
}

//! Trace and lower the `cf` dialect's control-flow operations.
//!
//! Each operation lives in its own submodule; this module holds the dialect
//! registration and the region/state helpers shared between them.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;

use gnx_array::{ArrayInfo, DType, Shape};
use gnx_expr::lower::{FixedRegion, LoweringContext, RegionCell, StructuredRegion};
use gnx_expr::{
    Capture, CellAccess, CellKey, DeferredApplyPlan, DeferredLocal, DeferredProjection,
    DeferredRegion, Dialect, Eqn, Expr, HEqn, HExpr, Invocation, Kind, LoweringError, ObjectTracer,
    Op, Trace, ValueInfo, Var,
};

mod branch;
mod call;
mod scan;
mod while_loop;

pub(crate) use branch::branch_op;
pub use branch::{CondOperation, SwitchOperation, cond, cond_op, switch, switch_op};
pub use call::{CallOperation, call, call_immutable, call_lowered, call_op, defer_call};
pub use scan::{ScanOperation, ScanParameters, scan, scan_op};
pub use while_loop::{WhileOperation, while_loop, while_op};

pub struct ControlFlowDialect;

static CONTROL_FLOW: ControlFlowDialect = ControlFlowDialect;

impl Dialect for ControlFlowDialect {
    fn name(&self) -> &'static str {
        "cf"
    }
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

fn scalar(dtype: DType) -> ValueInfo {
    ValueInfo::new(ArrayInfo::new(Shape::fixed([]), dtype))
}

pub(crate) fn value_results(body: &HExpr) -> RegionResult<Vec<ValueInfo>> {
    body.outputs()
        .iter()
        .map(|var| match body.kind(var) {
            Some(Kind::Value(info)) => Ok(info.clone()),
            _ => Err(signature(
                "control flow must return values, not cell handles",
            )),
        })
        .collect()
}

pub(crate) fn expect_inputs(
    body: &HExpr,
    expected: impl IntoIterator<Item = Kind>,
) -> RegionResult<()> {
    if body
        .inputs()
        .iter()
        .map(|(_, kind)| kind.clone())
        .eq(expected)
    {
        Ok(())
    } else {
        Err(signature("region input signature"))
    }
}

fn object_kind(object: &ObjectTracer) -> Kind {
    match object {
        ObjectTracer::Ref(trace) => Kind::Value(trace.info().clone()),
        ObjectTracer::Cell(cell) => Kind::Cell(cell.type_info().clone()),
    }
}

pub(crate) struct BoundRegion<'a> {
    pub(crate) capture: &'a Capture,
    pub(crate) inputs: &'a [ObjectTracer],
    pub(crate) read_only: bool,
}

pub(crate) fn defer_regions(
    op: Op,
    regions: Vec<BoundRegion<'_>>,
    inputs: Vec<ObjectTracer>,
    infos: Vec<ValueInfo>,
) -> RegionResult<Vec<Trace>> {
    let mut locals = BTreeMap::<CellKey, DeferredLocal>::new();
    let mut external_access = CellAccess::Unused;
    for bound in &regions {
        let capture = bound.capture;
        let body = capture.expr();
        for ((var, _), object) in body
            .closure()
            .iter()
            .zip(capture.closure())
            .chain(body.inputs().iter().zip(bound.inputs))
        {
            if let ObjectTracer::Cell(cell) = object {
                let access = body.cell_access(var).unwrap_or(CellAccess::Unused);
                let writes = matches!(access, CellAccess::Write | CellAccess::ReadWrite);
                if bound.read_only && writes {
                    return Err(signature("while predicate cannot write a captured cell"));
                }
                if cell.is_maybe_aliased() {
                    external_access = external_access.combine(access);
                } else {
                    locals
                        .entry(CellKey::from(cell))
                        .or_insert_with(|| DeferredLocal {
                            cell: cell.clone(),
                            written: false,
                        })
                        .written |= writes;
                }
            }
        }
    }
    Ok(Invocation::defer_apply(
        op,
        DeferredApplyPlan {
            regions: regions
                .into_iter()
                .map(|region| DeferredRegion {
                    body: Arc::new(region.capture.expr().clone()),
                    captures: region.capture.closure().to_vec(),
                })
                .collect(),
            inputs,
            projections: (0..infos.len()).map(DeferredProjection::Value).collect(),
            outputs: infos.into_iter().map(Kind::Value).collect(),
            locals: locals.into_values().collect(),
            external_access,
        },
    ))
}

// ---------------------------------------------------------------------------
// Lowering helpers
// ---------------------------------------------------------------------------

fn value_infos(outputs: &[(Var, Kind)]) -> RegionResult<Vec<ValueInfo>> {
    outputs
        .iter()
        .map(|(var, kind)| match kind {
            Kind::Value(info) => Ok(info.clone()),
            Kind::Cell(_) => Err(LoweringError::ExpectedValue(*var)),
        })
        .collect()
}

/// Cells written by any region, threaded through the lowered op as extra
/// carried state slots.
struct StateLayout {
    cells: Vec<RegionCell>,
    slots: Vec<(RegionCell, ValueInfo)>,
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
        eqn: &HEqn,
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

fn parameters(expr: &Expr) -> impl Iterator<Item = &ValueInfo> {
    expr.inputs().iter().map(|(_, info)| info)
}

fn check_child(
    expr: &Expr,
    inputs: impl IntoIterator<Item = ValueInfo>,
    outputs: &[ValueInfo],
) -> RegionResult<()> {
    if !parameters(expr).cloned().eq(inputs)
        || !expr
            .outputs()
            .iter()
            .map(|var| expr.value_info(var))
            .eq(outputs.iter().map(Some))
    {
        return Err(signature("fixed region signature"));
    }
    Ok(())
}

fn expr_output_infos(eqn: &Eqn) -> Vec<ValueInfo> {
    eqn.outputs().iter().map(|(_, info)| info.clone()).collect()
}

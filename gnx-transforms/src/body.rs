//! Shared machinery for function-transformation region operations (`map.map`,
//! `ad.jvp`, `ad.vjp`): one value-only body that may read, but not write,
//! captured cells.

use std::collections::BTreeMap;
use std::sync::Arc;

use gnx_expr::lower::{LoweringContext, RegionArgument};
use gnx_expr::{
    Capture, CellAccess, CellKey, DeferredApplyPlan, DeferredLocal, DeferredProjection,
    DeferredRegion, Eqn, Expr, HEqn, HExpr, Invocation, Kind, LoweringError, ObjectTracer, Op,
    Trace, ValueInfo, Var,
};

use crate::cf::value_results;
use crate::util;

/// A traced body and the values bound to its closure.
#[derive(Clone)]
pub struct Body {
    pub expr: Arc<HExpr>,
    pub captures: Vec<ObjectTracer>,
}

impl Body {
    pub fn new(expr: Arc<HExpr>, captures: Vec<ObjectTracer>) -> Self {
        assert_eq!(expr.closure().len(), captures.len(), "body capture arity");
        Self { expr, captures }
    }
}

impl From<&Capture> for Body {
    fn from(capture: &Capture) -> Self {
        Self::new(Arc::new(capture.expr().clone()), capture.closure().to_vec())
    }
}

pub(crate) fn signature(message: &'static str) -> LoweringError {
    LoweringError::RegionSignature(message)
}

pub(crate) fn strictly_increasing(indices: &[usize], bound: usize) -> Result<(), LoweringError> {
    if indices.windows(2).any(|pair| pair[0] >= pair[1]) || indices.iter().any(|&i| i >= bound) {
        return Err(signature(
            "transformation indices must be strictly increasing and in range",
        ));
    }
    Ok(())
}

/// Value metadata of a body's inputs and results.
pub struct BodySignature {
    pub inputs: Vec<ValueInfo>,
    pub outputs: Vec<ValueInfo>,
}

impl BodySignature {
    pub fn of_expr(body: &Expr) -> Result<Self, LoweringError> {
        Ok(Self {
            inputs: body.inputs().iter().map(|(_, info)| info.clone()).collect(),
            outputs: body
                .outputs()
                .iter()
                .map(|var| {
                    body.value_info(var)
                        .cloned()
                        .ok_or(LoweringError::UndefinedVariable(*var))
                })
                .collect::<Result<_, _>>()?,
        })
    }

    pub fn of_hexpr(body: &HExpr) -> Result<Self, LoweringError> {
        Ok(Self {
            inputs: body
                .inputs()
                .iter()
                .map(|(var, kind, info)| match kind {
                    Kind::Value => Ok(info.clone()),
                    Kind::Cell => Err(LoweringError::ExpectedValue(*var)),
                })
                .collect::<Result<_, _>>()?,
            outputs: value_results(body)?,
        })
    }
}

fn value_infos(outputs: &[(Var, Kind, ValueInfo)]) -> Result<Vec<ValueInfo>, LoweringError> {
    outputs
        .iter()
        .map(|(var, kind, info)| match kind {
            Kind::Value => Ok(info.clone()),
            Kind::Cell => Err(LoweringError::ExpectedValue(*var)),
        })
        .collect()
}

/// `Operation::lower` for a transformation: lower the body as a value-only
/// region (rejecting cell writes) and emit the operation unchanged. `check`
/// validates the operand and result metadata against the body.
pub(crate) fn lower_transform(
    eqn: &HEqn,
    ctx: &mut LoweringContext<'_>,
    check: impl FnOnce(&BodySignature, &[ValueInfo], &[ValueInfo]) -> Result<(), LoweringError>,
) -> Result<(), LoweringError> {
    let [sub] = eqn.subexprs() else {
        return Err(signature("transformations take exactly one body"));
    };
    let body = BodySignature::of_hexpr(sub.expr())?;
    let (values, infos): (Vec<_>, Vec<_>) = eqn
        .inputs()
        .iter()
        .map(|&var| ctx.value_operand(var))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .unzip();
    let outputs = value_infos(eqn.outputs())?;
    check(&body, &infos, &outputs)?;
    let region = ctx.lower_structured_region(
        sub,
        &vec![RegionArgument::Value; body.inputs.len()],
        &body.inputs,
    )?;
    if region.writes().next().is_some() {
        return Err(signature(
            "transformed functions cannot write captured cells",
        ));
    }
    let fixed = region.fix(ctx, &[], body.inputs.len(), None, &body.outputs)?;
    let physical = ctx.allocate_results(&outputs);
    ctx.bind_structured_results(eqn.outputs(), &[], 0, &physical)?;
    ctx.emit(eqn.op().clone(), values, physical, vec![fixed.subexpr()]);
    Ok(())
}

/// `Operation::check_expr` for a transformation with one body.
pub(crate) fn check_expr_with(
    eqn: &Eqn,
    check: impl FnOnce(&BodySignature, usize, &[ValueInfo]) -> Result<(), LoweringError>,
) -> Result<(), LoweringError> {
    let [sub] = eqn.subexprs() else {
        return Err(signature("transformations take exactly one body"));
    };
    check(
        &BodySignature::of_expr(sub.expr())?,
        eqn.inputs().len(),
        &util::output_infos(eqn),
    )
}

pub(crate) fn reject_cell_writes(body: &Body) -> Result<(), LoweringError> {
    let hexpr = &body.expr;
    let writes = hexpr
        .closure()
        .iter()
        .chain(hexpr.inputs())
        .filter_map(|(var, ..)| hexpr.cell_access(var))
        .any(|access| matches!(access, CellAccess::Write | CellAccess::ReadWrite));
    if writes {
        return Err(signature(
            "transformed functions cannot write captured cells",
        ));
    }
    Ok(())
}

/// Record a transformation of a read-only body with value operands. Captured
/// cells are read at the call (`reject_cell_writes` must hold).
pub(crate) fn record(op: Op, body: &Body, inputs: &[Trace], outputs: Vec<ValueInfo>) -> Vec<Trace> {
    let mut locals = BTreeMap::<CellKey, DeferredLocal>::new();
    let mut external_access = CellAccess::Unused;
    for ((var, ..), object) in body.expr.closure().iter().zip(&body.captures) {
        if let ObjectTracer::Cell(cell) = object {
            let access = body.expr.cell_access(var).unwrap_or(CellAccess::Unused);
            if cell.is_maybe_aliased() {
                external_access = external_access.combine(access);
            } else {
                locals
                    .entry(CellKey::from(cell))
                    .or_insert_with(|| DeferredLocal {
                        cell: cell.clone(),
                        written: false,
                    });
            }
        }
    }
    Invocation::defer_apply(
        op,
        DeferredApplyPlan {
            regions: vec![DeferredRegion {
                symbol_captures: body.expr.symbol_closure().iter().map(|(s, _)| *s).collect(),
                body: body.expr.clone(),
                captures: body.captures.clone(),
            }],
            inputs: inputs.iter().cloned().map(ObjectTracer::from).collect(),
            projections: (0..outputs.len()).map(DeferredProjection::Value).collect(),
            outputs: outputs.into_iter().map(|info| (Kind::Value, info)).collect(),
            locals: locals.into_values().collect(),
            external_access,
        },
    )
}

pub(crate) fn trace_infos(traces: &[Trace]) -> Vec<ValueInfo> {
    traces.iter().map(|trace| trace.info().clone()).collect()
}

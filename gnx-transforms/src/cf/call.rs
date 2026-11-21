//! `cf.call`: invoke a captured function body.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use gnx_expr::lower::LoweringContext;
use gnx_expr::{
    CallError, Capture, CellAccess, CellKey, CellTrace, DeferredApplyPlan, DeferredLocal,
    DeferredProjection, DeferredRegion, Dialect, Eqn, HEqn, HExpr, Invocation, Kind, LoweredExpr,
    LoweringError, ObjectTracer, Op, Operation, Trace, TraceInput, ValueInfo, Var,
};

use super::CONTROL_FLOW;

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct CallOperation;

impl fmt::Display for CallOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("cf::call")
    }
}

impl Operation for CallOperation {
    fn name(&self) -> &'static str {
        "call"
    }

    fn dialect(&self) -> &'static dyn Dialect {
        &CONTROL_FLOW
    }

    fn cell_access(&self, eqn: &HEqn, cell: Var) -> CellAccess {
        let explicit = eqn.subexprs().first().map_or(CellAccess::Unused, |body| {
            body.input_cell_access(eqn.inputs(), cell)
        });
        eqn.captured_cell_access(cell).combine(explicit)
    }

    fn lower(&self, eqn: &HEqn, ctx: &mut LoweringContext<'_>) -> Result<(), LoweringError> {
        let [sub] = eqn.subexprs() else {
            return Err(LoweringError::CallArity);
        };
        let body = sub.expr();
        if sub.closure().len() != body.closure().len()
            || eqn.inputs().len() != body.inputs().len()
            || eqn.outputs().len() != body.outputs().len()
        {
            return Err(LoweringError::CallArity);
        }
        let region = ctx.lower_region(sub, eqn.inputs())?;
        let physical = ctx.allocate_results(region.output_infos());
        ctx.bind_region_results(eqn.outputs(), &region, &physical)?;
        ctx.apply_region_updates(&region, &physical)?;
        if !region.program().expr.eqns().is_empty() || !physical.is_empty() {
            ctx.emit(
                eqn.op().clone(),
                region.inputs(),
                physical,
                vec![region.subexpr()],
            );
        }
        Ok(())
    }

    fn check_expr(&self, eqn: &Eqn) -> Result<(), LoweringError> {
        let [sub] = eqn.subexprs() else {
            return Err(LoweringError::CallArity);
        };
        if eqn.inputs().len() != sub.expr().inputs().len()
            || eqn.outputs().len() != sub.expr().outputs().len()
        {
            return Err(LoweringError::CallArity);
        }
        Ok(())
    }
}

pub fn call_op() -> Op {
    Op::new(CallOperation)
}

/// Bind a captured function without specializing potentially aliased cells.
pub fn call(capture: &Capture, inputs: &[ObjectTracer]) -> Vec<ObjectTracer> {
    defer_call(Arc::new(capture.expr().clone()), capture.closure(), inputs)
}

/// Record a call with a value-only boundary in the trace.
pub fn call_immutable<C, I>(
    body: Arc<HExpr>,
    captures: C,
    inputs: I,
    outputs: Vec<ValueInfo>,
) -> Vec<Trace>
where
    C: IntoIterator,
    C::Item: Into<TraceInput>,
    I: IntoIterator,
    I::Item: Into<TraceInput>,
{
    assert!(
        body.closure()
            .iter()
            .chain(body.inputs())
            .all(|(_, kind, _)| matches!(kind, Kind::Value))
    );
    assert!(
        body.outputs()
            .iter()
            .all(|var| matches!(body.kind(var), Some(Kind::Value)))
    );
    let captures: Vec<_> = captures.into_iter().map(Into::into).collect();
    let inputs: Vec<_> = inputs.into_iter().map(Into::into).collect();
    assert_eq!(
        inputs.len(),
        body.inputs().len(),
        "call input arity mismatch"
    );
    assert_eq!(
        outputs.len(),
        body.outputs().len(),
        "call output arity mismatch"
    );
    Invocation::invoke(call_op(), vec![(body, captures)], inputs, outputs)
        .into_iter()
        .map(|tracer| tracer.into_trace_ref())
        .collect()
}

/// Replay a specialization with arguments having the same alias pattern.
pub fn call_lowered(
    body: &LoweredExpr,
    captures: &[ObjectTracer],
    inputs: &[ObjectTracer],
) -> Result<Vec<ObjectTracer>, CallError> {
    body.replay(captures, inputs, call_immutable)
}

/// Bind a high-level body and explicit captures without lowering the call.
/// The caller's alias specialization is applied later, during lowering.
pub fn defer_call(
    body: Arc<HExpr>,
    captures: &[ObjectTracer],
    inputs: &[ObjectTracer],
) -> Vec<ObjectTracer> {
    let symbol_captures = body.symbol_closure().iter().map(|v| v.0).collect();
    defer_call_with_symbols(body, symbol_captures, captures, inputs)
}
/// Record a call binding the child’s symbol closure explicitly in caller order.
pub fn defer_call_with_symbols(
    body: Arc<HExpr>,
    symbol_captures: Vec<gnx_expr::Symbol>,
    captures: &[ObjectTracer],
    inputs: &[ObjectTracer],
) -> Vec<ObjectTracer> {
    let metas = |args: &[ObjectTracer]| {
        args.iter()
            .map(|obj| match obj {
                ObjectTracer::Ref(v) => v.info().clone(),
                ObjectTracer::Cell(c) => c.type_info().clone(),
            })
            .collect::<Vec<_>>()
    };
    let symbols = body
        .bind_symbols(&symbol_captures, &metas(captures), &metas(inputs), |s| {
            gnx_expr::TraceContext::current()?.symbol_info(s)
        })
        .expect("invalid call symbol bindings");
    assert_eq!(captures.len(), body.closure().len());
    assert_eq!(inputs.len(), body.inputs().len());
    let bindings: BTreeMap<_, _> = body
        .closure()
        .iter()
        .chain(body.inputs())
        .map(|(var, ..)| *var)
        .zip(captures.iter().chain(inputs))
        .collect();
    let mut locals = BTreeMap::<CellKey, DeferredLocal>::new();
    let mut external_access = CellAccess::Unused;
    for (var, object) in &bindings {
        if let ObjectTracer::Cell(cell) = object {
            let access = body.cell_access(var).unwrap_or(CellAccess::Unused);
            if cell.is_maybe_aliased() {
                external_access = external_access.combine(access);
            } else {
                let local = locals
                    .entry(CellKey::from(cell))
                    .or_insert_with(|| DeferredLocal {
                        cell: cell.clone(),
                        written: false,
                    });
                local.written |= matches!(access, CellAccess::Write | CellAccess::ReadWrite);
            }
        }
    }

    enum ResultSlot<'a> {
        Passthrough(&'a ObjectTracer),
        Value(usize),
        FreshCell(usize),
    }
    let outputs: Vec<_> = body
        .outputs()
        .iter()
        .map(|var| {
            body.definition(var)
                .map(|(kind, info)| (kind, info.remap_symbols(|s| symbols[&s])))
                .expect("undefined deferred call output")
        })
        .collect();
    let mut projections = Vec::new();
    let mut fresh = BTreeMap::new();
    let slots: Vec<_> = body
        .outputs()
        .iter()
        .zip(&outputs)
        .enumerate()
        .map(|(index, (var, (kind, _)))| {
            if let Some(&binding) = bindings.get(var) {
                return ResultSlot::Passthrough(binding);
            }
            match kind {
                Kind::Value => {
                    let slot = projections.len();
                    projections.push(DeferredProjection::Value(index));
                    ResultSlot::Value(slot)
                }
                Kind::Cell => ResultSlot::FreshCell(*fresh.entry(*var).or_insert_with(|| {
                    let slot = projections.len();
                    projections.push(DeferredProjection::CellContents(index));
                    slot
                })),
            }
        })
        .collect();
    let results = Invocation::defer_apply(
        call_op(),
        DeferredApplyPlan {
            regions: vec![DeferredRegion {
                symbol_captures,
                body,
                captures: captures.to_vec(),
            }],
            inputs: inputs.to_vec(),
            outputs,
            projections,
            locals: locals.into_values().collect(),
            external_access,
        },
    );
    let mut fresh = BTreeMap::new();
    slots
        .into_iter()
        .map(|slot| match slot {
            ResultSlot::Passthrough(value) => value.clone(),
            ResultSlot::Value(index) => results[index].clone().into(),
            ResultSlot::FreshCell(index) => fresh
                .entry(index)
                .or_insert_with(|| CellTrace::new(results[index].clone()))
                .clone()
                .into(),
        })
        .collect()
}

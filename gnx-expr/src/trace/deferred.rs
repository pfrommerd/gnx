use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, OnceLock, Weak};

use crate::{CellAccess, HExpr, Kind, Op};

use super::{
    CellKey, CellTrace, ContextID, Invocation, ObjectTracer, Trace, TraceContext, TraceData,
    ValueInfo,
};

/// A region and its bindings in a deferred application. How explicit operands
/// are passed to the region is defined by the operation.
pub struct DeferredRegion {
    pub body: Arc<HExpr>,
    pub captures: Vec<ObjectTracer>,
}

/// A trace result projected from a high-level equation output.
pub enum DeferredProjection {
    Value(usize),
    CellContents(usize),
}

/// A known caller-local cell to snapshot before recording the application.
pub struct DeferredLocal {
    pub cell: CellTrace,
    /// Whether the application may change the cell's contents.
    pub written: bool,
}

/// The operation supplies its effects and layout; the trace installs the node
/// and manages ordering, snapshots, and result slots.
pub struct DeferredApplyPlan {
    pub regions: Vec<DeferredRegion>,
    pub inputs: Vec<ObjectTracer>,
    pub outputs: Vec<Kind>,
    pub projections: Vec<DeferredProjection>,
    /// List each caller-local cell used as an input or capture once.
    pub locals: Vec<DeferredLocal>,
    pub external_access: CellAccess,
}

/// A region application awaiting the caller's alias specialization.
pub struct DeferredApply {
    pub(super) context_id: ContextID,
    pub(super) op: Op,
    pub(super) regions: Vec<DeferredRegion>,
    pub(super) inputs: Vec<ObjectTracer>,
    pub(super) kinds: Vec<Kind>,
    pub(super) results: Vec<Projection>,
    pub(super) value_parents: Vec<Trace>,
    pub(super) aliasing_parents: Vec<Arc<Invocation>>,
    pub(super) outputs: Vec<ValueInfo>,
    pub(super) result_slots: OnceLock<Box<[Weak<TraceData>]>>,
}

impl DeferredApply {
    pub fn op(&self) -> &Op {
        &self.op
    }
    pub fn regions(&self) -> &[DeferredRegion] {
        &self.regions
    }
    pub fn inputs(&self) -> &[ObjectTracer] {
        &self.inputs
    }
}

pub(super) enum Projection {
    Output(DeferredProjection),
    LocalUpdate(CellTrace),
}

impl Invocation {
    /// Record a region application without resolving aliases of external cells.
    /// The returned traces follow `plan.projections`; local updates are installed
    /// before returning and are not included in the result list.
    pub fn defer_apply(op: Op, plan: DeferredApplyPlan) -> Vec<Trace> {
        let context = TraceContext::current();
        let mut value_parents = Vec::new();
        let mut snapshots = BTreeMap::new();
        for local in &plan.locals {
            assert!(!local.cell.is_maybe_aliased());
            let mut initial = local.cell.read_home();
            initial.make_internal();
            value_parents.push(initial.internal_clone());
            assert!(
                snapshots
                    .insert(CellKey::from(&local.cell), CellTrace::new(initial))
                    .is_none()
            );
        }
        let mut used_snapshots = BTreeSet::new();
        let mut bind = |operand| match operand {
            ObjectTracer::Ref(value) => {
                value_parents.push(value.internal_clone());
                ObjectTracer::Ref(value.internal_clone())
            }
            ObjectTracer::Cell(cell) => {
                if cell.is_maybe_aliased() {
                    return ObjectTracer::Cell(cell);
                }
                let key = CellKey::from(&cell);
                used_snapshots.insert(key);
                ObjectTracer::Cell(
                    snapshots
                        .get(&key)
                        .expect("missing caller-local snapshot")
                        .clone(),
                )
            }
        };
        let regions = plan
            .regions
            .into_iter()
            .map(|region| {
                assert_eq!(region.body.closure().len(), region.captures.len());
                DeferredRegion {
                    body: region.body,
                    captures: region.captures.into_iter().map(&mut bind).collect(),
                }
            })
            .collect();
        let inputs = plan.inputs.into_iter().map(bind).collect();
        assert_eq!(
            used_snapshots.len(),
            snapshots.len(),
            "unused caller-local snapshot"
        );

        let mut infos = Vec::new();
        let mut results: Vec<_> = plan
            .projections
            .into_iter()
            .map(|projection| {
                let info = match (&projection, &plan.outputs) {
                    (DeferredProjection::Value(index), kinds) => match &kinds[*index] {
                        Kind::Value(info) => info,
                        _ => panic!("expected a value projection"),
                    },
                    (DeferredProjection::CellContents(index), kinds) => match &kinds[*index] {
                        Kind::Cell(info) => info,
                        _ => panic!("expected a cell projection"),
                    },
                };
                infos.push(info.clone());
                Projection::Output(projection)
            })
            .collect();
        let returned_count = results.len();
        let mut writebacks = Vec::new();
        for local in plan.locals {
            if local.written {
                let snapshot = snapshots[&CellKey::from(&local.cell)].clone();
                writebacks.push((local.cell, results.len()));
                infos.push(snapshot.type_info().clone());
                results.push(Projection::LocalUpdate(snapshot));
            }
        }
        let external_write = matches!(
            plan.external_access,
            CellAccess::Write | CellAccess::ReadWrite
        );
        let invocation = Arc::new(Self::DeferredApply(DeferredApply {
            context_id: TraceContext::current_id(),
            op,
            regions,
            inputs,
            kinds: plan.outputs,
            results,
            value_parents,
            aliasing_parents: context
                .as_ref()
                .filter(|_| plan.external_access != CellAccess::Unused)
                .map(|ctx| ctx.aliasing_parents(external_write))
                .unwrap_or_default(),
            outputs: infos,
            result_slots: OnceLock::new(),
        }));
        let mut returned = Self::returned_values(invocation.clone());
        if let Some(context) = context {
            if external_write {
                context.note_aliased_write(invocation);
            } else if plan.external_access == CellAccess::Read {
                context.note_aliased_read(invocation);
            }
        }
        for (cell, slot) in writebacks {
            cell.set(returned[slot].clone());
        }
        returned.truncate(returned_count);
        returned
    }
}

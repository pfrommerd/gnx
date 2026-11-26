use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, OnceLock, Weak};

use crate::{CellAccess, HExprHandle, Kind, Op, Symbol};

use super::{
    CellKey, CellTrace, ContextID, Invocation, ObjectTracer, Trace, TraceContext, TraceData,
    VarInfo,
};

/// A region of an application and its bindings. How explicit operands are
/// passed to the region is defined by the operation.
#[derive(Clone)]
pub struct ApplyRegion {
    pub symbol_captures: Vec<Symbol>,
    pub body: HExprHandle,
    pub captures: Vec<ObjectTracer>,
    /// The objects the operation binds to the body's explicit inputs, in
    /// order. They only determine how the body accesses cell operands, so an
    /// operation passing values alone leaves this empty. Not retained by the
    /// recorded node.
    pub arguments: Vec<ObjectTracer>,
}

/// A trace result projected from a high-level equation output.
pub enum ApplyProjection {
    Value(usize),
    CellContents(usize),
}

/// A known caller-local cell to snapshot before recording the application.
pub struct ApplyLocal {
    pub cell: CellTrace,
    /// Whether the application may change the cell's contents.
    pub written: bool,
}

/// The operation supplies its effects and layout; the trace installs the node
/// and manages ordering, snapshots, and result slots.
pub struct ApplyPlan {
    pub regions: Vec<ApplyRegion>,
    pub inputs: Vec<ObjectTracer>,
    pub outputs: Vec<(Kind, VarInfo)>,
    pub projections: Vec<ApplyProjection>,
    /// List each caller-local cell used as an input or capture once.
    pub locals: Vec<ApplyLocal>,
    pub external_access: CellAccess,
}

/// An operation applied to operands and regions. Aliases of external cells
/// are specialized by the caller, during lowering.
pub struct Apply {
    pub(super) context_id: ContextID,
    pub(super) op: Op,
    pub(super) regions: Vec<ApplyRegion>,
    pub(super) inputs: Vec<ObjectTracer>,
    pub(super) kinds: Vec<(Kind, VarInfo)>,
    pub(super) results: Vec<Projection>,
    pub(super) value_parents: Vec<Trace>,
    pub(super) aliasing_parents: Vec<Arc<Invocation>>,
    pub(super) outputs: Vec<VarInfo>,
    pub(super) result_slots: OnceLock<Box<[Weak<TraceData>]>>,
}

impl Apply {
    pub fn op(&self) -> &Op {
        &self.op
    }
    pub fn regions(&self) -> &[ApplyRegion] {
        &self.regions
    }
    pub fn inputs(&self) -> &[ObjectTracer] {
        &self.inputs
    }
}

pub(super) enum Projection {
    Output(ApplyProjection),
    LocalUpdate(CellTrace),
}

impl ApplyRegion {
    /// A body with the values and cells bound to its closure. Its closure
    /// symbols are provided under their own names.
    ///
    /// # Panics
    /// If there is not one capture per closure input of the body.
    pub fn new(body: HExprHandle, captures: Vec<ObjectTracer>) -> Self {
        let expr = body.view();
        assert_eq!(
            expr.closure_inputs().len(),
            captures.len(),
            "body capture arity"
        );
        Self {
            symbol_captures: expr.closure_symbols().to_vec(),
            body,
            captures,
            arguments: Vec::new(),
        }
    }

    /// Bind the body's explicit inputs to `arguments`, which may include cells.
    pub fn with_arguments(mut self, arguments: impl IntoIterator<Item = ObjectTracer>) -> Self {
        self.arguments = arguments.into_iter().collect();
        self
    }
}

impl From<&super::Capture> for ApplyRegion {
    fn from(capture: &super::Capture) -> Self {
        Self::new(capture.root().clone().handle(), capture.closure().to_vec())
    }
}

impl ApplyPlan {
    /// How `regions` access the cells bound to their closures and arguments:
    /// the caller-local cells, each listed once, and the combined access to
    /// cells that may be aliased.
    pub fn classify(regions: &[ApplyRegion]) -> (Vec<ApplyLocal>, CellAccess) {
        // Cell identities determine aliasing, not the order of the update
        // results. Use parameter order so a fresh decoder state reuses the IR.
        let mut indices = BTreeMap::<CellKey, usize>::new();
        let mut locals = Vec::new();
        let mut external_access = CellAccess::Unused;
        for region in regions {
            let body = region.body.view();
            for (param, object) in (body.closure_inputs().zip(&region.captures))
                .chain(body.inputs().zip(&region.arguments))
            {
                let ObjectTracer::Cell(cell) = object else {
                    continue;
                };
                let access = body.cell_access(param.var()).unwrap_or(CellAccess::Unused);
                if cell.is_maybe_aliased() {
                    external_access = external_access.combine(access);
                } else {
                    let i = *indices.entry(CellKey::from(cell)).or_insert_with(|| {
                        let i = locals.len();
                        locals.push(ApplyLocal {
                            cell: cell.clone(),
                            written: false,
                        });
                        i
                    });
                    locals[i].written |=
                        matches!(access, CellAccess::Write | CellAccess::ReadWrite);
                }
            }
        }
        (locals, external_access)
    }
}

impl Invocation {
    /// Record an application of `op` with value results. Regions may read and
    /// write the cells bound to their closures and arguments; caller-local
    /// cells are updated before returning.
    pub fn apply(
        op: Op,
        regions: Vec<ApplyRegion>,
        inputs: Vec<ObjectTracer>,
        outputs: Vec<VarInfo>,
    ) -> Vec<Trace> {
        let (locals, external_access) = ApplyPlan::classify(&regions);
        Self::apply_plan(
            op,
            ApplyPlan {
                regions,
                inputs,
                projections: (0..outputs.len()).map(ApplyProjection::Value).collect(),
                outputs: outputs
                    .into_iter()
                    .map(|info| (Kind::Value, info))
                    .collect(),
                locals,
                external_access,
            },
        )
    }

    /// Record an application without resolving aliases of external cells.
    /// The returned traces follow `plan.projections`; local updates are installed
    /// before returning and are not included in the result list.
    pub fn apply_plan(op: Op, plan: ApplyPlan) -> Vec<Trace> {
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
                let body = region.body.view();
                assert_eq!(
                    body.closure_inputs().len(),
                    region.captures.len(),
                    "region capture arity mismatch"
                );
                assert_eq!(
                    body.closure_symbols().len(),
                    region.symbol_captures.len(),
                    "symbol capture arity mismatch"
                );
                ApplyRegion {
                    symbol_captures: region.symbol_captures,
                    body: region.body,
                    captures: region.captures.into_iter().map(&mut bind).collect(),
                    arguments: Vec::new(),
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
                    (ApplyProjection::Value(index), kinds) => match &kinds[*index] {
                        (Kind::Value, info) => info,
                        _ => panic!("expected a value projection"),
                    },
                    (ApplyProjection::CellContents(index), kinds) => match &kinds[*index] {
                        (Kind::Cell, info) => info,
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
        let invocation = Arc::new(Self::Apply(Apply {
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

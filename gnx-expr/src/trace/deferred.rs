use std::collections::BTreeMap;
use std::sync::{Arc, OnceLock, Weak};

use crate::{CellAccess, HExpr, Kind};

use super::{
    CellKey, CellTrace, ContextID, Invocation, ObjectTracer, Trace, TraceContext, TraceData,
    ValueInfo,
};

/// Trace representation of a call awaiting the caller's alias specialization.
pub struct DeferredCall {
    pub(super) context_id: ContextID,
    pub(super) body: Arc<HExpr>,
    pub(super) closure: Vec<ObjectTracer>,
    pub(super) inputs: Vec<ObjectTracer>,
    pub(super) results: Vec<Projection>,
    pub(super) value_parents: Vec<Trace>,
    pub(super) aliasing_parents: Vec<Arc<Invocation>>,
    pub(super) outputs: Vec<ValueInfo>,
    pub(super) result_slots: OnceLock<Box<[Weak<TraceData>]>>,
}

impl DeferredCall {
    pub fn body(&self) -> &Arc<HExpr> {
        &self.body
    }
    pub fn captures(&self) -> &[ObjectTracer] {
        &self.closure
    }
    pub fn inputs(&self) -> &[ObjectTracer] {
        &self.inputs
    }
}

pub(super) enum Projection {
    Value(usize),
    CellContents(usize),
    LocalUpdate(CellTrace),
}

impl Invocation {
    /// Record a high-level call in the active trace without resolving external
    /// cell aliases. Known caller-local cells use SSA snapshots and writebacks.
    pub fn defer_call(
        body: Arc<HExpr>,
        closure: &[ObjectTracer],
        inputs: &[ObjectTracer],
    ) -> Vec<ObjectTracer> {
        let context = TraceContext::current();
        assert_eq!(closure.len(), body.closure().len());
        assert_eq!(inputs.len(), body.inputs().len());
        let mut locals = BTreeMap::<CellKey, (CellTrace, CellTrace, bool)>::new();
        let mut value_parents = Vec::new();
        let mut external_read = false;
        let mut external_write = false;
        let mut capture_operand = |((var, _), object): (&(_, _), &ObjectTracer)| {
            match object {
                ObjectTracer::Ref(value) => {
                    value_parents.push(value.internal_clone());
                    return ObjectTracer::Ref(value.internal_clone());
                }
                ObjectTracer::Cell(cell) if !cell.is_maybe_aliased() => {
                    let (_, snapshot, written) =
                        locals.entry(CellKey::from(cell)).or_insert_with(|| {
                            let mut initial = cell.read_home();
                            initial.make_internal();
                            value_parents.push(initial.internal_clone());
                            (cell.clone(), CellTrace::new(initial), false)
                        });
                    *written |= matches!(
                        body.cell_access(var),
                        Some(CellAccess::Write | CellAccess::ReadWrite)
                    );
                    return ObjectTracer::Cell(snapshot.clone());
                }
                ObjectTracer::Cell(_) => match body.cell_access(var) {
                    Some(CellAccess::Read) => external_read = true,
                    Some(CellAccess::Write) => external_write = true,
                    Some(CellAccess::ReadWrite) => {
                        external_read = true;
                        external_write = true;
                    }
                    _ => {}
                },
            }
            object.clone()
        };
        let closure_operands = body
            .closure()
            .iter()
            .zip(closure)
            .map(&mut capture_operand)
            .collect();
        let input_operands = body
            .inputs()
            .iter()
            .zip(inputs)
            .map(capture_operand)
            .collect();
        let bindings: BTreeMap<_, _> = body
            .closure()
            .iter()
            .chain(body.inputs())
            .map(|(var, _)| *var)
            .zip(closure.iter().chain(inputs))
            .collect();

        enum ResultSlot<'a> {
            Passthrough(&'a ObjectTracer),
            Value(usize),
            FreshCell(usize),
        }
        let mut projections = Vec::new();
        let mut infos = Vec::new();
        let mut fresh = BTreeMap::new();
        let slots: Vec<_> = body
            .outputs()
            .iter()
            .enumerate()
            .map(
                |(index, var)| match body.kind(var).expect("undefined deferred call output") {
                    Kind::Value(info) => {
                        if let Some(&binding) = bindings.get(var) {
                            return ResultSlot::Passthrough(binding);
                        }
                        let slot = projections.len();
                        projections.push(Projection::Value(index));
                        infos.push(info.clone());
                        ResultSlot::Value(slot)
                    }
                    Kind::Cell(info) => {
                        if let Some(&binding) = bindings.get(var) {
                            return ResultSlot::Passthrough(binding);
                        }
                        let slot = *fresh.entry(*var).or_insert_with(|| {
                            let slot = projections.len();
                            projections.push(Projection::CellContents(index));
                            infos.push(info.clone());
                            slot
                        });
                        ResultSlot::FreshCell(slot)
                    }
                },
            )
            .collect();
        let mut writebacks = Vec::new();
        for (original, snapshot, written) in locals.into_values() {
            if written {
                writebacks.push((original, projections.len()));
                infos.push(snapshot.type_info().clone());
                projections.push(Projection::LocalUpdate(snapshot));
            }
        }
        let invocation = Arc::new(Self::DeferredCall(DeferredCall {
            context_id: TraceContext::current_id(),
            body,
            closure: closure_operands,
            inputs: input_operands,
            results: projections,
            value_parents,
            aliasing_parents: context
                .as_ref()
                .filter(|_| external_read || external_write)
                .map(|ctx| ctx.aliasing_parents(external_write))
                .unwrap_or_default(),
            outputs: infos,
            result_slots: OnceLock::new(),
        }));
        let returned = Self::returned_values(invocation.clone());
        // Use the same ordering as direct accesses to possibly aliased cells.
        // Tracking writes also retains calls that have no value results.
        if let Some(context) = context {
            if external_write {
                context.note_aliased_write(invocation);
            } else if external_read {
                context.note_aliased_read(invocation);
            }
        }
        for (cell, slot) in writebacks {
            cell.set(returned[slot].clone());
        }
        let mut fresh = BTreeMap::new();
        slots
            .into_iter()
            .map(|slot| match slot {
                ResultSlot::Passthrough(object) => object.clone(),
                ResultSlot::Value(index) => returned[index].clone().into(),
                ResultSlot::FreshCell(index) => fresh
                    .entry(index)
                    .or_insert_with(|| CellTrace::new(returned[index].clone()))
                    .clone()
                    .into(),
            })
            .collect()
    }
}

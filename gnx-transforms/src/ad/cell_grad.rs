//! Present cell entry values to reverse AD; retain identity only at the boundary.
use std::collections::{BTreeMap, BTreeSet};

use gnx_expr::{ApplyRegion, CellKey, CellTrace, InputId, InputSpec, Kind, ObjectTracer, Trace};
use gnx_graph::Graph;

use crate::cache::TraceEntry;
use crate::traced::{zeros, zeros_like};
use crate::{TransformError, TransformResult};

pub(super) struct Operands {
    pub body: ApplyRegion,
    pub primals: Vec<Trace>,
    /// Physical cotangent index for each logical argument leaf.
    pub slots: Vec<Option<usize>>,
}

pub(super) fn prepare<O: Graph>(
    entry: &TraceEntry<O>,
    objects: &[ObjectTracer],
    selected: &[bool],
) -> TransformResult<Operands> {
    let mut spec = InputSpec::from_args(&entry.closure, objects);
    // Equal value tracers passed twice are two independent AD arguments.
    // Only mutable cell identities (including captured aliases) are unified.
    for (i, (id, kind, _)) in spec.inputs.iter_mut().enumerate() {
        if *kind == Kind::Value {
            *id = InputId::Input(i);
        }
    }
    for (i, (id, kind, _)) in spec.closure.iter_mut().enumerate() {
        if *kind == Kind::Value {
            *id = InputId::Closure(i);
        }
    }
    let view = entry.expr.view();
    // Distinct placeholders in an enclosing trace may be bound to the same
    // cell at replay. Never specialize their cotangents as if they were disjoint.
    let cells: BTreeMap<_, _> = objects
        .iter()
        .chain(&entry.closure)
        .filter_map(|o| match o {
            ObjectTracer::Cell(c) => Some((CellKey::from(c), c)),
            _ => None,
        })
        .collect();
    for (&key, cell) in &cells {
        if cell.is_placeholder()
            && cells.iter().any(|(&other_key, other)| {
                key != other_key
                    && other.is_maybe_aliased()
                    && cell.type_info() == other.type_info()
            })
        {
            return Err(TransformError::invalid(
                "grad needs known ArrayRef aliases; call grad outside the enclosing trace or pass entry values",
            ));
        }
    }
    let lowered = view.lower(&spec)?;
    let pure = lowered.read_only_value_body()?;
    let selected: BTreeSet<_> = spec
        .inputs
        .iter()
        .zip(selected)
        .filter_map(|((id, ..), yes)| yes.then_some(*id))
        .collect();
    let snapshot = |id: InputId| {
        let object = match id {
            InputId::Closure(i) => &entry.closure[i],
            InputId::Input(i) => &objects[i],
        };
        match object {
            ObjectTracer::Ref(v) => v.clone(),
            ObjectTracer::Cell(c) => c.get(),
        }
    };
    let mut closure: Vec<ObjectTracer> = lowered
        .closure
        .iter()
        .map(|&id| snapshot(id).into())
        .collect();
    let mut primals = Vec::new();
    let mut slots = BTreeMap::new();
    let mut captured = Vec::new();
    for &id in &lowered.inputs {
        let value = snapshot(id);
        let active = selected.contains(&id);
        captured.push(!active);
        if active {
            slots.insert(id, primals.len());
            primals.push(value);
        } else {
            closure.push(value.into());
        }
    }
    let body = pure.view().capture_inputs(&captured).build()?;
    Ok(Operands {
        body: ApplyRegion::new(body.handle(), closure),
        primals,
        slots: spec
            .inputs
            .iter()
            .map(|(id, ..)| slots.get(id).copied())
            .collect(),
    })
}

/// One fresh cell per primal identity, including for zero gradients.
pub(super) fn cotangents(
    objects: &[ObjectTracer],
    slots: &[Option<usize>],
    cts: &[Trace],
) -> Vec<ObjectTracer> {
    let mut cells = BTreeMap::new();
    objects
        .iter()
        .zip(slots)
        .map(|(object, slot)| match object {
            ObjectTracer::Ref(_) => {
                slot.map_or_else(|| zeros_like(object), |i| cts[i].clone().into())
            }
            ObjectTracer::Cell(c) => ObjectTracer::Cell(
                cells
                    .entry(CellKey::from(c))
                    .or_insert_with(|| {
                        CellTrace::new(
                            slot.map_or_else(|| zeros(c.type_info()), |i| cts[i].clone()),
                        )
                    })
                    .clone(),
            ),
        })
        .collect()
}

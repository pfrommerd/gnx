//! Map the value form of a cell-using body. Each mapped cell is snapshotted
//! before the map, and receives its stacked final values once afterwards.
use crate::{TransformError, TransformResult};
use gnx_expr::{
    ApplyRegion, CellKey, CellTrace, HExpr, InputId, InputSpec, Kind, ObjectTracer, lower::Output,
};
use std::collections::HashMap;

pub(super) fn apply(
    expr: HExpr<'_>,
    closure: &[ObjectTracer],
    inputs: &[ObjectTracer],
    axes: &[Option<usize>],
    out_axes: &[usize],
    size: Option<usize>,
) -> TransformResult<Vec<ObjectTracer>> {
    let mut spec = InputSpec::from_args(closure, inputs);
    let axis = |id: InputId| match id {
        InputId::Closure(_) => None,
        InputId::Input(i) => axes[i],
    };
    let arg = |id: InputId| match id {
        InputId::Closure(i) => &closure[i],
        InputId::Input(i) => &inputs[i],
    };
    for ((id, kind, info), (source, param)) in spec
        .inputs
        .iter_mut()
        .zip((0..).map(InputId::Input).zip(expr.inputs()))
        .chain(
            spec.closure
                .iter_mut()
                .zip((0..).map(InputId::Closure).zip(expr.closure_inputs())),
        )
    {
        // Immutable aliases may be mapped on different axes. Mutable aliases must agree.
        if *kind == Kind::Value {
            *id = source;
        } else if axis(*id) != axis(source) {
            return Err(TransformError::invalid(
                "aliased map cells must have the same in_axis",
            ));
        }
        *info = param.info_ref().to_owned();
    }
    let lowered = expr.lower(&spec)?;
    let mut selected = Vec::new();
    let mut stack_axes = Vec::new();
    enum Slot {
        Value(usize),
        Fresh(usize),
        Input(InputId),
    }
    let mut slots = Vec::new();
    let mut fresh = HashMap::new();
    for (out, &ax) in lowered.outputs.iter().zip(out_axes) {
        let mut select = |idx| {
            let i = selected.len();
            selected.push(idx);
            stack_axes.push(ax);
            i
        };
        slots.push(match *out {
            Output::Value { result_idx } => Slot::Value(select(result_idx)),
            Output::NewCell { id, result_idx } => {
                let (i, prev) = *fresh.entry(id).or_insert_with(|| (select(result_idx), ax));
                if prev != ax {
                    return Err(TransformError::invalid(
                        "aliased map outputs must have the same out_axis",
                    ));
                }
                Slot::Fresh(i)
            }
            Output::InputCell { input_id } => {
                if axis(input_id) != Some(ax) {
                    return Err(TransformError::invalid(
                        "returned map cell must retain its mapped axis",
                    ));
                }
                Slot::Input(input_id)
            }
        });
    }
    for &(id, idx) in &lowered.updates {
        stack_axes.push(axis(id).ok_or_else(|| {
            TransformError::invalid("transformed functions cannot write captured cells")
        })?);
        selected.push(idx);
        // As in cell_grad, two placeholders in an enclosing trace can replay
        // as the same cell. Don't snapshot/commit them as disjoint lanes.
        let ObjectTracer::Cell(written) = arg(id) else {
            unreachable!()
        };
        for other in inputs.iter().chain(closure) {
            if let ObjectTracer::Cell(other) = other {
                if CellKey::from(written) != CellKey::from(other)
                    && written.type_info() == other.type_info()
                    && ((written.is_placeholder() && other.is_maybe_aliased())
                        || (other.is_placeholder() && written.is_maybe_aliased()))
                {
                    return Err(TransformError::invalid(
                        "map needs known aliases for writable ArrayRefs; map the jitted function outside the enclosing trace",
                    ));
                }
            }
        }
    }
    let body = lowered.projected_value_body(&selected)?;
    let shared: Vec<_> = lowered
        .inputs
        .iter()
        .map(|&id| axis(id).is_none())
        .collect();
    let body = body.view().capture_inputs(&shared).build()?;
    let snapshot = |id| match arg(id) {
        ObjectTracer::Cell(c) => c.get(),
        ObjectTracer::Ref(t) => t.clone(),
    };
    let mut captures: Vec<_> = lowered
        .closure
        .iter()
        .map(|&id| snapshot(id).into())
        .collect();
    let mut mapped = Vec::new();
    let mut in_axes = Vec::new();
    for &id in &lowered.inputs {
        let value = snapshot(id);
        if let Some(ax) = axis(id) {
            mapped.push(value);
            in_axes.push(ax);
        } else {
            captures.push(value.into());
        }
    }
    let mapped = super::region::map(
        ApplyRegion::new(body.handle(), captures),
        &mapped,
        &in_axes,
        &stack_axes,
        size,
    )?;
    let mut cells = HashMap::new();
    let outputs = slots
        .into_iter()
        .map(|s| match s {
            Slot::Value(i) => mapped[i].clone().into(),
            Slot::Input(id) => arg(id).clone(),
            Slot::Fresh(i) => cells
                .entry(i)
                .or_insert_with(|| CellTrace::new(mapped[i].clone()))
                .clone()
                .into(),
        })
        .collect();
    for ((id, _), v) in lowered
        .updates
        .iter()
        .zip(&mapped[mapped.len() - lowered.updates.len()..])
    {
        let ObjectTracer::Cell(c) = arg(*id) else {
            unreachable!()
        };
        c.set(v.clone());
    }
    Ok(outputs)
}

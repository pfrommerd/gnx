use crate::ops::indexing::*;
use gnx_expr::pass::Builder;
use gnx_expr::{Eqn, Op};

use gnx_transforms::map::batch::{BatchRule, Batched, materialize};
use gnx_transforms::map::def::{leading, shifted};
use gnx_transforms::util::with_axis;
use gnx_transforms::{TransformError, TransformResult};

fn batch_indices(
    b: &mut Builder,
    size: usize,
    eqn: &Eqn,
    inputs: &[Batched],
    op: Op,
    array_count: usize,
) -> TransformResult<Vec<Batched>> {
    let (arrays, rest) = inputs.split_at(array_count);
    if rest.iter().any(|x| x.axis.is_some()) {
        return Err(TransformError::invalid(
            "operation does not batch scalar fill values",
        ));
    }
    let mut args = arrays
        .iter()
        .map(|&x| materialize(b, x, size, 0))
        .collect::<TransformResult<Vec<_>>>()?;
    args.extend(rest.iter().map(|x| x.var));
    let metadata = with_axis(&eqn.outputs()[0].1, 0, size)?;
    Ok(vec![Batched::at(
        b.emit(op, &args, [metadata], eqn.subexprs().to_vec())[0],
        0,
    )])
}

impl BatchRule for GatherOperation {
    fn batch(
        &self,
        b: &mut Builder,
        size: usize,
        eqn: &Eqn,
        inputs: &[Batched],
        _: &[Vec<Batched>],
    ) -> TransformResult<Vec<Batched>> {
        let mut p = self.parameters().clone();
        p.slice_sizes.insert(0, usize::from(size != 0));
        let d = &mut p.dimensions;
        d.offset = shifted(&d.offset);
        d.collapsed = shifted(&d.collapsed);
        d.start_index_map = shifted(&d.start_index_map);
        d.operand_batch = leading(&d.operand_batch);
        d.index_batch = leading(&d.index_batch);
        batch_indices(b, size, eqn, inputs, gather(p), 2)
    }
}

impl BatchRule for ScatterOperation {
    fn batch(
        &self,
        b: &mut Builder,
        size: usize,
        eqn: &Eqn,
        inputs: &[Batched],
        _: &[Vec<Batched>],
    ) -> TransformResult<Vec<Batched>> {
        let mut p = self.parameters().clone();
        let d = &mut p.dimensions;
        d.update_window = shifted(&d.update_window);
        d.inserted_window = shifted(&d.inserted_window);
        d.index_map = shifted(&d.index_map);
        d.operand_batch = leading(&d.operand_batch);
        d.index_batch = leading(&d.index_batch);
        let op = match self.update() {
            ScatterUpdate::Replace => scatter(p),
            ScatterUpdate::Add => scatter_add(p),
            ScatterUpdate::Subtract => scatter_sub(p),
            ScatterUpdate::Multiply => scatter_mul(p),
            ScatterUpdate::Min => scatter_min(p),
            ScatterUpdate::Max => scatter_max(p),
            ScatterUpdate::Apply => scatter_apply(p),
        };
        batch_indices(b, size, eqn, inputs, op, 3)
    }
}

gnx_expr::register_op_interface!(dyn BatchRule: GatherOperation, ScatterOperation);

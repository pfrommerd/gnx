use crate::ops::elementwise::neg_op;
use crate::ops::indexing::*;
use crate::ops::mul_op;
use gnx_expr::pass::Builder;
use gnx_expr::{Eqn, Var};

use gnx_transforms::ad::jvp::instantiate;
use gnx_transforms::ad::{Dual, JvpRule, Operand, Tangent, TransposeRule};
use gnx_transforms::util::{copy_eqn, dims, dtype, info, zeros};
use gnx_transforms::{TransformError, TransformResult};

fn primal(b: &mut Builder, eqn: &Eqn, inputs: &[Dual]) -> Var {
    let args: Vec<_> = inputs.iter().map(|d| d.primal).collect();
    copy_eqn(b, eqn, &args, &[])[0]
}

fn residual(operand: &Operand) -> TransformResult<Var> {
    match operand {
        Operand::Residual(var) => Ok(*var),
        _ => Err(TransformError::invalid("indices must be residual values")),
    }
}

fn tangent(b: &mut Builder, value: Dual) -> Var {
    let metadata = b.info(value.primal).clone();
    instantiate(b, value.tangent, &metadata)
}

impl JvpRule for GatherOperation {
    fn jvp(
        &self,
        b: &mut Builder,
        eqn: &Eqn,
        inputs: &[Dual],
        _: &[Vec<Dual>],
    ) -> TransformResult<Vec<Dual>> {
        let result = primal(b, eqn, inputs);
        let derivative = if let Tangent::Var(x) = inputs[0].tangent {
            let mut args = vec![x, inputs[1].primal];
            if self.parameters().mode == BoundsMode::FillOrDrop {
                let metadata = b.info(inputs[2].primal).clone();
                args.push(zeros(b, &metadata));
            }
            Tangent::Var(copy_eqn(b, eqn, &args, &[])[0])
        } else {
            Tangent::Zero
        };
        Ok(vec![Dual {
            primal: result,
            tangent: derivative,
        }])
    }
}

impl TransposeRule for GatherOperation {
    fn transpose(
        &self,
        b: &mut Builder,
        _: &Eqn,
        operands: &[Operand],
        _: &[Vec<Var>],
        cts: &[Tangent],
    ) -> TransformResult<Vec<Tangent>> {
        let ([Operand::Linear(metadata), indices, ..], [Tangent::Var(ct)]) = (operands, cts) else {
            return Err(TransformError::invalid("gather transpose signature"));
        };
        let p = self.parameters();
        let d = &p.dimensions;
        let op = scatter_add(ScatterParameters {
            dimensions: ScatterDimensions {
                update_window: d.offset.clone(),
                inserted_window: d.collapsed.clone(),
                index_map: d.start_index_map.clone(),
                operand_batch: d.operand_batch.clone(),
                index_batch: d.index_batch.clone(),
            },
            mode: p.mode,
            sorted: p.sorted,
            unique: p.unique,
        });
        let args = [zeros(b, metadata), residual(indices)?, *ct];
        let dx = b.emit1(op, &args, metadata.clone());
        let mut result = vec![Tangent::Zero; operands.len()];
        result[0] = Tangent::Var(dx);
        Ok(result)
    }
}

impl JvpRule for ScatterOperation {
    fn jvp(
        &self,
        b: &mut Builder,
        eqn: &Eqn,
        inputs: &[Dual],
        _: &[Vec<Dual>],
    ) -> TransformResult<Vec<Dual>> {
        let update = self.update();
        if update == ScatterUpdate::Apply {
            return Err(TransformError::invalid(
                "scatter_apply does not support differentiation",
            ));
        }
        let primal = primal(b, eqn, inputs);
        let [x, i, u] = inputs else {
            return Err(TransformError::invalid("scatter JVP signature"));
        };
        let ty = b.info(primal).clone();
        if matches!(update, ScatterUpdate::Min | ScatterUpdate::Max) {
            return extremal(b, primal, *x, *i, *u, self.parameters());
        }
        if update == ScatterUpdate::Replace && !self.parameters().unique {
            return replacement(b, primal, *x, *i, *u, self.parameters());
        }
        let tangent = if update == ScatterUpdate::Multiply {
            let dx = match x.tangent {
                Tangent::Zero => Tangent::Zero,
                Tangent::Var(dx) => {
                    Tangent::Var(b.emit1(eqn.op().clone(), &[dx, i.primal, u.primal], ty.clone()))
                }
            };
            let du = match u.tangent {
                Tangent::Zero => Tangent::Zero,
                Tangent::Var(du) => {
                    let zero = zeros(b, &ty);
                    let s = b.emit1(
                        scatter_add(self.parameters().clone()),
                        &[zero, i.primal, du],
                        ty.clone(),
                    );
                    Tangent::Var(b.emit1(mul_op(), &[x.primal, s], ty.clone()))
                }
            };
            gnx_transforms::ad::jvp::accumulate(b, dx, du)
        } else {
            let dx = tangent(b, *x);
            let du = tangent(b, *u);
            Tangent::Var(b.emit1(eqn.op().clone(), &[dx, i.primal, du], ty))
        };
        Ok(vec![Dual { primal, tangent }])
    }
}

fn gather_updates(
    b: &mut Builder,
    value: Var,
    index: Var,
    metadata: &gnx_expr::VarInfo,
    p: &ScatterParameters,
) -> TransformResult<Var> {
    let d = &p.dimensions;
    let shape = dims(metadata)?;
    let mut axes = d.update_window.iter();
    let slices = (0..dims(b.info(value))?.len())
        .map(|axis| {
            if d.inserted_window.contains(&axis) || d.operand_batch.contains(&axis) {
                Ok(1)
            } else {
                axes.next()
                    .and_then(|&i| shape.get(i))
                    .copied()
                    .ok_or_else(|| TransformError::invalid("invalid scatter window dimensions"))
            }
        })
        .collect::<TransformResult<_>>()?;
    let op = gather(GatherParameters {
        dimensions: GatherDimensions {
            offset: d.update_window.clone(),
            collapsed: d.inserted_window.clone(),
            start_index_map: d.index_map.clone(),
            operand_batch: d.operand_batch.clone(),
            index_batch: d.index_batch.clone(),
        },
        slice_sizes: slices,
        mode: p.mode,
        sorted: p.sorted,
        unique: p.unique,
    });
    let mut args = vec![value, index];
    if p.mode == BoundsMode::FillOrDrop {
        args.push(zeros(b, &info(&[], dtype(metadata)?)));
    }
    Ok(b.emit1(op, &args, metadata.clone()))
}

impl TransposeRule for ScatterOperation {
    fn transpose(
        &self,
        b: &mut Builder,
        eqn: &Eqn,
        operands: &[Operand],
        _: &[Vec<Var>],
        cts: &[Tangent],
    ) -> TransformResult<Vec<Tangent>> {
        let ([x, i, u], [Tangent::Var(ct)]) = (operands, cts) else {
            return Err(TransformError::invalid("scatter transpose signature"));
        };
        let i = residual(i)?;
        let update = self.update();
        let p = self.parameters();
        let mut dx = Tangent::Zero;
        let mut du = Tangent::Zero;
        if let Operand::Linear(metadata) = x {
            let dx_var = match update {
                ScatterUpdate::Add | ScatterUpdate::Subtract => *ct,
                ScatterUpdate::Multiply => {
                    b.emit1(eqn.op().clone(), &[*ct, i, residual(u)?], metadata.clone())
                }
                ScatterUpdate::Replace if p.unique => {
                    let updates = match u {
                        Operand::Residual(var) => b.info(*var).clone(),
                        Operand::Linear(info) => info.clone(),
                    };
                    let zero = zeros(b, &updates);
                    b.emit1(eqn.op().clone(), &[*ct, i, zero], metadata.clone())
                }
                _ => {
                    return Err(TransformError::invalid(
                        "unsupported scatter transpose rule",
                    ));
                }
            };
            dx = Tangent::Var(dx_var);
        }
        if let Operand::Linear(metadata) = u {
            let mut update_ct = *ct;
            if update == ScatterUpdate::Multiply {
                if !p.unique {
                    return Err(TransformError::invalid(
                        "scatter multiplication transpose needs unique indices",
                    ));
                }
                update_ct = b.emit1(mul_op(), &[*ct, residual(x)?], b.info(*ct).clone());
            }
            let delta = gather_updates(b, update_ct, i, metadata, p)?;
            du = Tangent::Var(if update == ScatterUpdate::Subtract {
                b.emit1(neg_op(), &[delta], metadata.clone())
            } else {
                delta
            });
        }
        Ok(vec![dx, Tangent::Zero, du])
    }
}

gnx_expr::register_op_interface!(dyn JvpRule: GatherOperation, ScatterOperation);
gnx_expr::register_op_interface!(dyn TransposeRule: GatherOperation, ScatterOperation);

impl JvpRule for DynamicUpdateSliceOperation {
    fn jvp(
        &self,
        b: &mut Builder,
        eqn: &Eqn,
        inputs: &[Dual],
        _: &[Vec<Dual>],
    ) -> TransformResult<Vec<Dual>> {
        let result = primal(b, eqn, inputs);
        let mut args = vec![tangent(b, inputs[0]), tangent(b, inputs[1])];
        args.extend(inputs[2..].iter().map(|value| value.primal));
        let tangent = copy_eqn(b, eqn, &args, &[])[0];
        Ok(vec![Dual {
            primal: result,
            tangent: Tangent::Var(tangent),
        }])
    }
}

impl TransposeRule for DynamicUpdateSliceOperation {
    fn transpose(
        &self,
        b: &mut Builder,
        _: &Eqn,
        operands: &[Operand],
        _: &[Vec<Var>],
        cts: &[Tangent],
    ) -> TransformResult<Vec<Tangent>> {
        let ([x, u, starts @ ..], [Tangent::Var(ct)]) = (operands, cts) else {
            return Err(TransformError::invalid(
                "dynamic_update_slice transpose signature",
            ));
        };
        let starts = starts
            .iter()
            .map(residual)
            .collect::<TransformResult<Vec<_>>>()?;
        let mut result = vec![Tangent::Zero; operands.len()];
        let update = match u {
            Operand::Linear(info) => info.clone(),
            Operand::Residual(var) => b.info(*var).clone(),
        };
        if let Operand::Linear(info) = x {
            let zero = zeros(b, &update);
            let mut args = vec![*ct, zero];
            args.extend_from_slice(&starts);
            result[0] = Tangent::Var(b.emit1(dynamic_update_slice(), &args, info.clone()));
        }
        if u.is_linear() {
            let mut args = vec![*ct];
            args.extend(starts);
            result[1] = Tangent::Var(b.emit1(dynamic_slice(), &args, update));
        }
        Ok(result)
    }
}

impl JvpRule for SelectNOperation {
    fn jvp(
        &self,
        b: &mut Builder,
        eqn: &Eqn,
        inputs: &[Dual],
        _: &[Vec<Dual>],
    ) -> TransformResult<Vec<Dual>> {
        let result = primal(b, eqn, inputs);
        let mut args = vec![inputs[0].primal];
        args.extend(inputs[1..].iter().map(|&value| tangent(b, value)));
        let tangent = copy_eqn(b, eqn, &args, &[])[0];
        Ok(vec![Dual {
            primal: result,
            tangent: Tangent::Var(tangent),
        }])
    }
}

impl TransposeRule for SelectNOperation {
    fn transpose(
        &self,
        b: &mut Builder,
        _: &Eqn,
        operands: &[Operand],
        _: &[Vec<Var>],
        cts: &[Tangent],
    ) -> TransformResult<Vec<Tangent>> {
        let [Tangent::Var(ct)] = cts else {
            return Err(TransformError::invalid("select_n transpose signature"));
        };
        let (selector, cases) = operands
            .split_first()
            .ok_or_else(|| TransformError::invalid("select_n operands"))?;
        let selector = residual(selector)?;
        let mut outputs = vec![Tangent::Zero];
        for (i, case) in cases.iter().enumerate() {
            if let Operand::Linear(info) = case {
                let zero = zeros(b, info);
                let mut args = vec![zero; cases.len() + 1];
                args[0] = selector;
                args[i + 1] = *ct;
                outputs.push(Tangent::Var(b.emit1(select_n(), &args, info.clone())));
            } else {
                outputs.push(Tangent::Zero);
            }
        }
        Ok(outputs)
    }
}

gnx_expr::register_op_interface!(dyn JvpRule: DynamicUpdateSliceOperation, SelectNOperation);
gnx_expr::register_op_interface!(dyn TransposeRule: DynamicUpdateSliceOperation, SelectNOperation);

fn equal(b: &mut Builder, a: Var, c: Var) -> TransformResult<Var> {
    let shape = dims(b.info(a))?;
    Ok(b.emit1(
        crate::ops::elementwise::eq_op(),
        &[a, c],
        info(&shape, gnx_common::DType::Bool),
    ))
}

fn cast_like(b: &mut Builder, predicate: Var, array: Var) -> Var {
    b.emit1(
        crate::ops::dtype::convert_element_type_op(),
        &[predicate],
        b.info(array).clone(),
    )
}

fn masked(b: &mut Builder, pred: Var, value: Var) -> Var {
    let zero = zeros(b, &b.info(value).clone());
    gnx_transforms::util::select(b, pred, value, zero)
}

fn extremal(
    b: &mut Builder,
    primal: Var,
    x: Dual,
    i: Dual,
    u: Dual,
    p: &ScatterParameters,
) -> TransformResult<Vec<Dual>> {
    let input_info = b.info(x.primal).clone();
    let update_info = b.info(u.primal).clone();
    let gathered = gather_updates(b, primal, i.primal, &update_info, p)?;
    let add = scatter_add(p.clone());
    let input_wins = equal(b, primal, x.primal)?;
    let update_wins = equal(b, gathered, u.primal)?;
    let input_count = cast_like(b, input_wins, primal);
    let update_count = cast_like(b, update_wins, u.primal);
    let count = b.emit1(
        add.clone(),
        &[input_count, i.primal, update_count],
        input_info.clone(),
    );
    let dx = tangent(b, x);
    let du = tangent(b, u);
    let dx = masked(b, input_wins, dx);
    let du = masked(b, update_wins, du);
    let combined = b.emit1(add, &[dx, i.primal, du], input_info.clone());
    let zero = zeros(b, &input_info);
    let one = b.emit1(crate::ops::ones_op(), &[], input_info.clone());
    let empty = equal(b, count, zero)?;
    let count = gnx_transforms::util::select(b, empty, one, count);
    let t = b.emit1(
        crate::ops::elementwise::div_op(),
        &[combined, count],
        input_info,
    );
    Ok(vec![Dual {
        primal,
        tangent: Tangent::Var(t),
    }])
}

fn replacement(
    b: &mut Builder,
    primal: Var,
    x: Dual,
    i: Dual,
    u: Dual,
    p: &ScatterParameters,
) -> TransformResult<Vec<Dual>> {
    use crate::ops::{add_op, creation::iota, ones_op, reshape_op};
    use gnx_common::DType;
    let output = b.info(x.primal).clone();
    let x_shape = dims(&output)?;
    let update_shape = dims(b.info(u.primal))?;
    let count = update_shape
        .iter()
        .try_fold(1usize, |prod, &dim| prod.checked_mul(dim))
        .ok_or_else(|| TransformError::invalid("scatter update size overflow"))?;
    let indices_info = info(&update_shape, DType::I64);
    let ids = b.emit1(iota(0), &[], info(&[count], DType::I64));
    let ids = b.emit1(reshape_op(), &[ids], indices_info.clone());
    let ones = b.emit1(ones_op(), &[], indices_info.clone());
    let ids = b.emit1(add_op(), &[ids, ones], indices_info.clone());
    let out_ids = info(&x_shape, DType::I64);
    let zero = zeros(b, &out_ids);
    let winners = b.emit1(scatter_max(p.clone()), &[zero, i.primal, ids], out_ids);
    let unused = equal(b, winners, zero)?;
    let selected = gather_updates(b, winners, i.primal, &indices_info, p)?;
    let selected = equal(b, selected, ids)?;
    let dx = tangent(b, x);
    let du = tangent(b, u);
    let dx = masked(b, unused, dx);
    let du = masked(b, selected, du);
    let value = b.emit1(scatter_add(p.clone()), &[dx, i.primal, du], output);
    Ok(vec![Dual {
        primal,
        tangent: Tangent::Var(value),
    }])
}

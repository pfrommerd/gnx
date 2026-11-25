use crate::ops::indexing::{BoundsMode, GatherDimensions, GatherParameters, gather, slice};
use crate::ops::reduction::{ReduceOperation, ReduceWindowOperation, Window, reduce_window};
use crate::ops::shape::pad;
use crate::ops::{broadcast_in_dim_op, reshape_op};
use gnx_expr::pass::Builder;
use gnx_expr::{Eqn, Expr, Op, Var};

use gnx_transforms::ad::jvp::{instantiate, jvp_expr};
use gnx_transforms::ad::{Dual, JvpRule, Operand, Tangent, TransposeRule};
use gnx_transforms::map::batch::{Batched, batch_expr, materialize};
use gnx_transforms::util::{dims, dtype, info, reduce_sum, transpose, zeros};
use gnx_transforms::{TransformError, TransformResult};

use gnx_transforms::rules::{ReduceJvpRule, additive_reducer, copy_reduction_primal, reducer_body};

struct NonAdditiveReduce;

impl ReduceJvpRule for NonAdditiveReduce {
    fn jvp(
        &self,
        b: &mut Builder,
        eqn: Eqn<'_>,
        axes: &[usize],
        inputs: &[Dual],
        reducer: Expr<'_>,
    ) -> TransformResult<Vec<Dual>> {
        let (operands, identities) = inputs.split_at(eqn.outputs().len());
        if extremum_body(reducer).is_some() {
            let primal = copy_reduction_primal(b, eqn, inputs)[0];
            let x = operands[0].primal;
            let ty = dtype(b.info(x))?;
            let input_shape = dims(b.info(x))?;
            let map: Vec<_> = (0..input_shape.len())
                .filter(|axis| !axes.contains(axis))
                .collect();
            let result = b.emit1(broadcast_in_dim_op(&map), &[primal], b.info(x).clone());
            let indicator = b.emit1(
                crate::ops::elementwise::eq_op(),
                &[x, result],
                info(&input_shape, gnx_common::DType::Bool),
            );
            let indicator = b.emit1(
                crate::ops::dtype::convert_element_type_op(),
                &[indicator],
                info(&input_shape, ty),
            );
            let t = instantiate(b, operands[0].tangent, &info(&input_shape, ty));
            let selected = gnx_transforms::util::mul(b, t, indicator);
            let sum = reduce_sum(b, selected, axes)?;
            let n = reduce_sum(b, indicator, axes)?;
            let derivative = b.emit1(
                crate::ops::elementwise::div_op(),
                &[sum, n],
                b.info(primal).clone(),
            );
            return Ok(vec![Dual {
                primal,
                tangent: Tangent::Var(derivative),
            }]);
        }
        tree_jvp(b, eqn, axes, operands, identities, reducer)
    }
}

fn apply_dual(
    b: &mut Builder,
    value: Dual,
    op: Op,
    extra: &[Var],
    shape: &[usize],
) -> TransformResult<Dual> {
    let metadata = info(shape, dtype(b.info(value.primal))?);
    let apply = |b: &mut Builder, operand| {
        let inputs: Vec<_> = std::iter::once(operand)
            .chain(extra.iter().copied())
            .collect();
        b.emit1(op.clone(), &inputs, metadata.clone())
    };
    let primal = apply(b, value.primal);
    let tangent = match value.tangent {
        Tangent::Zero => Tangent::Zero,
        Tangent::Var(var) => Tangent::Var(apply(b, var)),
    };
    Ok(Dual { primal, tangent })
}

fn tree_jvp(
    b: &mut Builder,
    eqn: Eqn<'_>,
    axes: &[usize],
    operands: &[Dual],
    identities: &[Dual],
    reducer: Expr<'_>,
) -> TransformResult<Vec<Dual>> {
    let input_shape = dims(b.info(operands[0].primal))?;
    let output_shape = dims(&gnx_transforms::util::output_infos(eqn)[0])?;
    let product = |sizes: &[usize]| {
        sizes
            .iter()
            .try_fold(1usize, |acc, &n| acc.checked_mul(n))
            .ok_or_else(|| TransformError::invalid("reduction shape overflow"))
    };
    let rows = product(&output_shape)?;
    let mut columns = product(
        &axes
            .iter()
            .map(|&axis| input_shape[axis])
            .collect::<Vec<_>>(),
    )?;
    if columns == 0 {
        return identities
            .iter()
            .map(|&value| apply_dual(b, value, broadcast_in_dim_op(&[]), &[], &output_shape))
            .collect();
    }

    let perm: Vec<_> = (0..input_shape.len())
        .filter(|axis| !axes.contains(axis))
        .chain(axes.iter().copied())
        .collect();
    let mut values = Vec::with_capacity(operands.len());
    for &value in operands {
        let primal = transpose(b, value.primal, &perm)?;
        let tangent = match value.tangent {
            Tangent::Zero => Tangent::Zero,
            Tangent::Var(var) => Tangent::Var(transpose(b, var, &perm)?),
        };
        values.push(apply_dual(
            b,
            Dual { primal, tangent },
            reshape_op(),
            &[],
            &[rows, columns],
        )?);
    }

    while columns > 1 {
        let pairs = columns.div_ceil(2);
        let count = product(&[rows, pairs])?;
        let mut lhs = Vec::with_capacity(values.len());
        let mut rhs = Vec::with_capacity(values.len());
        for (&value, identity) in values.iter().zip(identities) {
            let even = apply_dual(
                b,
                value,
                slice(&[(0, rows, 1), (0, columns, 2)]),
                &[],
                &[rows, pairs],
            )?;
            let mut odd = apply_dual(
                b,
                value,
                slice(&[(0, rows, 1), (1, columns, 2)]),
                &[],
                &[rows, columns / 2],
            )?;
            if columns % 2 != 0 {
                let op = pad(&[(0, 0, 0), (0, 1, 0)]);
                let primal = b.emit1(
                    op.clone(),
                    &[odd.primal, identity.primal],
                    info(&[rows, pairs], dtype(b.info(odd.primal))?),
                );
                let tangent = if let Tangent::Var(var) = odd.tangent {
                    let scalar_info = b.info(identity.primal).clone();
                    let zero = gnx_transforms::util::zeros(b, &scalar_info);
                    Tangent::Var(b.emit1(op, &[var, zero], b.info(primal).clone()))
                } else {
                    Tangent::Zero
                };
                odd = Dual { primal, tangent };
            }
            lhs.push(apply_dual(b, even, reshape_op(), &[], &[count])?);
            rhs.push(apply_dual(b, odd, reshape_op(), &[], &[count])?);
        }
        lhs.extend(rhs);
        let batch = vectorize(reducer, count)?;
        values = jvp_expr(b, batch.view(), &[], &lhs)?
            .into_iter()
            .map(|value| apply_dual(b, value, reshape_op(), &[], &[rows, pairs]))
            .collect::<TransformResult<_>>()?;
        columns = pairs;
    }

    values
        .into_iter()
        .map(|value| apply_dual(b, value, reshape_op(), &[], &output_shape))
        .collect()
}

fn vectorize(reducer: Expr<'_>, count: usize) -> TransformResult<gnx_expr::ExprHandle> {
    let mut b = Builder::new();
    let mut inputs = Vec::new();
    for scalar in gnx_transforms::util::input_infos(reducer) {
        inputs.push(Batched::at(b.input(info(&[count], dtype(&scalar)?)), 0));
    }
    let outputs = batch_expr(&mut b, count, reducer, &[], &inputs)?;
    let outputs = outputs
        .into_iter()
        .map(|value| materialize(&mut b, value, count, 0))
        .collect::<TransformResult<_>>()?;
    Ok(b.finish(outputs).build()?.handle())
}

impl JvpRule for ReduceWindowOperation {
    fn jvp(
        &self,
        b: &mut Builder,
        eqn: Eqn<'_>,
        inputs: &[Dual],
        _: &[Vec<Dual>],
    ) -> TransformResult<Vec<Dual>> {
        let count = eqn.outputs().len();
        let (operands, identities) = inputs.split_at(count);
        if identities.iter().any(|value| !value.tangent.is_zero()) {
            return Err(TransformError::invalid(
                "reduce_window does not differentiate its identities",
            ));
        }
        let body = reducer_body(eqn)?;
        if additive_reducer(body) {
            let primal = copy_reduction_primal(b, eqn, inputs)[0];
            let derivative = match operands[0].tangent {
                Tangent::Zero => Tangent::Zero,
                Tangent::Var(tangent) => {
                    let scalar = b.info(identities[0].primal).clone();
                    let identity = zeros(b, &scalar);
                    Tangent::Var(
                        reduce_window(
                            b,
                            &[tangent],
                            &[identity],
                            self.window().clone(),
                            reducer_handle(eqn)?,
                        )?[0],
                    )
                }
            };
            return Ok(vec![Dual {
                primal,
                tangent: derivative,
            }]);
        }
        if let Some(is_max) = extremum_body(body) {
            let primal = copy_reduction_primal(b, eqn, inputs)[0];
            let tangent = match operands[0].tangent {
                Tangent::Zero => Tangent::Zero,
                Tangent::Var(tangent) => Tangent::Var(gather_window_winners(
                    b,
                    &self.window(),
                    operands[0].primal,
                    identities[0].primal,
                    tangent,
                    is_max,
                )?),
            };
            return Ok(vec![Dual { primal, tangent }]);
        }
        let mut region = Builder::new();
        let slots = gnx_transforms::util::input_infos(body);
        let (lhs, rhs) = slots.split_at(count);
        let mut parameters = |slots: &[gnx_expr::VarInfo]| {
            let primals: Vec<_> = slots.iter().map(|ty| region.input(ty.clone())).collect();
            primals
                .into_iter()
                .zip(slots.iter())
                .map(|(primal, ty)| Dual {
                    primal,
                    tangent: Tangent::Var(region.input(ty.clone())),
                })
                .collect::<Vec<_>>()
        };
        let mut parameters_left = parameters(lhs);
        parameters_left.extend(parameters(rhs));
        let results = jvp_expr(&mut region, body, &[], &parameters_left)?;
        let mut outputs: Vec<_> = results.iter().map(|value| value.primal).collect();
        for value in &results {
            let ty = region.info(value.primal).clone();
            outputs.push(instantiate(&mut region, value.tangent, &ty));
        }
        let differentiated = region.finish(outputs).build()?.handle();
        let mut arrays: Vec<_> = operands.iter().map(|value| value.primal).collect();
        for value in operands {
            let ty = b.info(value.primal).clone();
            arrays.push(instantiate(b, value.tangent, &ty));
        }
        let mut initial: Vec<_> = identities.iter().map(|value| value.primal).collect();
        for value in identities {
            let ty = b.info(value.primal).clone();
            initial.push(gnx_transforms::util::zeros(b, &ty));
        }
        let outputs = crate::ops::reduction::reduce_window(
            b,
            &arrays,
            &initial,
            self.window().clone(),
            differentiated,
        )?;
        Ok(outputs[..count]
            .iter()
            .zip(&outputs[count..])
            .map(|(&primal, &var)| Dual {
                primal,
                tangent: Tangent::Var(var),
            })
            .collect())
    }
}

gnx_expr::register_op_interface!(dyn JvpRule: ReduceWindowOperation);
gnx_expr::register_op_interface!(dyn TransposeRule: ReduceWindowOperation);

fn extremum_body(body: Expr<'_>) -> Option<bool> {
    use crate::ops::elementwise::{MaxOperation, MinOperation};
    let mut eqns = body.eqns();
    let (Some(eqn), None) = (eqns.next(), eqns.next()) else {
        return None;
    };
    let inputs: Vec<_> = body.inputs().map(|input| input.var()).collect();
    let outputs: Vec<_> = eqn.outputs().map(|output| output.var()).collect();
    if inputs.len() != 2
        || eqn.inputs() != inputs
        || outputs.len() != 1
        || body.outputs() != outputs
    {
        return None;
    }
    if eqn.op().downcast_ref::<MaxOperation>().is_some() {
        Some(true)
    } else if eqn.op().downcast_ref::<MinOperation>().is_some() {
        Some(false)
    } else {
        None
    }
}

fn gather_window_winners(
    b: &mut Builder,
    window: &Window,
    operand: Var,
    identity: Var,
    tangent: Var,
    is_max: bool,
) -> TransformResult<Var> {
    use crate::ops::{
        creation::{constant, iota},
        elementwise::{ge_op, le_op},
        select_op,
    };
    use gnx_common::DType;
    let input = b.info(operand).clone();
    let scalar = info(&[], dtype(&input)?);
    let index_scalar = info(&[], DType::I64);
    let shape = dims(&input)?;
    let count = shape
        .iter()
        .try_fold(1usize, |a, &b| a.checked_mul(b))
        .ok_or_else(|| TransformError::invalid("window operand size overflow"))?;
    let index = b.emit1(iota(0), &[], info(&[count], DType::I64));
    let index = b.emit1(reshape_op(), &[index], info(&shape, DType::I64));
    let sentinel = b.emit1(constant(-1.), &[], index_scalar.clone());

    let mut reducer = Builder::new();
    let a = reducer.input(scalar.clone());
    let ai = reducer.input(index_scalar.clone());
    let c = reducer.input(scalar.clone());
    let ci = reducer.input(index_scalar.clone());
    let comparison = if is_max { ge_op() } else { le_op() };
    let pred = reducer.emit1(comparison, &[a, c], info(&[], DType::Bool));
    let winning_value = reducer.emit1(select_op(), &[pred, a, c], scalar.clone());
    let winning_index = reducer.emit1(select_op(), &[pred, ai, ci], index_scalar);
    let reducer = reducer
        .finish(vec![winning_value, winning_index])
        .build()?
        .handle();
    let results = reduce_window(
        b,
        &[operand, index],
        &[identity, sentinel],
        window.clone(),
        reducer,
    )?;
    let output = b.info(results[0]).clone();
    let mut index_shape = dims(&output)?;
    index_shape.push(1);
    let winner = b.emit1(reshape_op(), &[results[1]], info(&index_shape, DType::I64));
    if count == 0 {
        return Ok(zeros(b, &output));
    }
    let tangent = b.emit1(reshape_op(), &[tangent], info(&[count], dtype(&input)?));
    let fill = zeros(b, &scalar);
    let op = gather(GatherParameters {
        dimensions: GatherDimensions {
            offset: vec![],
            collapsed: vec![0],
            start_index_map: vec![0],
            operand_batch: vec![],
            index_batch: vec![],
        },
        slice_sizes: vec![1],
        mode: BoundsMode::FillOrDrop,
        sorted: false,
        unique: false,
    });
    Ok(b.emit1(op, &[tangent, winner, fill], output))
}

// The linear window sum can be transposed as a sum of strided views of the
// padded input, including interior padding from base dilation.
impl TransposeRule for ReduceWindowOperation {
    fn transpose(
        &self,
        b: &mut Builder,
        eqn: Eqn<'_>,
        operands: &[Operand],
        _: &[Vec<Var>],
        cotangents: &[Tangent],
    ) -> TransformResult<Vec<Tangent>> {
        let ([Operand::Linear(input), Operand::Residual(_)], [Tangent::Var(ct)]) =
            (operands, cotangents)
        else {
            return Err(TransformError::invalid(
                "cannot transpose this window reduction",
            ));
        };
        if !additive_reducer(reducer_body(eqn)?) {
            return Err(TransformError::invalid(
                "expected an additive linear window reducer",
            ));
        }
        let output = &gnx_transforms::util::output_infos(eqn)[0];
        let out_shape = dims(output)?;
        if out_shape.contains(&0) {
            return Ok(vec![Tangent::Zero, Tangent::Zero]);
        }
        let window = self.window();
        let ty = dtype(input)?;
        let in_shape = dims(input)?;
        let mut program = Builder::new();
        let x = program.input(input.clone());
        let fill = zeros(&mut program, &info(&[], ty));
        let padding: Vec<_> = window
            .padding
            .iter()
            .zip(&window.base_dilation)
            .map(|(&(lo, hi), &base)| (lo, hi, base - 1))
            .collect();
        let padded = in_shape
            .iter()
            .zip(&padding)
            .map(|(&size, &(lo, hi, interior))| {
                let n = (size as i128)
                    .checked_add(
                        (size.saturating_sub(1) as i128)
                            .checked_mul(interior as i128)
                            .ok_or_else(|| {
                                TransformError::invalid("window padding extent overflow")
                            })?,
                    )
                    .and_then(|n| n.checked_add(lo as i128 + hi as i128))
                    .ok_or_else(|| TransformError::invalid("window padding extent overflow"))?;
                usize::try_from(n)
                    .map_err(|_| TransformError::invalid("window padding extent overflow"))
            })
            .collect::<TransformResult<Vec<_>>>()?;
        let x = program.emit1(pad(&padding), &[x, fill], info(&padded, ty));
        let rank = out_shape.len();
        let mut offset = vec![0usize; rank];
        let mut sum = None;
        loop {
            let ranges = (0..rank)
                .map(|axis| {
                    let start = offset[axis] * window.window_dilation[axis];
                    let stride = window.strides[axis];
                    (start, start + stride * (out_shape[axis] - 1) + 1, stride)
                })
                .collect::<Vec<_>>();
            let slice = program.emit1(slice(&ranges), &[x], output.clone());
            sum = Some(match sum {
                None => slice,
                Some(prev) => gnx_transforms::util::add(&mut program, prev, slice),
            });
            let Some(axis) = (0..rank)
                .rev()
                .find(|&axis| offset[axis] + 1 < window.dimensions[axis])
            else {
                break;
            };
            offset[axis] += 1;
            offset[axis + 1..].fill(0);
        }
        let (_, mut cts) = gnx_transforms::ad::transpose::transpose_expr(
            b,
            program
                .finish(vec![sum.expect("nonempty reduction window")])
                .build()?
                .view(),
            &[None],
            &[Tangent::Var(*ct)],
        )?;
        Ok(vec![cts.remove(0), Tangent::Zero])
    }
}

gnx_expr::register_op_interface!(dyn ReduceJvpRule: ReduceOperation => NonAdditiveReduce);

/// A handle to the reducer of a reduction.
fn reducer_handle(eqn: Eqn<'_>) -> TransformResult<gnx_expr::ExprHandle> {
    let region = gnx_transforms::util::single_region(eqn)
        .ok_or_else(|| TransformError::invalid("reduction needs one reducer region"))?;
    gnx_transforms::util::region_body(&gnx_expr::SubExprBuilder::new(region.to_ref()))
}

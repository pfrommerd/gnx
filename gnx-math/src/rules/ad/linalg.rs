use crate::ops::{
    broadcast_in_dim_op,
    contraction::{DotDimensions, dot_general},
    creation::{constant, iota},
    dtype::{conj_op, convert_element_type_op, real_op},
    elementwise::{div_op, eq_op, ge_op, gt_op, neg_op, sub_op},
    eye_op,
    indexing::{BoundsMode, GatherDimensions, GatherParameters, gather, slice},
    linalg::*,
    mul_op, ones_op, reshape_op,
    shape::{concatenate, pad},
};
use gnx_common::DType;
use gnx_expr::pass::Builder;
use gnx_expr::{Eqn, Op, Var};

use gnx_transforms::ad::jvp::instantiate;
use gnx_transforms::ad::{Dual, JvpRule, Operand, Tangent, TransposeRule};
use gnx_transforms::util::{
    add, copy_eqn, dims, dtype, info, reduce_sum, select, transpose, zeros,
};
use gnx_transforms::{TransformError, TransformResult};

type Result<T> = TransformResult<T>;

fn unary(b: &mut Builder, op: Op, x: Var) -> Var {
    b.emit1(op, &[x], b.info(x).clone())
}

fn binary(b: &mut Builder, op: Op, x: Var, y: Var) -> Var {
    b.emit1(op, &[x, y], b.info(x).clone())
}

fn shape(b: &Builder, x: Var) -> Result<Vec<usize>> {
    dims(b.info(x))
}

fn mul(b: &mut Builder, x: Var, y: Var) -> Var {
    binary(b, mul_op(), x, y)
}
fn sub(b: &mut Builder, x: Var, y: Var) -> Var {
    binary(b, sub_op(), x, y)
}
fn div(b: &mut Builder, x: Var, y: Var) -> Var {
    binary(b, div_op(), x, y)
}

fn complex(ty: DType) -> bool {
    matches!(ty, DType::C64 | DType::C128)
}
fn real_dtype(ty: DType) -> DType {
    match ty {
        DType::C64 => DType::F32,
        DType::C128 => DType::F64,
        ty => ty,
    }
}

fn convert(b: &mut Builder, x: Var, ty: DType) -> Result<Var> {
    if dtype(b.info(x))? == ty {
        return Ok(x);
    }
    Ok(b.emit1(convert_element_type_op(), &[x], info(&shape(b, x)?, ty)))
}

fn real(b: &mut Builder, x: Var) -> Result<Var> {
    let ty = dtype(b.info(x))?;
    if !complex(ty) {
        return Ok(x);
    }
    Ok(b.emit1(real_op(), &[x], info(&shape(b, x)?, real_dtype(ty))))
}

fn scalar(b: &mut Builder, value: f64, like: Var) -> Result<Var> {
    let ty = dtype(b.info(like))?;
    let scalar = b.emit1(constant(value), &[], info(&[], ty));
    Ok(b.emit1(broadcast_in_dim_op(&[]), &[scalar], b.info(like).clone()))
}

fn adjoint(b: &mut Builder, x: Var, conjugate: bool) -> Result<Var> {
    let rank = shape(b, x)?.len();
    if rank < 2 {
        return Err(TransformError::invalid("expected a matrix"));
    }
    let mut axes: Vec<_> = (0..rank).collect();
    axes.swap(rank - 2, rank - 1);
    let value = transpose(b, x, &axes)?;
    Ok(if conjugate && complex(dtype(b.info(x))?) {
        unary(b, conj_op(), value)
    } else {
        value
    })
}

fn matmul(b: &mut Builder, lhs: Var, rhs: Var) -> Result<Var> {
    let l = shape(b, lhs)?;
    let r = shape(b, rhs)?;
    let rank = l.len();
    if rank < 2 || r.len() != rank {
        return Err(TransformError::invalid("matrix product rank"));
    }
    let batch: Vec<_> = (0..rank - 2).collect();
    let d = DotDimensions {
        lhs_contracting: vec![rank - 1],
        rhs_contracting: vec![rank - 2],
        lhs_batch: batch.clone(),
        rhs_batch: batch,
    };
    let mut output = l;
    output[rank - 1] = r[rank - 1];
    Ok(b.emit1(
        dot_general(d),
        &[lhs, rhs],
        info(&output, dtype(b.info(lhs))?),
    ))
}

fn matrix_mask(b: &mut Builder, x: Var, compare: Op) -> Result<Var> {
    let shape = shape(b, x)?;
    let rank = shape.len();
    if rank < 2 {
        return Err(TransformError::invalid("expected a matrix"));
    }
    let ix = info(&shape, DType::I64);
    let row = b.emit1(iota(rank - 2), &[], ix.clone());
    let col = b.emit1(iota(rank - 1), &[], ix);
    Ok(b.emit1(compare, &[row, col], info(&shape, DType::Bool)))
}

fn triangle(b: &mut Builder, x: Var, lower: bool, diagonal: bool) -> Result<Var> {
    use crate::ops::elementwise::{le_op, lt_op};
    let pred = matrix_mask(
        b,
        x,
        match (lower, diagonal) {
            (true, true) => ge_op(),
            (true, false) => gt_op(),
            (false, true) => le_op(),
            (false, false) => lt_op(),
        },
    )?;
    let zero = zeros(b, &b.info(x).clone());
    Ok(select(b, pred, x, zero))
}

fn rectangular_eye(b: &mut Builder, like: Var) -> Result<Var> {
    let mask = matrix_mask(b, like, eq_op())?;
    convert(b, mask, dtype(b.info(like))?)
}

fn tri_solve(b: &mut Builder, a: Var, rhs: Var, options: TriangularSolveOptions) -> Var {
    b.emit1(triangular_solve(options), &[a, rhs], b.info(rhs).clone())
}

fn normal_tri(left_side: bool, lower: bool, unit_diagonal: bool) -> TriangularSolveOptions {
    TriangularSolveOptions {
        left_side,
        lower,
        unit_diagonal,
        transpose: false,
        conjugate: false,
    }
}

fn slice_axis(b: &mut Builder, x: Var, axis: usize, start: usize, end: usize) -> Result<Var> {
    let mut shape = shape(b, x)?;
    let mut ranges: Vec<_> = shape.iter().map(|&size| (0, size, 1)).collect();
    ranges[axis] = (start, end, 1);
    shape[axis] = end - start;
    Ok(b.emit1(slice(&ranges), &[x], info(&shape, dtype(b.info(x))?)))
}

fn padding(b: &mut Builder, x: Var, config: &[(i64, i64, usize)], result: &[usize]) -> Result<Var> {
    let ty = dtype(b.info(x))?;
    let zero = zeros(b, &info(&[], ty));
    Ok(b.emit1(pad(config), &[x, zero], info(result, ty)))
}

fn permute_axis(b: &mut Builder, x: Var, permutation: Var, axis: usize) -> Result<Var> {
    let shape = shape(b, x)?;
    let rank = shape.len();
    let mut index_shape = gnx_transforms::util::dims(b.info(permutation))?;
    let batch = index_shape.len() - 1;
    index_shape.push(1);
    let ind = b.emit1(
        reshape_op(),
        &[permutation],
        info(&index_shape, dtype(b.info(permutation))?),
    );
    let mut sizes = shape.clone();
    sizes[..batch].fill(1);
    sizes[axis] = 1;
    let output_axis = batch;
    let cfg = GatherParameters {
        dimensions: GatherDimensions {
            offset: (batch + 1..rank).collect(),
            collapsed: vec![axis],
            start_index_map: vec![axis],
            operand_batch: (0..batch).collect(),
            index_batch: (0..batch).collect(),
        },
        slice_sizes: sizes,
        mode: BoundsMode::PromiseInBounds,
        sorted: false,
        unique: true,
    };
    let mut out_shape = shape.clone();
    let axis_size = out_shape.remove(axis);
    out_shape.insert(output_axis, axis_size);
    let result = b.emit1(gather(cfg), &[x, ind], info(&out_shape, dtype(b.info(x))?));
    let mut axes: Vec<_> = (0..rank).collect();
    axes.remove(output_axis);
    axes.insert(axis, output_axis);
    transpose(b, result, &axes)
}

fn primal(b: &mut Builder, eqn: &Eqn, inputs: &[Dual]) -> Vec<Var> {
    let args: Vec<_> = inputs.iter().map(|d| d.primal).collect();
    copy_eqn(b, eqn, &args, &[])
}

fn tangent(b: &mut Builder, value: Dual) -> Var {
    let ty = b.info(value.primal).clone();
    instantiate(b, value.tangent, &ty)
}

fn dual(primal: Var, tangent: Var) -> Dual {
    Dual {
        primal,
        tangent: Tangent::Var(tangent),
    }
}

impl JvpRule for CholeskyOperation {
    fn jvp(
        &self,
        b: &mut Builder,
        eqn: &Eqn,
        inputs: &[Dual],
        _: &[Vec<Dual>],
    ) -> Result<Vec<Dual>> {
        let l = primal(b, eqn, inputs)[0];
        let dx = tangent(b, inputs[0]);
        let dx = if self.symmetrize() {
            let dxh = adjoint(b, dx, true)?;
            let sum = add(b, dx, dxh);
            let half = scalar(b, 0.5, dx)?;
            mul(b, sum, half)
        } else {
            let lower = triangle(b, dx, true, true)?;
            let strict_lower = triangle(b, dx, true, false)?;
            let other = adjoint(b, strict_lower, true)?;
            add(b, lower, other)
        };
        let options = TriangularSolveOptions {
            transpose: true,
            conjugate: true,
            ..normal_tri(false, true, false)
        };
        let left = tri_solve(b, l, dx, options);
        let s = tri_solve(b, l, left, normal_tri(true, true, false));
        let lower = triangle(b, s, true, true)?;
        let eye = rectangular_eye(b, s)?;
        let ones = b.emit1(ones_op(), &[], b.info(s).clone());
        let denominator = add(b, ones, eye);
        let phi = div(b, lower, denominator);
        Ok(vec![dual(l, matmul(b, l, phi)?)])
    }
}

impl JvpRule for TriangularSolveOperation {
    fn jvp(
        &self,
        b: &mut Builder,
        eqn: &Eqn,
        inputs: &[Dual],
        _: &[Vec<Dual>],
    ) -> Result<Vec<Dual>> {
        let x = primal(b, eqn, inputs)[0];
        let [a, rhs] = inputs else {
            return Err(TransformError::invalid("triangular solve signature"));
        };
        let options = self.options();
        let mut dx = rhs.tangent;
        if let Tangent::Var(da) = a.tangent {
            let mut da = triangle(b, da, options.lower, !options.unit_diagonal)?;
            if options.transpose {
                da = adjoint(b, da, options.conjugate)?;
            } else if options.conjugate && complex(dtype(b.info(da))?) {
                da = unary(b, conj_op(), da);
            }

            let mut xs = shape(b, x)?;
            let vector = xs.len() + 1 == shape(b, da)?.len();
            let x_matrix = if vector {
                let axis = if options.left_side {
                    xs.len()
                } else {
                    xs.len() - 1
                };
                xs.insert(axis, 1);
                b.emit1(reshape_op(), &[x], info(&xs, dtype(b.info(x))?))
            } else {
                x
            };
            let mut contribution = if options.left_side {
                matmul(b, da, x_matrix)?
            } else {
                matmul(b, x_matrix, da)?
            };
            if vector {
                contribution = b.emit1(reshape_op(), &[contribution], b.info(x).clone());
            }
            let negative = unary(b, neg_op(), contribution);
            dx = gnx_transforms::ad::jvp::accumulate(b, dx, Tangent::Var(negative));
        }
        let tangent = match dx {
            Tangent::Zero => Tangent::Zero,
            Tangent::Var(dx) => Tangent::Var(tri_solve(b, a.primal, dx, options)),
        };
        Ok(vec![Dual { primal: x, tangent }])
    }
}

impl TransposeRule for TriangularSolveOperation {
    fn transpose(
        &self,
        b: &mut Builder,
        _: &Eqn,
        args: &[Operand],
        _: &[Vec<Var>],
        cts: &[Tangent],
    ) -> Result<Vec<Tangent>> {
        let ([Operand::Residual(a), Operand::Linear(_)], [Tangent::Var(ct)]) = (args, cts) else {
            return Err(TransformError::invalid(
                "triangular solve transpose requires a residual matrix",
            ));
        };
        let mut options = self.options();
        options.transpose = !options.transpose;
        Ok(vec![
            Tangent::Zero,
            Tangent::Var(tri_solve(b, *a, *ct, options)),
        ])
    }
}

impl JvpRule for LuOperation {
    fn jvp(
        &self,
        b: &mut Builder,
        eqn: &Eqn,
        inputs: &[Dual],
        _: &[Vec<Dual>],
    ) -> Result<Vec<Dual>> {
        let out = primal(b, eqn, inputs);
        let [lu, pivots, perm] = out[..] else {
            return Err(TransformError::invalid("LU results"));
        };
        let dx = tangent(b, inputs[0]);
        let mut dims = shape(b, lu)?;
        let rank = dims.len();
        let (m, n) = (dims[rank - 2], dims[rank - 1]);
        let k = m.min(n);
        let dx = permute_axis(b, dx, perm, rank - 2)?;
        let l_rect = slice_axis(b, lu, rank - 1, 0, k)?;
        let l_rect = triangle(b, l_rect, true, false)?;
        let mut pad_l = vec![(0, 0, 0); rank];
        pad_l[rank - 1].1 = (m - k) as i64;
        dims[rank - 1] = m;
        let l_padded = padding(b, l_rect, &pad_l, &dims)?;
        let l_eye = rectangular_eye(b, l_padded)?;
        let l = add(b, l_padded, l_eye);
        let u_rect = slice_axis(b, lu, rank - 2, 0, k)?;
        let u_rect = triangle(b, u_rect, false, true)?;
        dims[rank - 1] = n;
        dims[rank - 2] = n;
        let mut pad_u = vec![(0, 0, 0); rank];
        pad_u[rank - 2].1 = (n - k) as i64;
        let u_padded = padding(b, u_rect, &pad_u, &dims)?;
        let tail_eye = {
            dims[rank - 1] = n - k;
            dims[rank - 2] = n - k;
            let small = b.emit1(eye_op(), &[], info(&[n - k, n - k], dtype(b.info(lu))?));
            let map = [rank - 2, rank - 1];
            let small = b.emit1(
                broadcast_in_dim_op(&map),
                &[small],
                info(&dims, dtype(b.info(lu))?),
            );
            pad_u[rank - 2] = (k as i64, 0, 0);
            pad_u[rank - 1] = (k as i64, 0, 0);
            dims[rank - 1] = n;
            dims[rank - 2] = n;
            padding(b, small, &pad_u, &dims)?
        };
        let u = add(b, u_padded, tail_eye);
        let left = tri_solve(b, l, dx, normal_tri(true, true, true));
        let split = tri_solve(b, u, left, normal_tri(false, false, false));
        let l_tangent = triangle(b, split, true, false)?;
        let u_tangent = triangle(b, split, false, true)?;
        let l_tangent = matmul(b, l, l_tangent)?;
        let u_tangent = matmul(b, u_tangent, u)?;
        let gradient = add(b, l_tangent, u_tangent);
        Ok(vec![
            dual(lu, gradient),
            Dual::constant(pivots),
            Dual::constant(perm),
        ])
    }
}

fn diagonal(b: &mut Builder, x: Var) -> Result<Var> {
    let shape = shape(b, x)?;
    let rank = shape.len();
    let n = shape[rank - 1].min(shape[rank - 2]);
    let idx = b.emit1(iota(0), &[], info(&[n, 1], DType::I32));
    let indices = b.emit1(concatenate(1), &[idx, idx], info(&[n, 2], DType::I32));
    let mut slices = shape.clone();
    slices[rank - 2..].fill(1);
    let op = gather(GatherParameters {
        dimensions: GatherDimensions {
            offset: (0..rank - 2).collect(),
            collapsed: vec![rank - 2, rank - 1],
            start_index_map: vec![rank - 2, rank - 1],
            operand_batch: vec![],
            index_batch: vec![],
        },
        slice_sizes: slices,
        mode: BoundsMode::PromiseInBounds,
        sorted: true,
        unique: true,
    });
    let mut output = shape;
    output.truncate(rank - 2);
    output.push(n);
    Ok(b.emit1(op, &[x, indices], info(&output, dtype(b.info(x))?)))
}

fn across_matrix(
    b: &mut Builder,
    vector: Var,
    shape: &[usize],
    dropped_axis: usize,
) -> Result<Var> {
    let map: Vec<_> = (0..shape.len())
        .filter(|&axis| axis != dropped_axis)
        .collect();
    Ok(b.emit1(
        broadcast_in_dim_op(&map),
        &[vector],
        info(shape, dtype(b.info(vector))?),
    ))
}

impl JvpRule for EighOperation {
    fn jvp(
        &self,
        b: &mut Builder,
        eqn: &Eqn,
        inputs: &[Dual],
        _: &[Vec<Dual>],
    ) -> Result<Vec<Dual>> {
        let out = primal(b, eqn, inputs);
        let [vectors, values] = out[..] else {
            return Err(TransformError::invalid("eigh results"));
        };
        let da = tangent(b, inputs[0]);
        let vh = adjoint(b, vectors, true)?;
        let projected = matmul(b, vh, da)?;
        let projected = matmul(b, projected, vectors)?;
        let real = real(b, projected)?;
        let dw = diagonal(b, real)?;
        let shape = shape(b, vectors)?;
        let rank = shape.len();
        let cols = across_matrix(b, values, &shape, rank - 2)?;
        let rows = across_matrix(b, values, &shape, rank - 1)?;
        let delta = sub(b, cols, rows);
        let eye = rectangular_eye(b, delta)?;
        let adjusted = add(b, delta, eye);
        let one = b.emit1(ones_op(), &[], b.info(delta).clone());
        let f = div(b, one, adjusted);
        let f = sub(b, f, eye);
        let f = convert(b, f, dtype(b.info(projected))?)?;
        let weight = mul(b, f, projected);
        let dv = matmul(b, vectors, weight)?;
        Ok(vec![dual(vectors, dv), dual(values, dw)])
    }
}

gnx_expr::register_op_interface!(dyn JvpRule: CholeskyOperation, TriangularSolveOperation, LuOperation, EighOperation);
gnx_expr::register_op_interface!(dyn TransposeRule: TriangularSolveOperation);

fn dense_solve(b: &mut Builder, a: Var, rhs: Var) -> Result<Var> {
    let dimensions = shape(b, a)?;
    let rank = dimensions.len();
    let pivot = info(&dimensions[..rank - 1], DType::I32);
    let lu = b.emit(
        lu(),
        &[a],
        [b.info(a).clone(), pivot.clone(), pivot],
        vec![],
    );
    let rhs = permute_axis(b, rhs, lu[2], rank - 2)?;
    let y = tri_solve(b, lu[0], rhs, normal_tri(true, true, true));
    Ok(tri_solve(b, lu[0], y, normal_tri(true, false, false)))
}

fn thin_qr(b: &mut Builder, q: Var, r: Var, da: Var) -> Result<(Var, Var)> {
    let dxrinv = tri_solve(b, r, da, normal_tri(false, false, false));
    let qh = adjoint(b, q, true)?;
    let projected = matmul(b, qh, dxrinv)?;
    let lower = triangle(b, projected, true, false)?;
    let trans = adjoint(b, lower, true)?;
    let mut rotation = sub(b, lower, trans);
    let ty = dtype(b.info(projected))?;
    if complex(ty) {
        let real = real(b, projected)?;
        let real = convert(b, real, ty)?;
        let imaginary = sub(b, projected, real);
        let eye = rectangular_eye(b, projected)?;
        let diagonal = mul(b, eye, imaginary);
        rotation = add(b, rotation, diagonal);
    }
    let diff = sub(b, rotation, projected);
    let dprojected = matmul(b, q, diff)?;
    let dq = add(b, dprojected, dxrinv);
    let negative = unary(b, neg_op(), diff);
    let dr = matmul(b, negative, r)?;
    Ok((dq, dr))
}

impl JvpRule for QrOperation {
    fn jvp(
        &self,
        b: &mut Builder,
        eqn: &Eqn,
        inputs: &[Dual],
        _: &[Vec<Dual>],
    ) -> Result<Vec<Dual>> {
        let outputs = primal(b, eqn, inputs);
        let (q, r) = (outputs[0], outputs[1]);
        let mut x = inputs[0].primal;
        let mut dx = tangent(b, inputs[0]);
        let shape = shape(b, x)?;
        let rank = shape.len();
        let (m, n) = (shape[rank - 2], shape[rank - 1]);
        let (dq, dr) = if m == 0 || n == 0 {
            (zeros(b, &b.info(q).clone()), zeros(b, &b.info(r).clone()))
        } else {
            if self.pivoting() {
                x = permute_axis(b, x, outputs[2], rank - 1)?;
                dx = permute_axis(b, dx, outputs[2], rank - 1)?;
            }
            if m < n {
                let x2 = slice_axis(b, x, rank - 1, m, n)?;
                let dx1 = slice_axis(b, dx, rank - 1, 0, m)?;
                let dx2 = slice_axis(b, dx, rank - 1, m, n)?;
                let r1 = slice_axis(b, r, rank - 1, 0, m)?;
                let (dq, dr1) = thin_qr(b, q, r1, dx1)?;
                let dqh = adjoint(b, dq, true)?;
                let dr2a = matmul(b, dqh, x2)?;
                let qh = adjoint(b, q, true)?;
                let dr2b = matmul(b, qh, dx2)?;
                let dr2 = add(b, dr2a, dr2b);
                (
                    dq,
                    b.emit1(concatenate(rank - 1), &[dr1, dr2], b.info(r).clone()),
                )
            } else if self.full() && m > n {
                let qmn = slice_axis(b, q, rank - 1, 0, n)?;
                let rnn = slice_axis(b, r, rank - 2, 0, n)?;
                let (dqmn, drnn) = thin_qr(b, qmn, rnn, dx)?;
                let qnn = slice_axis(b, qmn, rank - 2, 0, n)?;
                let qpn = slice_axis(b, qmn, rank - 2, n, m)?;
                let qmp = slice_axis(b, q, rank - 1, n, m)?;
                let dqnn = slice_axis(b, dqmn, rank - 2, 0, n)?;
                let dqpn = slice_axis(b, dqmn, rank - 2, n, m)?;
                let eye = rectangular_eye(b, qnn)?;
                let sq = sub(b, qnn, eye);
                let square_h = adjoint(b, sq, true)?;
                let qpn_h = adjoint(b, qpn, true)?;
                let zh = dense_solve(b, square_h, qpn_h)?;
                let z = adjoint(b, zh, true)?;
                let first = matmul(b, dqmn, zh)?;
                let qmpz = matmul(b, qmp, z)?;
                let u = add(b, qmn, qmpz);
                let zdq = matmul(b, z, dqnn)?;
                let v = sub(b, dqpn, zdq);
                let vh = adjoint(b, v, true)?;
                let second = matmul(b, u, vh)?;
                let dqmp = sub(b, first, second);
                let dq = b.emit1(concatenate(rank - 1), &[dqmn, dqmp], b.info(q).clone());
                let mut padding_config = vec![(0, 0, 0); rank];
                padding_config[rank - 2].1 = (m - n) as i64;
                let dr = padding(b, drnn, &padding_config, &shape)?;
                (dq, dr)
            } else {
                thin_qr(b, q, r, dx)?
            }
        };
        let mut values = vec![dual(q, dq), dual(r, dr)];
        if self.pivoting() {
            values.push(Dual::constant(outputs[2]));
        }
        Ok(values)
    }
}

impl JvpRule for SvdOperation {
    fn jvp(
        &self,
        b: &mut Builder,
        _eqn: &Eqn,
        inputs: &[Dual],
        _: &[Vec<Dual>],
    ) -> Result<Vec<Dual>> {
        let a = inputs[0].primal;
        let shape = shape(b, a)?;
        let rank = shape.len();
        let (m, n) = (shape[rank - 2], shape[rank - 1]);
        let k = m.min(n);
        if self.full() && self.vectors() && m != n {
            return Err(TransformError::invalid(
                "full rectangular SVD does not support JVP",
            ));
        }
        let ty = dtype(b.info(a))?;
        let d = &shape[..rank - 2];
        let matrix = |row, col| info(&d.iter().copied().chain([row, col]).collect::<Vec<_>>(), ty);
        let vector = info(
            &d.iter().copied().chain([k]).collect::<Vec<_>>(),
            real_dtype(ty),
        );
        let results = b.emit(
            svd(false, true),
            &[a],
            [matrix(m, k), vector, matrix(k, n)],
            vec![],
        );
        let (u, s, vt) = (results[0], results[1], results[2]);
        let vu = adjoint(b, u, true)?;
        let v = adjoint(b, vt, true)?;
        let da = tangent(b, inputs[0]);
        let dv_a = matmul(b, vu, da)?;
        let projected = matmul(b, dv_a, v)?;
        let real = real(b, projected)?;
        let ds = diagonal(b, real)?;
        if !self.vectors() {
            return Ok(vec![dual(s, ds)]);
        }

        let mut square_shape = shape;
        square_shape[rank - 2] = k;
        square_shape[rank - 1] = k;
        let sc = across_matrix(b, s, &square_shape, rank - 2)?;
        let sr = across_matrix(b, s, &square_shape, rank - 1)?;
        let sum = add(b, sc, sr);
        let diff = sub(b, sc, sr);
        let diff = mul(b, diff, sum);
        let eye = rectangular_eye(b, diff)?;
        let safe = add(b, diff, eye);
        let ones = b.emit1(ones_op(), &[], b.info(safe).clone());
        let f = div(b, ones, safe);
        let f = sub(b, f, eye);
        let f = convert(b, f, ty)?;
        let xc = convert(b, sc, ty)?;
        let xr = convert(b, sr, ty)?;
        let ds_times_s = mul(b, xc, projected);
        let s_times_ds = mul(b, xr, projected);
        let ds_h = adjoint(b, projected, true)?;
        let mut skew = sub(b, projected, ds_h);
        let zero = zeros(b, &b.info(sc).clone());
        let equal = b.emit1(eq_op(), &[sc, zero], info(&square_shape, DType::Bool));
        let equal = convert(b, equal, real_dtype(ty))?;
        let sc_safe = add(b, sc, equal);
        let reciprocal = div(b, ones, sc_safe);
        let reciprocal = sub(b, reciprocal, equal);
        let half = scalar(b, 0.5, reciprocal)?;
        let diag_scale = mul(b, reciprocal, half);
        let diag_scale = mul(b, diag_scale, eye);
        let diag_scale = convert(b, diag_scale, ty)?;
        skew = mul(b, skew, diag_scale);
        let h1 = adjoint(b, ds_times_s, true)?;
        let h2 = adjoint(b, s_times_ds, true)?;
        let y1 = add(b, ds_times_s, h1);
        let y2 = add(b, s_times_ds, h2);
        let weighted1 = mul(b, f, y1);
        let weighted2 = mul(b, f, y2);
        let weighted1 = add(b, weighted1, skew);
        let mut du = matmul(b, u, weighted1)?;
        let mut dv = matmul(b, v, weighted2)?;
        if m > n {
            let dav = matmul(b, da, v)?;
            let projected = matmul(b, vu, dav)?;
            let back = matmul(b, u, projected)?;
            let remainder = sub(b, dav, back);
            let mut scale_shape = square_shape.clone();
            scale_shape[rank - 2] = m;
            let scale = across_matrix(b, s, &scale_shape, rank - 2)?;
            let scale = convert(b, scale, ty)?;
            let correction = div(b, remainder, scale);
            du = add(b, du, correction);
        } else if n > m {
            let dah = adjoint(b, da, true)?;
            let dahu = matmul(b, dah, u)?;
            let projected = matmul(b, vt, dahu)?;
            let back = matmul(b, v, projected)?;
            let remainder = sub(b, dahu, back);
            square_shape[rank - 2] = n;
            let scale = across_matrix(b, s, &square_shape, rank - 2)?;
            let scale = convert(b, scale, ty)?;
            let correction = div(b, remainder, scale);
            dv = add(b, dv, correction);
        }
        Ok(vec![
            dual(u, du),
            dual(s, ds),
            dual(vt, adjoint(b, dv, true)?),
        ])
    }
}

fn complex_part(b: &mut Builder, op: Op, x: Var) -> Result<Var> {
    let dtype = real_dtype(dtype(b.info(x))?);
    let shape = shape(b, x)?;
    Ok(b.emit1(op, &[x], info(&shape, dtype)))
}

fn eigvec_jvp(b: &mut Builder, w: Var, v: Var, da: Var) -> Result<(Var, Var)> {
    let shape = shape(b, v)?;
    let rank = shape.len();
    let row = rank - 2;
    let along_columns = across_matrix(b, w, &shape, row)?;
    let along_rows = across_matrix(b, w, &shape, rank - 1)?;
    let eye = rectangular_eye(b, v)?;
    let delta = sub(b, along_columns, along_rows);
    let delta = add(b, delta, eye);
    let one = b.emit1(ones_op(), &[], b.info(delta).clone());
    let factor = div(b, one, delta);
    let factor = sub(b, factor, eye);
    let projection = dense_solve(b, v, da)?;
    let projection = matmul(b, projection, v)?;
    let dw = diagonal(b, projection)?;
    let correction = mul(b, factor, projection);
    let correction = matmul(b, v, correction)?;
    let conj_v = unary(b, conj_op(), v);
    let norm = mul(b, conj_v, correction);
    let sum = reduce_sum(b, norm, &[row])?;
    let real = complex_part(b, real_op(), sum)?;
    let norm = unary(b, neg_op(), real);
    let magnitude = complex_part(b, crate::ops::elementwise::abs_op(), v)?;
    let maxima = crate::ops::reduction::argmax(b, magnitude, row, DType::I32)?;
    let maxima = across_matrix(b, maxima, &shape, row)?;
    let index = b.emit1(iota(row), &[], info(&shape, DType::I32));
    let mask = b.emit1(eq_op(), &[maxima, index], info(&shape, DType::Bool));
    let mask = convert(b, mask, dtype(b.info(v))?)?;
    let pivot = mul(b, mask, v);
    let pivot_c = mul(b, mask, correction);
    let pivot = reduce_sum(b, pivot, &[row])?;
    let pivot_c = reduce_sum(b, pivot_c, &[row])?;
    let re = complex_part(b, real_op(), pivot)?;
    let im = complex_part(b, crate::ops::dtype::imag_op(), pivot_c)?;
    let phase = div(b, im, re);
    let phase = unary(b, neg_op(), phase);
    let c = b.emit1(
        crate::ops::dtype::complex_op(),
        &[norm, phase],
        b.info(w).clone(),
    );
    let c = across_matrix(b, c, &shape, row)?;
    let correction2 = mul(b, v, c);
    Ok((dw, add(b, correction, correction2)))
}

impl JvpRule for EigOperation {
    fn jvp(
        &self,
        b: &mut Builder,
        eqn: &Eqn,
        inputs: &[Dual],
        _: &[Vec<Dual>],
    ) -> Result<Vec<Dual>> {
        let vectors = self.left() || self.right();
        if vectors && !self.differentiate_vectors() {
            return Err(TransformError::invalid(
                "general eigenvector derivatives require an explicit opt-in",
            ));
        }
        let a = inputs[0].primal;
        let da = tangent(b, inputs[0]);
        let ty = dtype(&eqn.outputs()[0].1)?;
        let info_a = info(&shape(b, a)?, ty);
        let mut outputs = vec![eqn.outputs()[0].1.clone(), info_a.clone()];
        if self.left() {
            outputs.push(info_a);
        }
        let out = b.emit(eig(self.left(), true), &[a], outputs, vec![]);
        let values = out[0];
        let right = *out.last().expect("right eigenvectors");
        let da = convert(b, da, ty)?;
        if !vectors {
            let projected = dense_solve(b, right, da)?;
            let projected = matmul(b, projected, right)?;
            return Ok(vec![dual(values, diagonal(b, projected)?)]);
        }
        let (dw, dvr) = eigvec_jvp(b, values, right, da)?;
        let mut result = vec![dual(values, dw)];
        if self.left() {
            let left = out[1];
            let w = unary(b, conj_op(), values);
            let a = adjoint(b, da, true)?;
            let (_, dvl) = eigvec_jvp(b, w, left, a)?;
            result.push(dual(left, dvl));
        }
        if self.right() {
            result.push(dual(right, dvr));
        }
        Ok(result)
    }
}

gnx_expr::register_op_interface!(dyn JvpRule: QrOperation, SvdOperation, EigOperation);

fn diagonal_product(b: &mut Builder, dl: Var, d: Var, du: Var, x: Var) -> Result<Var> {
    let dims = shape(b, x)?;
    let rank = dims.len();
    let axis = rank - 2;
    let n = dims[axis];
    let result = across_matrix(b, d, &dims, rank - 1)?;
    let result = mul(b, result, x);
    if n < 2 {
        return Ok(result);
    }
    let low = slice_axis(b, dl, axis, 1, n)?;
    let high = slice_axis(b, du, axis, 0, n - 1)?;
    let initial = slice_axis(b, x, axis, 0, n - 1)?;
    let rest = slice_axis(b, x, axis, 1, n)?;
    let mut reduced = dims.clone();
    reduced[axis] -= 1;
    let low = across_matrix(b, low, &reduced, rank - 1)?;
    let high = across_matrix(b, high, &reduced, rank - 1)?;
    let l = mul(b, low, initial);
    let u = mul(b, high, rest);
    let mut pad = vec![(0, 0, 0); rank];
    pad[axis] = (1, 0, 0);
    let l = padding(b, l, &pad, &dims)?;
    pad[axis] = (0, 1, 0);
    let u = padding(b, u, &pad, &dims)?;
    let out = add(b, l, u);
    Ok(add(b, result, out))
}

impl JvpRule for TridiagonalSolveOperation {
    fn jvp(
        &self,
        b: &mut Builder,
        eqn: &Eqn,
        inputs: &[Dual],
        _: &[Vec<Dual>],
    ) -> Result<Vec<Dual>> {
        let [dl, d, du, rhs] = inputs else {
            return Err(TransformError::invalid("tridiagonal solve signature"));
        };
        let x = primal(b, eqn, inputs)[0];
        let mut drhs = tangent(b, *rhs);
        if [dl, d, du].iter().any(|v| !v.tangent.is_zero()) {
            let (l, m, u) = (tangent(b, *dl), tangent(b, *d), tangent(b, *du));
            let product = diagonal_product(b, l, m, u, x)?;
            drhs = sub(b, drhs, product);
        }
        let out = b.emit1(
            eqn.op().clone(),
            &[dl.primal, d.primal, du.primal, drhs],
            b.info(x).clone(),
        );
        Ok(vec![dual(x, out)])
    }
}

impl TransposeRule for TridiagonalSolveOperation {
    fn transpose(
        &self,
        b: &mut Builder,
        eqn: &Eqn,
        operands: &[Operand],
        _: &[Vec<Var>],
        cts: &[Tangent],
    ) -> Result<Vec<Tangent>> {
        let (
            [
                Operand::Residual(dl),
                Operand::Residual(d),
                Operand::Residual(du),
                Operand::Linear(rhs),
            ],
            [Tangent::Var(ct)],
        ) = (operands, cts)
        else {
            return Err(TransformError::invalid(
                "tridiagonal transpose requires residual diagonals",
            ));
        };
        let shape = shape(b, *d)?;
        let axis = shape.len() - 1;
        let n = shape[axis];
        let (lower, upper) = if n == 0 {
            (*du, *dl)
        } else {
            let low = slice_axis(b, *du, axis, 0, n - 1)?;
            let up = slice_axis(b, *dl, axis, 1, n)?;
            let mut pad = vec![(0, 0, 0); shape.len()];
            pad[axis] = (1, 0, 0);
            let low = padding(b, low, &pad, &shape)?;
            pad[axis] = (0, 1, 0);
            let up = padding(b, up, &pad, &shape)?;
            (low, up)
        };
        let x = b.emit1(eqn.op().clone(), &[lower, *d, upper, *ct], rhs.clone());
        Ok(vec![
            Tangent::Zero,
            Tangent::Zero,
            Tangent::Zero,
            Tangent::Var(x),
        ])
    }
}
gnx_expr::register_op_interface!(dyn JvpRule: TridiagonalSolveOperation);
gnx_expr::register_op_interface!(dyn TransposeRule: TridiagonalSolveOperation);

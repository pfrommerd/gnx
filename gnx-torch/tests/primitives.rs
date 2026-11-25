use gnx_common::pass::Builder;
use gnx_expr::{Op, VarInfo};
use gnx_math::{
    ArrayInfo, DType, Shape,
    ops::{bitwise, contraction::*, elementwise as ew, fft::*, indexing, shape, special},
};
use gnx_torch::TorchDevice;
use tch::{Kind, Tensor};

fn ty(k: DType, dims: &[usize]) -> VarInfo {
    VarInfo::new(ArrayInfo::new(Shape::fixed(dims.iter().copied()), k))
}
fn program(op: Op, args: &[VarInfo], out: VarInfo) -> gnx_torch::TorchExecutable {
    let mut b = Builder::new();
    let args: Vec<_> = args.iter().map(|t| b.input(t.clone())).collect();
    let out = b.emit1(op, &args, out);
    futures_executor::block_on(
        TorchDevice::cpu(gnx_common::cache::Cache::in_memory())
            .unwrap()
            .compile(&b.finish(vec![out]).build().unwrap()),
    )
    .unwrap()
}
fn run(op: Op, args: &[(&Tensor, DType)], shape: &[usize], dtype: DType) -> Tensor {
    let ins: Vec<_> = args
        .iter()
        .map(|(t, k)| {
            ty(
                *k,
                &t.size().iter().map(|&i| i as usize).collect::<Vec<_>>(),
            )
        })
        .collect();
    let p = program(op, &ins, ty(dtype, shape));
    p.modules()
        .next()
        .unwrap()
        .forward_ts(&args.iter().map(|(t, _)| *t).collect::<Vec<_>>())
        .unwrap()
}
#[test]
fn single_cpu_accepts_replicated_but_not_named_sharding() {
    use gnx_math::shard::{Sharding, constraint_op, replicate_op};
    let x = Tensor::from_slice(&[1i32, 5]);
    for op in [replicate_op(), constraint_op(Sharding::replicated(1))] {
        assert!(run(op, &[(&x, DType::I32)], &[2], DType::I32).equal(&x));
    }
    let mut b = Builder::new();
    let t = ty(DType::I32, &[2]);
    let arg = b.input(t.clone());
    let y = b.emit1(
        constraint_op(Sharding::new(vec![vec!["devices".into()]]).unwrap()),
        &[arg],
        t,
    );
    assert_eq!(
        futures_executor::block_on(
            TorchDevice::cpu(gnx_common::cache::Cache::in_memory())
                .unwrap()
                .compile(&b.finish(vec![y]).build().unwrap())
        )
        .err()
        .unwrap()
        .kind(),
        std::io::ErrorKind::Unsupported
    );
}

#[test]
fn bit_patterns_and_high_products_match_rust() {
    let a = [i64::MIN, -1, -7, i64::MAX, i64::MAX - 17, 0];
    let b = [-1i64, 1, i64::MAX, i64::MIN, i64::MAX - 15, 1];
    let ta = Tensor::from_slice(&a);
    let tb = Tensor::from_slice(&b);
    let n = a.len();
    let args = [(&ta, DType::I64), (&tb, DType::I64)];
    let hi = run(ew::mulhi(), &args, &[n], DType::I64);
    assert_eq!(
        Vec::<i64>::try_from(hi).unwrap(),
        a.iter()
            .zip(b)
            .map(|(&x, y)| ((i128::from(x) * i128::from(y)) >> 64) as i64)
            .collect::<Vec<_>>()
    );
    for (op, expected) in [
        (bitwise::clz(), a.map(|x| x.leading_zeros() as i64)),
        (
            bitwise::population_count(),
            a.map(|x| x.count_ones() as i64),
        ),
    ] {
        assert_eq!(
            Vec::<i64>::try_from(run(op, &args[..1], &[n], DType::I64)).unwrap(),
            expected
        );
    }
    let counts = [0i64, 1, 63, 64, 65, -1];
    let tc = Tensor::from_slice(&counts);
    assert_eq!(
        Vec::<i64>::try_from(run(
            bitwise::shift_right_logical(),
            &[args[0], (&tc, DType::I64)],
            &[n],
            DType::I64
        ))
        .unwrap(),
        std::array::from_fn::<_, 6, _>(|i| if (0..64).contains(&counts[i]) {
            ((a[i] as u64) >> counts[i]) as i64
        } else {
            0
        })
    );
}
#[test]
fn dynamic_slice_and_update_wrap_then_clamp_on_replay() {
    let i = DType::I32;
    let r = program(
        indexing::dynamic_slice(),
        &[ty(i, &[2, 4]), ty(i, &[]), ty(i, &[])],
        ty(i, &[1, 2]),
    );
    let w = program(
        indexing::dynamic_update_slice(),
        &[ty(i, &[2, 4]), ty(i, &[1, 2]), ty(i, &[]), ty(i, &[])],
        ty(i, &[2, 4]),
    );
    let x = Tensor::arange(8, (Kind::Int, tch::Device::Cpu)).reshape([2, 4]);
    let v = Tensor::from_slice(&[11i32, 12]).reshape([1, 2]);
    for (row, col, pos) in [(0i32, 1i32, 1usize), (-1, -1, 6), (9, -9, 4)] {
        let row = Tensor::from(row);
        let col = Tensor::from(col);
        let actual = r
            .modules()
            .next()
            .unwrap()
            .forward_ts(&[&x, &row, &col])
            .unwrap();
        assert_eq!(
            Vec::<i32>::try_from(actual.reshape([-1])).unwrap(),
            [pos as i32, pos as i32 + 1]
        );
        let actual = w
            .modules()
            .next()
            .unwrap()
            .forward_ts(&[&x, &v, &row, &col])
            .unwrap();
        let mut expected: Vec<i32> = (0..8).collect();
        expected[pos..pos + 2].copy_from_slice(&[11, 12]);
        assert_eq!(
            Vec::<i32>::try_from(actual.reshape([-1])).unwrap(),
            expected
        );
    }
}
#[test]
fn padding_crops_and_inserts_live_fill() {
    let x = Tensor::from_slice(&[1i32, 2, 3]);
    let y = Tensor::from(9i32);
    let out = run(
        shape::pad(&[(-1, 2, 1)]),
        &[(&x, DType::I32), (&y, DType::I32)],
        &[6],
        DType::I32,
    );
    assert_eq!(Vec::<i32>::try_from(out).unwrap(), [9, 2, 9, 3, 9, 9]);
}
#[test]
fn grouped_convolution_respects_layout_and_asymmetric_padding() {
    let x = Tensor::from_slice(&[1f32, 4., 2., 5., 3., 6.]).reshape([1, 3, 2]);
    let w = Tensor::from_slice(&[2f32, 3., 1., 2.]).reshape([2, 1, 2]);
    let layout = ConvAxes {
        batch: 0,
        feature: 2,
        spatial: vec![1],
    };
    let p = ConvParameters {
        input: layout.clone(),
        output: layout,
        kernel: ConvKernelAxes {
            output_feature: 0,
            input_feature: 1,
            spatial: vec![2],
        },
        strides: vec![1],
        padding: vec![(1, 0)],
        lhs_dilation: vec![1],
        rhs_dilation: vec![1],
        feature_groups: 2,
        batch_groups: 1,
    };
    let y = run(
        conv_general_dilated(p),
        &[(&x, DType::F32), (&w, DType::F32)],
        &[1, 3, 2],
        DType::F32,
    );
    assert_eq!(
        Vec::<f32>::try_from(y.reshape([-1])).unwrap(),
        [3., 8., 8., 14., 13., 17.]
    );
}
#[test]
fn scan_replays_live_carry_and_stacks_outputs() {
    let mut b = Builder::new();
    let s = ty(DType::F32, &[]);
    let row = ty(DType::F32, &[3]);
    let initial = b.input(s.clone());
    let xs = b.input(row.clone());
    let body = b
        .region::<std::io::Error>(&[], &[s.clone(), s.clone()], |b, _, x| {
            let sum = b.emit1(gnx_math::ops::add_op(), x, s.clone());
            Ok(vec![sum, sum])
        })
        .unwrap();
    let ys = b.emit(
        gnx_transforms::cf::scan_op(1, 3),
        &[initial, xs],
        [s, row],
        vec![body],
    );
    let p = futures_executor::block_on(
        TorchDevice::cpu(gnx_common::cache::Cache::in_memory())
            .unwrap()
            .compile(&b.finish(ys).build().unwrap()),
    )
    .unwrap();
    let args = [Tensor::from(3f32), Tensor::from_slice(&[2f32, -1., 5.])];
    let vals = p
        .modules()
        .map(|m| m.forward_ts(&args).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(f32::try_from(&vals[0]).unwrap(), 9.);
    assert_eq!(Vec::<f32>::try_from(&vals[1]).unwrap(), [5., 4., 9.]);
}
#[test]
fn dot_general_pairs_nonleading_batch_and_contraction_dimensions() {
    let a = Tensor::arange(12, (Kind::Float, tch::Device::Cpu)).reshape([2, 2, 3]);
    let b = Tensor::arange(12, (Kind::Float, tch::Device::Cpu)).reshape([3, 2, 2]);
    let out = run(
        dot_general(DotDimensions {
            lhs_batch: vec![1],
            rhs_batch: vec![1],
            lhs_contracting: vec![2],
            rhs_contracting: vec![0],
        }),
        &[(&a, DType::F32), (&b, DType::F32)],
        &[2, 2, 2],
        DType::F32,
    );
    let mut expected = vec![];
    for batch in 0..2 {
        for row in 0..2 {
            for col in 0..2 {
                expected.push(
                    (0..3)
                        .map(|k| ((row * 6 + batch * 3 + k) * (k * 4 + batch * 2 + col)) as f32)
                        .sum::<f32>(),
                );
            }
        }
    }
    assert_eq!(Vec::<f32>::try_from(out.reshape([-1])).unwrap(), expected);
}
#[test]
fn cholesky_lower_factor_and_pivoted_qr_reconstruct_inputs() {
    use gnx_math::ops::linalg;
    let x = Tensor::from_slice(&[4f32, 2., 2., 5.]).reshape([2, 2]);
    let l = run(
        linalg::cholesky(false),
        &[(&x, DType::F32)],
        &[2, 2],
        DType::F32,
    );
    assert_eq!(
        Vec::<f32>::try_from(l.reshape([-1])).unwrap(),
        [2., 0., 1., 2.]
    );
    let mut b = Builder::new();
    let mat = ty(DType::F32, &[2, 2]);
    let a = b.input(mat.clone());
    let outputs = b.emit(
        linalg::qr(false, true),
        &[a],
        [mat.clone(), mat, ty(DType::I32, &[2])],
        vec![],
    );
    let p = futures_executor::block_on(
        TorchDevice::cpu(gnx_common::cache::Cache::in_memory())
            .unwrap()
            .compile(&b.finish(outputs).build().unwrap()),
    )
    .unwrap();
    let out = p
        .modules()
        .map(|m| m.forward_ts(&[&x]).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(Vec::<i32>::try_from(&out[2]).unwrap(), [1, 0]);
    assert!(out[0].matmul(&out[1]).allclose(
        &x.index_select(1, &out[2].to_kind(Kind::Int64)),
        1e-5,
        1e-5,
        false
    ));
}
#[test]
fn gather_fill_and_scatter_duplicates_with_out_of_bounds_indices() {
    use indexing::*;
    let x = Tensor::from_slice(&[1i32, 2, 3, 4]);
    let ix = Tensor::from_slice(&[-1i32, 1, 1, 4]).reshape([4, 1]);
    let fill = Tensor::from(99i32);
    let i = DType::I32;
    let g = gather(GatherParameters {
        dimensions: GatherDimensions {
            offset: vec![1],
            collapsed: vec![],
            start_index_map: vec![0],
            operand_batch: vec![],
            index_batch: vec![],
        },
        slice_sizes: vec![2],
        mode: BoundsMode::FillOrDrop,
        sorted: false,
        unique: false,
    });
    let result = run(g, &[(&x, i), (&ix, i), (&fill, i)], &[4, 2], i);
    assert_eq!(
        Vec::<i32>::try_from(result.reshape([-1])).unwrap(),
        [99, 99, 2, 3, 2, 3, 99, 99]
    );
    let s = scatter_add(ScatterParameters {
        dimensions: ScatterDimensions {
            update_window: vec![],
            inserted_window: vec![0],
            index_map: vec![0],
            operand_batch: vec![],
            index_batch: vec![],
        },
        mode: BoundsMode::FillOrDrop,
        sorted: false,
        unique: false,
    });
    let result = run(s, &[(&x, i), (&ix, i), (&x, i)], &[4], i);
    assert_eq!(Vec::<i32>::try_from(result).unwrap(), [1, 7, 3, 4]);
}
#[test]
fn stable_lexicographic_sort_permutates_payloads() {
    use gnx_math::ops::ordering;
    let mut b = Builder::new();
    let row = ty(DType::I32, &[5]);
    let args = std::array::from_fn::<_, 3, _>(|_| b.input(row.clone()));
    let outputs = b.emit(
        ordering::sort(0, 2, true),
        &args,
        [row.clone(), row.clone(), row],
        vec![],
    );
    let p = futures_executor::block_on(
        TorchDevice::cpu(gnx_common::cache::Cache::in_memory())
            .unwrap()
            .compile(&b.finish(vec![outputs[2]]).build().unwrap()),
    )
    .unwrap();
    let args = [[1, 0, 1, 0, 0], [4, 2, 1, 2, 1], [0, 1, 2, 3, 4]]
        .map(|v: [i32; 5]| Tensor::from_slice(&v));
    let payload = p.modules().next().unwrap().forward_ts(&args).unwrap();
    assert_eq!(Vec::<i32>::try_from(payload).unwrap(), [4, 1, 3, 2, 0]);
}
#[test]
fn special_functions_match_jax_and_fft_round_trips() {
    let x = Tensor::from_slice(&[0.17f32, 0.62, 0.95]);
    for (op, e) in [
        (special::erf(), [0.189992473, 0.619411409, 0.820890784]),
        (
            special::bessel_i1e(),
            [0.0719708726, 0.174905121, 0.205219984],
        ),
    ] {
        let actual = Vec::<f32>::try_from(run(op, &[(&x, DType::F32)], &[3], DType::F32)).unwrap();
        for (y, e) in actual.into_iter().zip(e) {
            assert!((y - e).abs() < 1e-5)
        }
    }
    let f = run(
        fft(FftKind::Rfft, &[3]),
        &[(&x, DType::F32)],
        &[2],
        DType::C64,
    );
    let y = run(
        fft(FftKind::Irfft, &[3]),
        &[(&f, DType::C64)],
        &[3],
        DType::F32,
    );
    assert!(y.allclose(&x, 1e-6, 1e-6, false));
}

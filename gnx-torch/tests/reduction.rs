use gnx_expr::{ValueInfo, pass::Builder};
use gnx_math::{
    ArrayInfo, DType, Shape,
    ops::{
        elementwise,
        reduction::{Window, reduce, reduce_window},
        select_op,
    },
};
use gnx_torch::{TorchDevice, TorchExecutable};
use tch::{Kind, Tensor};

fn info(shape: &[usize], dtype: DType) -> ValueInfo {
    ValueInfo::new(ArrayInfo::new(Shape::fixed(shape.iter().copied()), dtype))
}

// A left-biased maximum that carries the integer payload from the winning
// operand. The second result depends on the first accumulator's comparison.
fn maximum_with_payload(shape: &[usize], axes: &[usize]) -> TorchExecutable {
    maximum(shape, axes, None)
}
fn maximum(shape: &[usize], axes: &[usize], window: Option<Window>) -> TorchExecutable {
    let f = info(&[], DType::F32);
    let i = info(&[], DType::I64);
    let mut r = Builder::new();
    let a = r.input(f.clone());
    let ai = r.input(i.clone());
    let b = r.input(f.clone());
    let bi = r.input(i.clone());
    let take = r.emit1(elementwise::gt(), &[b, a], info(&[], DType::Bool));
    let v = r.emit1(select_op(), &[take, b, a], f.clone());
    let id = r.emit1(select_op(), &[take, bi, ai], i.clone());
    let mut p = Builder::new();
    let x = p.input(info(shape, DType::F32));
    let ids = p.input(info(shape, DType::I64));
    let v0 = p.input(f);
    let id0 = p.input(i);
    let reducer = r.finish(vec![v, id]).into();
    let out = match window {
        Some(w) => reduce_window(&mut p, &[x, ids], &[v0, id0], w, reducer),
        None => reduce(&mut p, &[x, ids], &[v0, id0], axes, reducer),
    }
    .unwrap();
    TorchDevice::cpu().unwrap().compile(&p.finish(out)).unwrap()
}
fn run(p: &TorchExecutable, args: &[Tensor]) -> (Tensor, Tensor) {
    let mut traces = p.modules();
    let values = traces.next().unwrap().forward_ts(args).unwrap();
    let ids = traces.next().unwrap().forward_ts(args).unwrap();
    assert_eq!(values.kind(), Kind::Float);
    assert_eq!(ids.kind(), Kind::Int64);
    (values, ids)
}
#[test]
fn variadic_reduce_nonleading_axes_replays_with_new_winners() {
    let p = maximum_with_payload(&[2, 2, 2, 2], &[1, 3]);
    for sign in [1f32, -1.] {
        let a = [
            1., 9., 5., 5., 8., 2., 4., 5., 7., 3., -2., -1., 6., 4., 0., -3.,
        ]
        .map(|x| x * sign);
        let dims = [2, 2, 2, 2];
        let args = [
            Tensor::from_slice(&a).reshape(dims),
            Tensor::arange(16, (Kind::Int64, tch::Device::Cpu)).reshape(dims),
            Tensor::from(f32::NEG_INFINITY),
            Tensor::from(-1i64),
        ];
        let (v, id) = run(&p, &args);
        assert_eq!(v.size(), [2, 2]);
        assert_eq!(id.size(), [2, 2]);
        let winners = if sign > 0. {
            [1, 2, 8, 14]
        } else {
            [0, 6, 9, 15]
        };
        assert_eq!(
            Vec::<i64>::try_from(id.flatten(0, -1)).unwrap(),
            winners.map(|n| n as i64)
        );
        assert_eq!(
            Vec::<f32>::try_from(v.flatten(0, -1)).unwrap(),
            winners.map(|n| a[n])
        );
    }
}
#[test]
fn empty_reduced_axis_broadcasts_each_live_initializer() {
    let p = maximum_with_payload(&[2, 0, 3], &[1]);
    for (x, k) in [(-10f32, 27i64), (-20., 39)] {
        let args = [
            Tensor::zeros([2, 0, 3], (Kind::Float, tch::Device::Cpu)),
            Tensor::zeros([2, 0, 3], (Kind::Int64, tch::Device::Cpu)),
            Tensor::from(x),
            Tensor::from(k),
        ];
        let (v, id) = run(&p, &args);
        assert_eq!(v.size(), [2, 3]);
        assert_eq!(id.size(), [2, 3]);
        assert_eq!(Vec::<f32>::try_from(v.flatten(0, -1)).unwrap(), [x; 6]);
        assert_eq!(Vec::<i64>::try_from(id.flatten(0, -1)).unwrap(), [k; 6]);
    }
}
#[test]
fn variadic_window_stride_crop_and_dilation_replay() {
    let w = Window {
        dimensions: vec![2, 2],
        strides: vec![2, 2],
        padding: vec![(0, 2), (-1, 2)],
        base_dilation: vec![2, 1],
        window_dilation: vec![2, 2],
    };
    let p = maximum(&[3, 4], &[], Some(w));
    for (sign, winners) in [(1f32, [3, 3, 9, 7, 9, 11]), (-1., [1, 7, 11, 11, 11, 11])] {
        let values = [99., 1., 0., 8., 98., 7., 0., 6., 97., 9., 0., 4.].map(|x| x * sign);
        let args = [
            Tensor::from_slice(&values).reshape([3, 4]),
            Tensor::arange(12, (Kind::Int64, tch::Device::Cpu)).reshape([3, 4]),
            Tensor::from(f32::NEG_INFINITY),
            Tensor::from(-1i64),
        ];
        let (v, id) = run(&p, &args);
        assert_eq!(v.size(), [3, 2]);
        assert_eq!(id.size(), [3, 2]);
        assert_eq!(
            Vec::<i64>::try_from(id.flatten(0, -1)).unwrap(),
            winners.map(|n| n as i64)
        );
        assert_eq!(
            Vec::<f32>::try_from(v.flatten(0, -1)).unwrap(),
            winners.map(|n| values[n])
        );
    }
}
#[test]
fn reduce_window_empty_outputs_and_padding_only_windows() {
    for (n, padding, expected_len) in [(2, (0, 0), 0), (0, (2, 2), 2)] {
        let w = Window {
            dimensions: vec![3],
            strides: vec![1],
            padding: vec![padding],
            base_dilation: vec![1],
            window_dilation: vec![1],
        };
        let p = maximum(&[n], &[], Some(w));
        let args = [
            Tensor::zeros([n as i64], (Kind::Float, tch::Device::Cpu)),
            Tensor::zeros([n as i64], (Kind::Int64, tch::Device::Cpu)),
            Tensor::from(-5f32),
            Tensor::from(12i64),
        ];
        let (v, id) = run(&p, &args);
        assert_eq!(v.size(), [expected_len as i64]);
        assert_eq!(id.size(), v.size());
        assert_eq!(Vec::<f32>::try_from(v).unwrap(), vec![-5.; expected_len]);
        assert_eq!(Vec::<i64>::try_from(id).unwrap(), vec![12; expected_len]);
    }
}
#[test]
fn empty_kept_axis_retains_zero_sized_output_shape() {
    let p = maximum_with_payload(&[0, 3, 2], &[1]);
    let args = [
        Tensor::zeros([0, 3, 2], (Kind::Float, tch::Device::Cpu)),
        Tensor::zeros([0, 3, 2], (Kind::Int64, tch::Device::Cpu)),
        Tensor::from(0f32),
        Tensor::from(0i64),
    ];
    let (v, id) = run(&p, &args);
    assert_eq!(v.size(), [0, 2]);
    assert_eq!(id.size(), [0, 2]);
}

use futures_executor::block_on;
use gnx_base::ops::reduction::reduce_add_op;
use gnx_expr::{EqnSource, ExprBuilder, Value, VarInfo, VarScope};
use gnx_math::{
    ArrayInfo, DType, DataHandle, Device, ExecOpts, HostBuffer, Shape, TransferPolicy, ops::*,
};
use gnx_torch::TorchDevice;

const COPY: ExecOpts = ExecOpts {
    transfers: TransferPolicy::AllowCopy,
};
fn ty(shape: &[usize]) -> VarInfo {
    VarInfo::new(ArrayInfo::new(
        Shape::fixed(shape.iter().copied()),
        DType::F32,
    ))
}
fn array(shape: &[usize], data: &[f32]) -> Value {
    let info = ArrayInfo::new(Shape::fixed(shape.iter().copied()), DType::F32);
    let bytes = data
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    Value::new(DataHandle::new(
        HostBuffer::copy_from_slice(info, &bytes).unwrap(),
    ))
}
fn data(v: &Value) -> &DataHandle {
    v.downcast().unwrap()
}
fn read(device: &TorchDevice, value: &Value) -> Vec<f32> {
    block_on(device.download(data(value)))
        .unwrap()
        .bytes()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

// libtch lives in the native bridge; its stream callbacks live in Rust.
#[test]
fn tensor_stream_round_trip_across_native_bridge() {
    let mut bytes = Vec::new();
    tch::Tensor::from(42i64).save_to_stream(&mut bytes).unwrap();
    let tensor = tch::Tensor::load_from_stream(std::io::Cursor::new(bytes)).unwrap();
    assert_eq!(tensor.int64_value(&[]), 42);
}

#[test]
fn replay_with_fresh_captures_and_inputs() {
    let d = TorchDevice::cpu(gnx_base::cache::Cache::in_memory()).unwrap();
    let mut vars = VarScope::new();
    let [x, y, s, p] = std::array::from_fn(|_| vars.create_var());
    let t = ty(&[2]);
    let body = ExprBuilder::from_parts(
        vec![],
        vec![(x, t.clone())],
        vec![(y, t.clone())],
        vec![
            EqnSource::new(add_op(), vec![], vec![x, y], vec![(s, t.clone())]),
            EqnSource::new(mul_op(), vec![], vec![s, x], vec![(p, t)]),
        ],
        vec![p, x, s, p],
    )
    .build()
    .unwrap();
    let compiled = futures_executor::block_on(d.compile(&body)).unwrap();
    for (xs, ys, product, sum) in [
        ([1., 2.], [3., 4.], [4., 12.], [4., 6.]),
        ([-1., 4.], [2., 3.], [-1., 28.], [1., 7.]),
    ] {
        let args = [array(&[2], &xs), array(&[2], &ys)];
        assert_eq!(
            compiled.run(&args, ExecOpts::default()).unwrap_err().kind(),
            std::io::ErrorKind::Unsupported
        );
        let args = args.map(|v| Value::new(block_on(d.upload(data(&v), COPY.transfers)).unwrap()));
        let out = compiled.run(&args, ExecOpts::default()).unwrap();
        assert_eq!(read(&d, &out[0]), product);
        assert_eq!(read(&d, &out[2]), sum);
        let id = |v: &Value| data(v).single_allocation().unwrap().id();
        assert_eq!(id(&out[1]), id(&args[0]));
        assert_eq!(id(&out[0]), id(&out[3]));
    }
}

#[test]
fn device_executes_shape_ops_and_reduction() {
    let d = TorchDevice::cpu(gnx_base::cache::Cache::in_memory()).unwrap();
    let mut vars = VarScope::new();
    let [x, t, r, b, s] = std::array::from_fn(|_| vars.create_var());
    let body = ExprBuilder::from_parts(
        vec![],
        vec![],
        vec![(x, ty(&[2, 2]))],
        vec![
            EqnSource::new(
                transpose_op(&[1, 0]),
                vec![],
                vec![x],
                vec![(t, ty(&[2, 2]))],
            ),
            EqnSource::new(reshape_op(), vec![], vec![t], vec![(r, ty(&[4]))]),
            EqnSource::new(
                broadcast_in_dim_op(&[0]),
                vec![],
                vec![r],
                vec![(b, ty(&[4, 3]))],
            ),
            EqnSource::new(reduce_add_op(&[1]), vec![], vec![b], vec![(s, ty(&[4]))]),
        ],
        vec![t, s],
    )
    .build()
    .unwrap();
    let out = block_on(d.execute(body, [array(&[2, 2], &[1., 2., 3., 4.])], COPY)).unwrap();
    assert_eq!(read(&d, &out[0]), [1., 3., 2., 4.]);
    assert_eq!(read(&d, &out[1]), [3., 9., 6., 12.]);
}

#[test]
fn unsupported_live_op_fails_but_dead_op_is_pruned() {
    let d = TorchDevice::cpu(gnx_base::cache::Cache::in_memory()).unwrap();
    let mut vars = VarScope::new();
    gnx_base::unit_operations! {
        struct Unknown;
    }
    impl gnx_expr::Operation for Unknown {
        fn display(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("unknown")
        }
    }
    gnx_expr::dialect! { dyn gnx_expr::Operation, "array" => { "unknown" => Unknown } }
    assert_eq!(gnx_base::ops::array_dialect(), "array");
    let [x, unknown, one] = std::array::from_fn(|_| vars.create_var());
    let body = |result| {
        ExprBuilder::from_parts(
            vec![],
            vec![],
            vec![(x, ty(&[2]))],
            vec![
                EqnSource::new(
                    gnx_expr::Op::new(Unknown),
                    vec![],
                    vec![x],
                    vec![(unknown, ty(&[2]))],
                ),
                EqnSource::new(ones_op(), vec![], vec![], vec![(one, ty(&[2]))]),
            ],
            vec![result],
        )
        .build()
        .unwrap()
    };
    let err = futures_executor::block_on(d.compile(&body(unknown)))
        .err()
        .unwrap();
    assert_eq!(err.kind(), std::io::ErrorKind::Unsupported);
    let out = futures_executor::block_on(d.compile(&body(one)))
        .unwrap()
        .run(&[array(&[2], &[3., 4.])], ExecOpts::default())
        .unwrap();
    assert_eq!(read(&d, &out[0]), [1., 1.]);
}

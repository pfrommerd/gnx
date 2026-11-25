use futures_executor::block_on;
use gnx::Eval;
use gnx_expr::{
    Capture, Eqn, EqnSource, ExprBuilder, Invocation, ObjectTracer, Op, Operation, SubExprBuilder,
    Trace, TraceContext, Value, VarInfo, VarScope,
};
use gnx_math::{
    ArrayInfo, DType, DataHandle, ExecOpts, HostBuffer, HostId, LocalAllocations, MemoryLocation,
    Shape, ShardStorage, TransferPolicy, ops::add_op,
};
use gnx_transforms::{cf::call_op, cond, scan, switch, while_loop};
use gnx_xla::{LoweringContext, XlaDevice, XlaLowering, XlaOp, register_xla_lowering};

const COPY: ExecOpts = ExecOpts {
    transfers: TransferPolicy::AllowCopy,
};

fn array_info(dims: &[usize], dtype: DType) -> ArrayInfo {
    ArrayInfo::new(Shape::fixed(dims.iter().copied()), dtype)
}
fn host(dims: &[usize], dtype: DType, bytes: Vec<u8>) -> DataHandle {
    DataHandle::new(HostBuffer::copy_from_slice(array_info(dims, dtype), &bytes).unwrap())
}
fn floats(dims: &[usize], data: &[f32]) -> DataHandle {
    host(
        dims,
        DType::F32,
        data.iter().flat_map(|x| x.to_le_bytes()).collect(),
    )
}
fn unpack(device: &XlaDevice, value: &Value) -> Vec<f32> {
    device
        .download(value.downcast::<DataHandle>().unwrap())
        .unwrap()
        .bytes()
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

#[test]
fn array_arithmetic_broadcasts_and_evaluates_lazily() {
    use gnx::Array;
    let device = XlaDevice::cpu().unwrap();
    let a = Array::from_data(floats(&[2, 1], &[-5., 8.]));
    let b = Array::from_data(floats(&[2], &[2., 4.]));
    let results = vec![
        &a + &b,
        a.clone() - &b,
        &a * b.clone(),
        a.clone() / b.clone(),
        a % b,
    ];
    assert!(
        results
            .iter()
            .all(|v| v.data().is_none() && v.shape() == Shape::fixed([2, 2]))
    );
    block_on(device.eval(&results, COPY)).unwrap();
    for (result, expected) in results.iter().zip([
        [-3., -1., 10., 12.],
        [-7., -9., 6., 4.],
        [-10., -20., 16., 32.],
        [-2.5, -1.25, 4., 2.],
        [-1., -1., 0., 0.],
    ]) {
        assert_eq!(
            unpack(&device, &Value::new(result.data().unwrap().clone())),
            expected
        );
    }
}

#[test]
fn compile_nested_call_reuse_outputs_and_forward_remote_refs() {
    let device = XlaDevice::cpu().unwrap();
    let array = array_info(&[2], DType::F32);
    let ty = VarInfo::new(array.clone());
    let mut vars = VarScope::new();
    let [a, b, r, sum] = std::array::from_fn(|_| vars.create_var());
    let child = ExprBuilder::from_parts(
        vec![],
        vec![(a, ty.clone())],
        vec![(b, ty.clone())],
        vec![EqnSource::new(
            add_op(),
            vec![],
            vec![a, b],
            vec![(sum, ty.clone())],
        )],
        vec![sum, sum],
    );
    let [c, d, dead] = std::array::from_fn(|_| vars.create_var());
    let program = ExprBuilder::from_parts(
        vec![],
        vec![(a, ty.clone())],
        vec![(b, ty.clone()), (r, ty.clone())],
        vec![
            EqnSource::new(
                call_op(),
                vec![SubExprBuilder::new(child).with_vars([a])],
                vec![b],
                vec![(c, ty.clone()), (d, ty.clone())],
            ),
            EqnSource::new(add_op(), vec![], vec![r, r], vec![(dead, ty)]),
        ],
        vec![d, r, c, d],
    )
    .build()
    .unwrap();
    let compiled = device.compile(&program).unwrap();
    let foreign = LocalAllocations::new();
    let reference = foreign.insert(());
    let remote_id = reference.id();
    let remote = Value::new(
        DataHandle::single(
            array,
            MemoryLocation::host(HostId::fresh()),
            ShardStorage::Reference(reference),
        )
        .unwrap(),
    );
    let left = Value::new(floats(&[2], &[1., 2.]));
    let right = Value::new(floats(&[2], &[3., 4.]));
    let outputs = compiled
        .run(&[left, right, remote.clone()], ExecOpts::default())
        .unwrap();
    assert_eq!(unpack(&device, &outputs[0]), [4., 6.]);
    assert_eq!(unpack(&device, &outputs[2]), [4., 6.]);
    let allocation = |value: &Value| {
        value
            .downcast::<DataHandle>()
            .unwrap()
            .single_allocation()
            .unwrap()
            .id()
    };
    assert_eq!(allocation(&outputs[1]), remote_id);
    assert_eq!(allocation(&outputs[0]), allocation(&outputs[3]));
    let outputs = compiled
        .run(
            &[outputs[0].clone(), outputs[2].clone(), remote],
            ExecOpts::default(),
        )
        .unwrap();
    assert_eq!(unpack(&device, &outputs[0]), [8., 12.]);
    assert_eq!(allocation(&outputs[1]), remote_id);
    assert!(
        device
            .download(outputs[1].downcast::<DataHandle>().unwrap())
            .is_err()
    );
}

fn trace(data: DataHandle) -> Trace {
    let info = VarInfo::new(data.info());
    Trace::concrete(Value::new(data), info)
}
fn scalar(n: f32) -> Trace {
    trace(floats(&[], &[n]))
}
fn add(a: &Trace, b: &Trace) -> Trace {
    invoke(add_op(), a, b, a.info().clone())
}
fn invoke(op: Op, a: &Trace, b: &Trace, info: VarInfo) -> Trace {
    Invocation::invoke(op, vec![], vec![a.clone(), b.clone()], vec![info])
        .pop()
        .unwrap()
        .into_trace_ref()
}
fn region(infos: &[VarInfo], build: impl FnOnce(&[Trace]) -> Vec<Trace>) -> Capture {
    let ctx = TraceContext::enter();
    let args: Vec<_> = infos.iter().cloned().map(Trace::placeholder).collect();
    let outputs = build(&args);
    Capture::from_context(
        ctx.context(),
        &args.into_iter().map(ObjectTracer::from).collect::<Vec<_>>(),
        &outputs
            .into_iter()
            .map(ObjectTracer::from)
            .collect::<Vec<_>>(),
    )
    .unwrap()
}

gnx_common::unit_operations! {
    struct LessThan;
}
impl Operation for LessThan {}
gnx_expr::dialect! {
    dyn Operation, "xla_test" => { "less_than" => LessThan }
}
impl XlaLowering for LessThan {
    fn lower_xla(
        &self,
        _: &LoweringContext<'_>,
        _: Eqn<'_>,
        operands: &[XlaOp],
    ) -> std::io::Result<Vec<XlaOp>> {
        Ok(vec![
            operands[0]
                .lt(&operands[1])
                .map_err(std::io::Error::other)?,
        ])
    }
}
register_xla_lowering!(LessThan);

#[test]
fn runs_structured_regions() {
    let device = XlaDevice::cpu().unwrap();
    let f32_scalar = VarInfo::new(array_info(&[], DType::F32));
    let bool_scalar = VarInfo::new(array_info(&[], DType::Bool));
    let values = |t: &Trace| unpack(&device, t.try_concrete().unwrap());
    let double = region(std::slice::from_ref(&f32_scalar), |a| {
        vec![add(&a[0], &a[0])]
    });
    let triple = region(std::slice::from_ref(&f32_scalar), |a| {
        vec![add(&add(&a[0], &a[0]), &a[0])]
    });
    let identity = region(std::slice::from_ref(&f32_scalar), |a| a.to_vec());
    let args = [scalar(4.).into()];
    let yes = trace(host(&[], DType::Bool, vec![1]));
    let branch = cond(yes, &double, &identity, &args).unwrap();
    block_on(device.eval(&branch, COPY)).unwrap();
    assert_eq!(values(&branch[0]), [8.]);
    for (index, expected) in [(-4_i32, 4.), (1, 8.), (12, 12.)] {
        let index = trace(host(&[], DType::I32, index.to_le_bytes().to_vec()));
        let selected = switch(index, &[&identity, &double, &triple], &args).unwrap();
        block_on(device.eval(&selected, COPY)).unwrap();
        assert_eq!(values(&selected[0]), [expected]);
    }
    let predicate = region(std::slice::from_ref(&f32_scalar), |a| {
        vec![invoke(Op::new(LessThan), &a[0], &scalar(5.), bool_scalar)]
    });
    let carried = while_loop(&predicate, &double, &[scalar(1.)]).unwrap();
    block_on(device.eval(&carried, COPY)).unwrap();
    assert_eq!(values(&carried[0]), [8.]);
    let step = region(&[f32_scalar.clone(), f32_scalar], |a| {
        let c = add(&a[0], &a[1]);
        vec![c.clone(), c]
    });
    for (xs, expected) in [(&[1., 2., 3.][..], &[1., 3., 6.][..]), (&[][..], &[][..])] {
        let (carry, ys) = scan(
            &step,
            &[scalar(0.)],
            &[trace(floats(&[xs.len()], xs))],
            None,
        )
        .unwrap();
        block_on(device.eval(&(carry[0].clone(), ys[0].clone()), COPY)).unwrap();
        assert_eq!(values(&carry[0]), [expected.last().copied().unwrap_or(0.)]);
        assert_eq!(values(&ys[0]), expected);
    }
}

#[test]
fn generic_reducers_execute_with_mixed_results_and_windows() {
    use gnx_expr::pass::Builder;
    use gnx_math::ops::{
        creation::constant,
        elementwise::{max_op, min_op},
        reduction,
    };

    let device = XlaDevice::cpu().unwrap();
    let tensor = VarInfo::new(array_info(&[5], DType::F32));
    let scalar = VarInfo::new(array_info(&[], DType::F32));
    let mut b = Builder::new();
    let input = b.input(tensor);
    let high = b.emit1(constant(f64::INFINITY), &[], scalar.clone());
    let low = b.emit1(constant(f64::NEG_INFINITY), &[], scalar.clone());
    let combine = b
        .region(
            &[],
            &[
                scalar.clone(),
                scalar.clone(),
                scalar.clone(),
                scalar.clone(),
            ],
            |cb, _, args| {
                Ok::<_, gnx_expr::LoweringError>(vec![
                    cb.emit1(min_op(), &[args[0], args[2]], scalar.clone()),
                    cb.emit1(max_op(), &[args[1], args[3]], scalar.clone()),
                ])
            },
        )
        .unwrap();
    // The reducer is closed, so it can be stored as a root of its own.
    let gnx_expr::ExprRef::Internal(combine) = combine.expr().clone() else {
        unreachable!("a region is built in place")
    };
    let combine = (*combine).clone().build().unwrap().handle();
    let mut out =
        reduction::reduce(&mut b, &[input, input], &[high, low], &[0], combine.clone()).unwrap();
    let window = reduction::Window {
        dimensions: vec![2],
        strides: vec![1],
        padding: vec![(0, 0)],
        base_dilation: vec![1],
        window_dilation: vec![2],
    };
    out.extend(
        reduction::reduce_window(
            &mut b,
            &[input, input],
            &[high, low],
            window,
            combine.clone(),
        )
        .unwrap(),
    );
    let result = device
        .compile(&b.finish(out).build().unwrap())
        .unwrap()
        .run(&[Value::new(floats(&[5], &[5., 3., 7., 9., 6.]))], COPY)
        .unwrap();
    assert_eq!(unpack(&device, &result[0]), [3.]);
    assert_eq!(unpack(&device, &result[1]), [9.]);
    assert_eq!(unpack(&device, &result[2]), [5., 3., 6.]);
    assert_eq!(unpack(&device, &result[3]), [7., 9., 7.]);
}

#[test]
fn differentiates_and_batches_axis_and_window_reductions() {
    use gnx_expr::pass::Builder;
    use gnx_math::ops::{
        creation::constant,
        elementwise::{max_op, min_op},
        mul_op, ones_op, reduction, zeros_op,
    };
    use gnx_transforms::ops::{ad, map};

    let vector = VarInfo::new(array_info(&[3], DType::F32));
    let pair = VarInfo::new(array_info(&[2], DType::F32));
    let scalar = VarInfo::new(array_info(&[], DType::F32));
    let mut b = Builder::new();
    let x = b.input(vector.clone());
    let combiner = |op| {
        let mut combine = Builder::new();
        let lhs = combine.input(scalar.clone());
        let rhs = combine.input(scalar.clone());
        let result = combine.emit1(op, &[lhs, rhs], scalar.clone());
        combine.finish(vec![result]).build().unwrap().handle()
    };
    let mut axis_reduce = |op, identity| {
        let initial = b.emit1(identity, &[], scalar.clone());
        reduction::reduce(&mut b, &[x], &[initial], &[0], combiner(op)).unwrap()[0]
    };
    let sum = axis_reduce(add_op(), zeros_op());
    let product = axis_reduce(mul_op(), ones_op());
    let min = axis_reduce(min_op(), constant(f64::INFINITY));
    let max = axis_reduce(max_op(), constant(f64::NEG_INFINITY));
    let initial = b.emit1(zeros_op(), &[], scalar.clone());
    let window = reduction::Window {
        dimensions: vec![2],
        strides: vec![1],
        padding: vec![(0, 0)],
        base_dilation: vec![1],
        window_dilation: vec![1],
    };
    let windows =
        reduction::reduce_window(&mut b, &[x], &[initial], window, combiner(add_op())).unwrap()[0];
    let body = SubExprBuilder::new(b.finish(vec![sum, product, min, max, windows]));

    let mut b = Builder::new();
    let x = b.input(vector.clone());
    let t = b.input(vector.clone());
    let outputs: Vec<_> = std::iter::repeat_n(scalar.clone(), 4)
        .chain([pair.clone()])
        .collect();
    let mut results = b.emit(
        ad::jvp(&[0], &[0, 1, 2, 3, 4]),
        &[x, t],
        outputs.iter().cloned().chain(outputs.iter().cloned()),
        vec![body.clone()],
    );
    let cotangents: Vec<_> = outputs.iter().cloned().map(|info| b.input(info)).collect();
    let reverse_inputs: Vec<_> = std::iter::once(x).chain(cotangents).collect();
    let reverse = b.emit(
        ad::vjp(&[0], &[0, 1, 2, 3, 4]),
        &reverse_inputs,
        outputs.into_iter().chain([vector]),
        vec![body.clone()],
    );
    results.push(reverse[5]);
    let batch = b.input(VarInfo::new(array_info(&[2, 3], DType::F32)));
    let matrix = VarInfo::new(array_info(&[2, 2], DType::F32));
    let batched = b.emit(
        map::map(2, &[0], &[0, 0, 0, 0, 0]),
        &[batch],
        std::iter::repeat_n(pair, 4).chain([matrix]),
        vec![body],
    );
    results.extend(batched);

    let device = XlaDevice::cpu().unwrap();
    let output = device
        .compile(&b.finish(results).build().unwrap())
        .unwrap()
        .run(
            &[
                Value::new(floats(&[3], &[2., 2., 2.])),
                Value::new(floats(&[3], &[1., 2., 6.])),
                Value::new(floats(&[], &[1.])),
                Value::new(floats(&[], &[1.])),
                Value::new(floats(&[], &[3.])),
                Value::new(floats(&[], &[6.])),
                Value::new(floats(&[2], &[2., 5.])),
                Value::new(floats(&[2, 3], &[2., 2., 2., 1., 4., 9.])),
            ],
            COPY,
        )
        .unwrap();
    let output: Vec<_> = output.iter().map(|value| unpack(&device, value)).collect();
    assert_eq!(
        output,
        [
            vec![6.],
            vec![8.],
            vec![2.],
            vec![2.],
            vec![4., 4.],
            vec![9.],
            vec![36.],
            vec![3.],
            vec![3.],
            vec![3., 8.],
            vec![10., 15., 13.],
            vec![6., 14.],
            vec![8., 36.],
            vec![2., 1.],
            vec![2., 9.],
            vec![4., 4., 5., 13.],
        ]
    );
}

#[test]
fn popcount_uses_the_operand_width() {
    use gnx_expr::pass::Builder;
    use gnx_math::ops::bitwise::population_count;
    let device = XlaDevice::cpu().unwrap();
    let info = VarInfo::new(array_info(&[4], DType::I8));
    let mut builder = Builder::new();
    let x = builder.input(info.clone());
    let y = builder.emit1(population_count(), &[x], info);
    let result = device
        .compile(&builder.finish(vec![y]).build().unwrap())
        .unwrap()
        .run(
            &[Value::new(host(&[4], DType::I8, vec![255, 128, 13, 0]))],
            COPY,
        )
        .unwrap();
    let data = result[0].downcast::<DataHandle>().unwrap();
    assert_eq!(device.download(data).unwrap().bytes(), &[8, 1, 3, 0]);
}

#[test]
#[cfg(feature = "lapack")]
fn lapack_factorizations_execute_on_the_local_device() {
    use gnx_expr::pass::Builder;
    use gnx_math::ops::linalg::{lu, svd};
    let device = XlaDevice::cpu().unwrap();
    let matrix = VarInfo::new(array_info(&[2, 2], DType::F32));
    let pivot = VarInfo::new(array_info(&[2], DType::I32));
    let singular = VarInfo::new(array_info(&[2], DType::F32));
    let mut builder = Builder::new();
    let x = builder.input(matrix.clone());
    let mut out = builder.emit(lu(), &[x], [matrix, pivot.clone(), pivot], vec![]);
    out.push(builder.emit1(svd(false, false), &[x], singular));
    let result = device
        .compile(&builder.finish(out).build().unwrap())
        .unwrap()
        .run(&[Value::new(floats(&[2, 2], &[0., 2., 3., 0.]))], COPY)
        .unwrap();
    assert_eq!(unpack(&device, &result[0]), [3., 0., 0., 2.]);
    for (result, expected) in [(&result[1], [1_i32, 1]), (&result[2], [1_i32, 0])] {
        assert_eq!(
            device
                .download(result.downcast::<DataHandle>().unwrap())
                .unwrap()
                .bytes(),
            &expected
                .into_iter()
                .flat_map(i32::to_le_bytes)
                .collect::<Vec<_>>(),
        );
    }
    assert_eq!(unpack(&device, &result[3]), [3., 2.]);
}

#[test]
fn window_vjp_accumulates_overlapping_winners_and_additive_windows() {
    use gnx_expr::pass::Builder;
    use gnx_math::ops::{creation::constant, elementwise::max_op, reduction};

    let device = XlaDevice::cpu().unwrap();
    let input_info = VarInfo::new(array_info(&[3], DType::F32));
    let output_info = VarInfo::new(array_info(&[2], DType::F32));
    let scalar_info = VarInfo::new(array_info(&[], DType::F32));

    for (op, identity, expected) in [
        (max_op(), f64::NEG_INFINITY, [0., 3., 0.]),
        (add_op(), 0., [1., 3., 2.]),
    ] {
        let mut reducer = Builder::new();
        let left = reducer.input(scalar_info.clone());
        let right = reducer.input(scalar_info.clone());
        let result = reducer.emit1(op, &[left, right], scalar_info.clone());
        let reducer = reducer.finish(vec![result]).build().unwrap().handle();

        let mut body = Builder::new();
        let x = body.input(input_info.clone());
        let init = body.emit1(constant(identity), &[], scalar_info.clone());
        let window = reduction::Window {
            dimensions: vec![2],
            strides: vec![1],
            padding: vec![(0, 0)],
            base_dilation: vec![1],
            window_dilation: vec![1],
        };
        let result = reduction::reduce_window(&mut body, &[x], &[init], window, reducer).unwrap();

        let mut main = Builder::new();
        let x = main.input(input_info.clone());
        let cotangent = main.input(output_info.clone());
        let region = SubExprBuilder::new(body.finish(result));
        let outputs = main.emit(
            gnx_transforms::ops::ad::vjp(&[0], &[0]),
            &[x, cotangent],
            [output_info.clone(), input_info.clone()],
            vec![region],
        );

        let outputs = device
            .compile(&main.finish(outputs).build().unwrap())
            .unwrap()
            .run(
                &[
                    Value::new(floats(&[3], &[1., 8., 3.])),
                    Value::new(floats(&[2], &[1., 2.])),
                ],
                COPY,
            )
            .unwrap();
        assert_eq!(unpack(&device, &outputs[1]), expected);
    }
}

#[test]
fn convolution_vjp_uses_the_forward_window_geometry() {
    use gnx_expr::pass::Builder;
    use gnx_math::ops::contraction::{
        ConvAxes, ConvKernelAxes, ConvParameters, conv_general_dilated,
    };
    let device = XlaDevice::cpu().unwrap();
    let input = VarInfo::new(array_info(&[1, 1, 3], DType::F32));
    let kernel = VarInfo::new(array_info(&[1, 1, 2], DType::F32));
    let output = kernel.clone();
    let axes = ConvAxes {
        batch: 0,
        feature: 1,
        spatial: vec![2],
    };
    let mut body = Builder::new();
    let x = body.input(input.clone());
    let k = body.input(kernel.clone());
    let out = body.emit1(
        conv_general_dilated(ConvParameters {
            input: axes.clone(),
            output: axes,
            kernel: ConvKernelAxes {
                output_feature: 0,
                input_feature: 1,
                spatial: vec![2],
            },
            strides: vec![1],
            padding: vec![(0, 0)],
            lhs_dilation: vec![1],
            rhs_dilation: vec![1],
            feature_groups: 1,
            batch_groups: 1,
        }),
        &[x, k],
        output.clone(),
    );
    let mut b = Builder::new();
    let x = b.input(input.clone());
    let k = b.input(kernel.clone());
    let ct = b.input(output.clone());
    let results = b.emit(
        gnx_transforms::ops::ad::vjp(&[0, 1], &[0]),
        &[x, k, ct],
        [output, input, kernel],
        vec![SubExprBuilder::new(body.finish(vec![out]))],
    );
    let result = device
        .compile(&b.finish(results).build().unwrap())
        .unwrap()
        .run(
            &[
                Value::new(floats(&[1, 1, 3], &[1., 2., 4.])),
                Value::new(floats(&[1, 1, 2], &[2., 3.])),
                Value::new(floats(&[1, 1, 2], &[1., 1.])),
            ],
            COPY,
        )
        .unwrap();
    assert_eq!(unpack(&device, &result[0]), [8., 16.]);
    assert_eq!(unpack(&device, &result[1]), [2., 5., 3.]);
    assert_eq!(unpack(&device, &result[2]), [3., 6.]);
}

#[test]
fn real_fft_and_top_k_return_the_expected_values_and_dtypes() {
    use gnx_expr::pass::Builder;
    use gnx_math::ops::{
        fft::{FftKind, fft},
        ordering::top_k,
    };
    let device = XlaDevice::cpu().unwrap();
    let mut b = Builder::new();
    let x = b.input(VarInfo::new(array_info(&[4], DType::F32)));
    let spectrum = b.emit1(
        fft(FftKind::Rfft, &[4]),
        &[x],
        VarInfo::new(array_info(&[3], DType::C64)),
    );
    let mut out = b.emit(
        top_k(2),
        &[x],
        [
            VarInfo::new(array_info(&[2], DType::F32)),
            VarInfo::new(array_info(&[2], DType::I32)),
        ],
        vec![],
    );
    out.push(spectrum);
    let outputs = device
        .compile(&b.finish(out).build().unwrap())
        .unwrap()
        .run(&[Value::new(floats(&[4], &[1., 2., 3., 4.]))], COPY)
        .unwrap();
    assert_eq!(unpack(&device, &outputs[0]), [4., 3.]);
    let indices = device
        .download(outputs[1].downcast::<DataHandle>().unwrap())
        .unwrap();
    assert_eq!(indices.bytes(), &[3, 0, 0, 0, 2, 0, 0, 0]);
    assert_eq!(unpack(&device, &outputs[2]), [10., 0., -2., 2., -2., 0.]);
}

#[test]
fn sort_treats_signed_zeros_as_ties_and_stably_orders_integer_keys() {
    use gnx_expr::pass::Builder;
    use gnx_math::ops::ordering::sort;

    let device = XlaDevice::cpu().unwrap();
    let data = VarInfo::new(array_info(&[4], DType::F32));
    let indices = VarInfo::new(array_info(&[4], DType::I32));
    let mut b = Builder::new();
    let x = b.input(data.clone());
    let key = b.input(indices.clone());
    let mut out = b.emit(
        sort(0, 1, true),
        &[x, key],
        [data.clone(), indices.clone()],
        vec![],
    );
    out.extend(b.emit(sort(0, 1, true), &[key, x], [indices, data], vec![]));
    let keys = [3_i32, 2, 1, 0]
        .into_iter()
        .flat_map(i32::to_le_bytes)
        .collect();
    let outputs = device
        .compile(&b.finish(out).build().unwrap())
        .unwrap()
        .run(
            &[
                Value::new(floats(&[4], &[1., 0., -0., f32::NAN])),
                Value::new(host(&[4], DType::I32, keys)),
            ],
            COPY,
        )
        .unwrap();
    let v = unpack(&device, &outputs[0]);
    assert!(!v[0].is_sign_negative());
    assert!(v[1].is_sign_negative());
    assert_eq!(v[2], 1.);
    assert!(v[3].is_nan());
    let v = unpack(&device, &outputs[3]);
    assert!(v[0].is_nan());
    assert!(v[1].is_sign_negative());
    assert!(!v[2].is_sign_negative());
    assert_eq!(v[3], 1.);
}

#[test]
fn qr_and_cholesky_jvps_run_on_cpu() {
    use gnx_expr::pass::Builder;
    use gnx_math::ops::linalg::{cholesky, qr};
    let device = XlaDevice::cpu().unwrap();
    let meta = VarInfo::new(array_info(&[2, 2], DType::F32));
    let mut b = Builder::new();
    let a = b.input(meta.clone());
    let l = b.emit1(cholesky(true), &[a], meta.clone());
    let mut results = b.emit(qr(false, false), &[a], [meta.clone(), meta.clone()], vec![]);
    results.insert(0, l);
    let body = SubExprBuilder::new(b.finish(results));
    let mut b = Builder::new();
    let a = b.input(meta.clone());
    let t = b.input(meta.clone());
    let out = b.emit(
        gnx_transforms::ops::ad::jvp(&[0], &[0, 1, 2]),
        &[a, t],
        std::iter::repeat_n(meta, 6),
        vec![body],
    );
    let output = device
        .compile(&b.finish(out).build().unwrap())
        .unwrap()
        .run(
            &[
                Value::new(floats(&[2, 2], &[4., 0., 0., 9.])),
                Value::new(floats(&[2, 2], &[2., 0., 0., 6.])),
            ],
            COPY,
        )
        .unwrap();
    assert_eq!(unpack(&device, &output[0]), [2., 0., 0., 3.]);
    assert_eq!(unpack(&device, &output[3]), [0.5, 0., 0., 1.]);
    assert_eq!(unpack(&device, &output[4]), [0., 0., 0., 0.]);
    let r = unpack(&device, &output[2]);
    let dr = unpack(&device, &output[5]);
    assert_eq!(dr[0].abs(), 2.);
    assert_eq!(dr[3].abs(), 6.);
    assert_eq!(r[0].is_sign_negative(), dr[0].is_sign_negative());
    assert_eq!(r[3].is_sign_negative(), dr[3].is_sign_negative());
}

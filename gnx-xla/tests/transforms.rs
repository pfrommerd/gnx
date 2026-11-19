//! `ad.map` / `ad.jvp` / `ad.vjp` traced, lowered with the transformations
//! preserved, then expanded and executed by the XLA pipeline.

use futures_executor::block_on;
use gnx::Eval;
use gnx_expr::pass::Builder;
use gnx_expr::{
    Capture, Dialect, Eqn, ObjectTracer, Op, Operation, Trace, TraceContext, Value, ValueInfo,
    register_op_interface,
};
use gnx_math::ops::{add_op, mul_op};
use gnx_math::{ArrayInfo, DType, DataHandle, ExecOpts, HostBuffer, Shape, TransferPolicy};
use gnx_transforms::TransformResult;
use gnx_transforms::ad::region::{jvp, vjp};
use gnx_transforms::map::batch::{BatchRule, Batched, materialize};
use gnx_transforms::util::with_axis;
/// Record `map.map` at the region level with an inferred size.
fn map(
    body: &Capture,
    inputs: &[Trace],
    in_axes: &[usize],
    out_axes: &[usize],
) -> Result<Vec<Trace>, gnx_expr::LoweringError> {
    gnx_transforms::map::region::map(body, inputs, in_axes, out_axes, None)
}
use gnx_transforms::{cond, scan, while_loop};
use gnx_xla::{LoweringContext, XlaDevice, XlaLowering, XlaOp, register_xla_lowering};

const COPY: ExecOpts = ExecOpts {
    transfers: TransferPolicy::AllowCopy,
};

fn info(dims: &[usize], dtype: DType) -> ValueInfo {
    ValueInfo::new(ArrayInfo::new(Shape::fixed(dims.iter().copied()), dtype))
}
fn constant(dims: &[usize], dtype: DType, bytes: Vec<u8>) -> Trace {
    let data = DataHandle::new(
        HostBuffer::copy_from_slice(
            ArrayInfo::new(Shape::fixed(dims.iter().copied()), dtype),
            &bytes,
        )
        .unwrap(),
    );
    let info = ValueInfo::new(data.info());
    Trace::concrete(Value::new(data), info)
}
fn floats(dims: &[usize], data: &[f32]) -> Trace {
    constant(
        dims,
        DType::F32,
        data.iter().flat_map(|x| x.to_le_bytes()).collect(),
    )
}
fn bools(dims: &[usize], data: &[bool]) -> Trace {
    constant(dims, DType::Bool, data.iter().map(|&x| x as u8).collect())
}
fn apply(op: Op, inputs: &[&Trace], info: ValueInfo) -> Trace {
    gnx_expr::Invocation::invoke(
        op,
        vec![],
        inputs.iter().map(|t| (*t).clone()).collect(),
        vec![info],
    )
    .pop()
    .unwrap()
    .into_trace_ref()
}
fn add(a: &Trace, b: &Trace) -> Trace {
    apply(add_op(), &[a, b], a.info().clone())
}
fn mul(a: &Trace, b: &Trace) -> Trace {
    apply(mul_op(), &[a, b], a.info().clone())
}
fn region(infos: &[ValueInfo], build: impl FnOnce(&[Trace]) -> Vec<Trace>) -> Capture {
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
fn eval(device: &XlaDevice, traces: &[Trace]) -> Vec<Vec<f32>> {
    block_on(device.eval(&traces.to_vec(), COPY)).unwrap();
    traces
        .iter()
        .map(|trace| {
            device
                .download(
                    trace
                        .try_concrete()
                        .unwrap()
                        .downcast::<DataHandle>()
                        .unwrap(),
                )
                .unwrap()
                .bytes()
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect()
        })
        .collect()
}

#[track_caller]
fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{actual:?} vs {expected:?}");
    for (a, e) in actual.iter().zip(expected) {
        assert!((a - e).abs() < 1e-4, "{actual:?} vs {expected:?}");
    }
}

// A user-defined comparison with its own XLA lowering and batching rule.
#[derive(Clone, PartialEq, Eq, Hash)]
struct LessThan;
struct TestDialect;
impl Dialect for TestDialect {
    fn name(&self) -> &'static str {
        "transform_test"
    }
}
impl std::fmt::Display for LessThan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}::{}", Operation::dialect(self).name(), Operation::name(self))
    }
}

impl Operation for LessThan {
    fn name(&self) -> &'static str {
        "less_than"
    }
    fn dialect(&self) -> &'static dyn Dialect {
        &TestDialect
    }
}
impl XlaLowering for LessThan {
    fn lower_xla(
        &self,
        _: &LoweringContext<'_>,
        _: &Eqn,
        operands: &[XlaOp],
    ) -> std::io::Result<Vec<XlaOp>> {
        Ok(vec![
            operands[0]
                .lt(&operands[1])
                .map_err(std::io::Error::other)?,
        ])
    }
}
impl BatchRule for LessThan {
    fn batch(
        &self,
        b: &mut Builder,
        size: usize,
        eqn: &Eqn,
        inputs: &[Batched],
        _: &[Vec<Batched>],
    ) -> TransformResult<Vec<Batched>> {
        let operands = inputs
            .iter()
            .map(|&value| materialize(b, value, size, 0))
            .collect::<TransformResult<Vec<_>>>()?;
        let info = with_axis(&eqn.outputs()[0].1, 0, size)?;
        Ok(vec![Batched::at(
            b.emit1(eqn.op().clone(), &operands, info),
            0,
        )])
    }
}
register_xla_lowering!(LessThan);
register_op_interface!(dyn BatchRule: LessThan);
fn less(a: &Trace, b: &Trace) -> Trace {
    apply(Op::new(LessThan), &[a, b], info(&[], DType::Bool))
}

#[test]
fn map_vectorizes_with_captures_and_axes() {
    let device = XlaDevice::cpu().unwrap();
    let offset = floats(&[3], &[10., 20., 30.]);
    let body = region(&[info(&[3], DType::F32)], |x| {
        vec![add(&mul(&x[0], &x[0]), &offset)]
    });
    let xs = floats(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
    let stacked = map(&body, &[xs.clone()], &[0], &[1]).unwrap();
    assert_eq!(stacked[0].info(), &info(&[3, 2], DType::F32));
    // Map over the second axis of the transposed input as well.
    let columns = floats(&[3, 2], &[1., 4., 2., 5., 3., 6.]);
    let rows = map(&body, &[columns], &[1], &[0]).unwrap();
    let values = eval(&device, &[stacked[0].clone(), rows[0].clone()]);
    close(&values[0], &[11., 26., 24., 45., 39., 66.]);
    close(&values[1], &[11., 24., 39., 26., 45., 66.]);
}

#[test]
fn jvp_and_vjp_of_a_polynomial() {
    let device = XlaDevice::cpu().unwrap();
    let scalar = info(&[2], DType::F32);
    // f(x, y) = x * y + x
    let body = region(&[scalar.clone(), scalar.clone()], |a| {
        vec![add(&mul(&a[0], &a[1]), &a[0])]
    });
    let (x, y) = (floats(&[2], &[2., 3.]), floats(&[2], &[5., 7.]));
    let (tx, ty) = (floats(&[2], &[1., 0.5]), floats(&[2], &[-1., 2.]));
    let (out, tangent) = jvp(&body, &[x.clone(), y.clone()], &[tx, ty], &[0, 1], &[0]).unwrap();
    let ct = floats(&[2], &[1., 2.]);
    let (out2, grads) = vjp(&body, &[x, y], &[ct], &[0, 1], &[0]).unwrap();
    let values = eval(
        &device,
        &[
            out[0].clone(),
            tangent[0].clone(),
            out2[0].clone(),
            grads[0].clone(),
            grads[1].clone(),
        ],
    );
    close(&values[0], &[12., 24.]);
    // tx * y + x * ty + tx
    close(&values[1], &[1. * 5. - 2. + 1., 0.5 * 7. + 3. * 2. + 0.5]);
    close(&values[2], &[12., 24.]);
    close(&values[3], &[1. * 6., 2. * 8.]); // ct * (y + 1)
    close(&values[4], &[1. * 2., 2. * 3.]); // ct * x
}

#[test]
fn vjp_through_scan_and_cond() {
    let device = XlaDevice::cpu().unwrap();
    let scalar = info(&[], DType::F32);
    // f(c0, xs) = c0 * prod(xs), also stacking the running product.
    let step = region(&[scalar.clone(), scalar.clone()], |a| {
        let next = mul(&a[0], &a[1]);
        vec![next.clone(), next]
    });
    let body = region(&[scalar.clone(), info(&[3], DType::F32)], |a| {
        let (carry, ys) = scan(&step, &[a[0].clone()], &[a[1].clone()], None).unwrap();
        vec![carry[0].clone(), ys[0].clone()]
    });
    let c0 = floats(&[], &[2.]);
    let xs = floats(&[3], &[3., 4., 5.]);
    let (_, final_grads) = vjp(
        &body,
        &[c0.clone(), xs.clone()],
        &[floats(&[], &[1.])],
        &[0, 1],
        &[0],
    )
    .unwrap();
    // Cotangents for the stacked running products [6, 24, 120].
    let (_, ys_grads) = vjp(
        &body,
        &[c0, xs],
        &[floats(&[3], &[1., 1., 1.])],
        &[0, 1],
        &[1],
    )
    .unwrap();

    // g(p, x) = p ? x * x : x + x
    let square = region(&[scalar.clone()], |a| vec![mul(&a[0], &a[0])]);
    let double = region(&[scalar.clone()], |a| vec![add(&a[0], &a[0])]);
    let branch = region(&[info(&[], DType::Bool), scalar.clone()], |a| {
        cond(a[0].clone(), &square, &double, &[a[1].clone().into()]).unwrap()
    });
    let x = floats(&[], &[3.]);
    let one = floats(&[], &[1.]);
    let (_, yes) = vjp(
        &branch,
        &[bools(&[], &[true]), x.clone()],
        &[one.clone()],
        &[1],
        &[0],
    )
    .unwrap();
    let (_, no) = vjp(&branch, &[bools(&[], &[false]), x], &[one], &[1], &[0]).unwrap();

    let values = eval(
        &device,
        &[
            final_grads[0].clone(),
            final_grads[1].clone(),
            ys_grads[0].clone(),
            ys_grads[1].clone(),
            yes[0].clone(),
            no[0].clone(),
        ],
    );
    close(&values[0], &[60.]);
    close(&values[1], &[40., 30., 24.]);
    // d(6 + 24 + 120)/d(c0, x1, x2, x3)
    close(&values[2], &[3. + 12. + 60.]);
    close(&values[3], &[2. + 8. + 40., 6. + 30., 24.]);
    close(&values[4], &[6.]);
    close(&values[5], &[2.]);
}

#[test]
fn map_over_data_dependent_control_flow() {
    let device = XlaDevice::cpu().unwrap();
    let scalar = info(&[], DType::F32);
    let limit = floats(&[], &[5.]);
    let predicate = region(&[scalar.clone()], |a| vec![less(&a[0], &limit)]);
    let double = region(&[scalar.clone()], |a| vec![add(&a[0], &a[0])]);
    let grow = region(&[scalar.clone()], |a| {
        while_loop(&predicate, &double, &[a[0].clone()]).unwrap()
    });
    let grown = map(&grow, &[floats(&[3], &[1., 3., 7.])], &[0], &[0]).unwrap();

    let identity = region(&[scalar.clone()], |a| a.to_vec());
    let choose = region(&[info(&[], DType::Bool), scalar.clone()], |a| {
        cond(a[0].clone(), &double, &identity, &[a[1].clone().into()]).unwrap()
    });
    let chosen = map(
        &choose,
        &[
            bools(&[3], &[true, false, true]),
            floats(&[3], &[1., 2., 3.]),
        ],
        &[0, 0],
        &[0],
    )
    .unwrap();
    let values = eval(&device, &[grown[0].clone(), chosen[0].clone()]);
    close(&values[0], &[8., 6., 7.]);
    close(&values[1], &[2., 2., 6.]);
}

#[test]
fn jvp_through_while_and_nested_transformations() {
    let device = XlaDevice::cpu().unwrap();
    let scalar = info(&[], DType::F32);
    let limit = floats(&[], &[5.]);
    let predicate = region(&[scalar.clone()], |a| vec![less(&a[0], &limit)]);
    let double = region(&[scalar.clone()], |a| vec![add(&a[0], &a[0])]);
    let grow = region(&[scalar.clone()], |a| {
        while_loop(&predicate, &double, &[a[0].clone()]).unwrap()
    });
    let (out, tangent) = jvp(
        &grow,
        &[floats(&[], &[1.])],
        &[floats(&[], &[1.])],
        &[0],
        &[0],
    )
    .unwrap();

    // Reverse mode through a while loop is rejected when the program is expanded.
    let (_, grad) = vjp(
        &grow,
        &[floats(&[], &[1.])],
        &[floats(&[], &[1.])],
        &[0],
        &[0],
    )
    .unwrap();
    let error = block_on(device.eval(&grad, COPY))
        .err()
        .expect("while_loop is not transposable");
    assert!(error.to_string().contains("while_loop"), "{error}");

    // Per-example gradients: map(vjp(w * x, wrt w)) over x.
    let w = floats(&[], &[3.]);
    let product = region(&[scalar.clone(), scalar.clone()], |a| {
        vec![mul(&a[0], &a[1])]
    });
    let per_example = region(&[scalar.clone()], |a| {
        vjp(
            &product,
            &[w.clone(), a[0].clone()],
            &[floats(&[], &[1.])],
            &[0],
            &[0],
        )
        .unwrap()
        .1
    });
    let grads = map(&per_example, &[floats(&[4], &[1., 2., 3., 4.])], &[0], &[0]).unwrap();
    // Forward-over-map: tangent of a mapped square.
    let square = region(&[scalar.clone()], |a| vec![mul(&a[0], &a[0])]);
    let mapped = region(&[info(&[2], DType::F32)], |a| {
        map(&square, &[a[0].clone()], &[0], &[0]).unwrap()
    });
    let (_, mapped_tangent) = jvp(
        &mapped,
        &[floats(&[2], &[3., 4.])],
        &[floats(&[2], &[1., 1.])],
        &[0],
        &[0],
    )
    .unwrap();

    let values = eval(
        &device,
        &[
            out[0].clone(),
            tangent[0].clone(),
            grads[0].clone(),
            mapped_tangent[0].clone(),
        ],
    );
    close(&values[0], &[8.]);
    close(&values[1], &[8.]);
    close(&values[2], &[1., 2., 3., 4.]);
    close(&values[3], &[6., 8.]);
}

/// `cf.call` keeps its boundary under jvp, vjp and map; the results must match
/// the inlined computation.
#[test]
fn transformations_through_a_call() {
    let device = XlaDevice::cpu().unwrap();
    let vector = info(&[2], DType::F32);
    // g(x, y) = x * y behind a call; f(x, y) = g(x, y) * x + y.
    let g = region(&[vector.clone(), vector.clone()], |a| vec![mul(&a[0], &a[1])]);
    let f = region(&[vector.clone(), vector.clone()], |a| {
        let inner = gnx_transforms::call(&g, &[a[0].clone().into(), a[1].clone().into()]);
        vec![add(&mul(&inner[0].resolve(), &a[0]), &a[1])]
    });
    let (x, y) = (floats(&[2], &[2., 3.]), floats(&[2], &[5., 7.]));
    let (tx, ty) = (floats(&[2], &[1., 0.5]), floats(&[2], &[-1., 2.]));
    let (out, tangent) =
        jvp(&f, &[x.clone(), y.clone()], &[tx, ty], &[0, 1], &[0]).unwrap();
    let ct = floats(&[2], &[1., 2.]);
    let (_, grads) = vjp(&f, &[x.clone(), y.clone()], &[ct], &[0, 1], &[0]).unwrap();
    let values = eval(
        &device,
        &[
            out[0].clone(),
            tangent[0].clone(),
            grads[0].clone(),
            grads[1].clone(),
        ],
    );
    // f = x² y + y
    close(&values[0], &[4. * 5. + 5., 9. * 7. + 7.]);
    // 2 x y tx + (x² + 1) ty
    close(
        &values[1],
        &[2. * 2. * 5. * 1. + 5. * -1., 2. * 3. * 7. * 0.5 + 10. * 2.],
    );
    close(&values[2], &[1. * 2. * 2. * 5., 2. * 2. * 3. * 7.]); // ct · 2xy
    close(&values[3], &[1. * 5., 2. * 10.]); // ct · (x² + 1)

    let xs = floats(&[3, 2], &[1., 2., 3., 4., 5., 6.]);
    let ys = floats(&[3, 2], &[1., 1., 2., 2., 3., 3.]);
    let mapped = map(&f, &[xs, ys], &[0, 0], &[0]).unwrap();
    let values = eval(&device, &[mapped[0].clone()]);
    close(
        &values[0],
        &[1. + 1., 4. + 1., 18. + 2., 32. + 2., 75. + 3., 108. + 3.],
    );
}

/// Operations without operands (`zeros`, `ones`, `eye`) need no batching or
/// differentiation rules: their results are unbatched constants, shared by
/// every instance, with zero tangents.
#[test]
fn transformations_with_operand_free_operations() {
    let device = XlaDevice::cpu().unwrap();
    let matrix = info(&[2, 2], DType::F32);
    // f(x) = x * eye + ones + zeros
    let f = region(std::slice::from_ref(&matrix), |a| {
        let eye = apply(gnx_math::ops::eye_op(), &[], matrix.clone());
        let ones = apply(gnx_math::ops::ones_op(), &[], matrix.clone());
        let zeros = apply(gnx_math::ops::zeros_op(), &[], matrix.clone());
        vec![add(&add(&mul(&a[0], &eye), &ones), &zeros)]
    });
    let xs = floats(&[3, 2, 2], &[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]);
    let mapped = map(&f, &[xs], &[0], &[0]).unwrap();
    let x = floats(&[2, 2], &[1., 2., 3., 4.]);
    let t = floats(&[2, 2], &[1., 1., 1., 1.]);
    let (_, tangent) = jvp(&f, &[x.clone()], &[t.clone()], &[0], &[0]).unwrap();
    let (_, grad) = vjp(&f, &[x], &[t], &[0], &[0]).unwrap();
    let values = eval(&device, &[mapped[0].clone(), tangent[0].clone(), grad[0].clone()]);
    close(
        &values[0],
        &[2., 1., 1., 5., 6., 1., 1., 9., 10., 1., 1., 13.],
    );
    close(&values[1], &[1., 0., 0., 1.]);
    close(&values[2], &[1., 0., 0., 1.]);
}

//! `ad` operations lowered natively with `jax.vmap`, `jax.jvp`, and `jax.vjp`,
//! and the same programs expanded by GNX first, giving the same results.

use futures_executor::block_on;
use gnx::Eval;
use gnx_expr::pass::Pipeline;
use gnx_expr::{Capture, Invocation, ObjectTracer, Op, Trace, TraceContext, Value, ValueInfo};
use gnx_jax::jaxpr::trace_expr;
use gnx_math::ops::{add_op, mul_op};
use gnx_math::{ArrayInfo, DType, DataHandle, ExecOpts, HostBuffer, Shape, TransferPolicy};
use gnx_transforms::ExpandTransforms;
use gnx_transforms::ad::region::{jvp, vjp};
/// Record `map.map` at the region level with an inferred size.
fn map(
    body: &Capture,
    inputs: &[Trace],
    in_axes: &[usize],
    out_axes: &[usize],
) -> Result<Vec<Trace>, gnx_expr::LoweringError> {
    gnx_transforms::map::region::map(body, inputs, in_axes, out_axes, None)
}
use gnx_transforms::scan;
use pyo3::prelude::*;

fn info(shape: &[usize]) -> ValueInfo {
    ValueInfo::new(ArrayInfo::new(
        Shape::fixed(shape.iter().copied()),
        DType::F32,
    ))
}
fn f32s(shape: &[usize], values: &[f32]) -> Trace {
    let info = info(shape);
    let bytes = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let storage = HostBuffer::new(info.downcast::<ArrayInfo>().unwrap().clone(), bytes).unwrap();
    Trace::concrete(Value::new(DataHandle::new(storage)), info)
}
fn values(trace: &Trace) -> Vec<f32> {
    trace
        .try_concrete()
        .unwrap()
        .downcast::<DataHandle>()
        .unwrap()
        .to_host()
        .unwrap()
        .bytes()
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect()
}
fn apply(op: Op, a: &Trace, b: &Trace) -> Trace {
    Invocation::invoke(
        op,
        vec![],
        vec![a.clone(), b.clone()],
        vec![a.info().clone()],
    )
    .pop()
    .unwrap()
    .into_trace_ref()
}
fn region(infos: &[ValueInfo], body: impl FnOnce(&[Trace]) -> Vec<Trace>) -> Capture {
    let guard = TraceContext::enter();
    let inputs: Vec<_> = infos.iter().cloned().map(Trace::placeholder).collect();
    let outputs = body(&inputs);
    Capture::from_context(
        guard.context(),
        &inputs
            .into_iter()
            .map(ObjectTracer::from)
            .collect::<Vec<_>>(),
        &outputs
            .into_iter()
            .map(ObjectTracer::from)
            .collect::<Vec<_>>(),
    )
    .unwrap()
}

#[test]
fn jax_lowers_map_jvp_and_vjp_natively() {
    Python::initialize();
    let device = Python::attach(|py| gnx_jax::devices(py, Some("cpu")).unwrap().remove(0));
    let polynomial = region(&[info(&[2]), info(&[2])], |a| {
        vec![apply(add_op(), &apply(mul_op(), &a[0], &a[1]), &a[0])]
    });
    let (x, y) = (f32s(&[2], &[2., 3.]), f32s(&[2], &[5., 7.]));
    let (_, tangent) = jvp(
        &polynomial,
        &[x.clone(), y.clone()],
        &[f32s(&[2], &[1., 0.5]), f32s(&[2], &[-1., 2.])],
        &[0, 1],
        &[0],
    )
    .unwrap();
    let (_, grads) = vjp(
        &polynomial,
        &[x, y],
        &[f32s(&[2], &[1., 2.])],
        &[0, 1],
        &[0],
    )
    .unwrap();
    let square = region(&[info(&[3])], |a| vec![apply(mul_op(), &a[0], &a[0])]);
    let mapped = map(
        &square,
        &[f32s(&[2, 3], &[1., 2., 3., 4., 5., 6.])],
        &[0],
        &[1],
    )
    .unwrap();
    let step = region(&[info(&[]), info(&[])], |a| {
        let next = apply(mul_op(), &a[0], &a[1]);
        vec![next.clone(), next]
    });
    let product = region(&[info(&[]), info(&[3])], |a| {
        scan(&step, &[a[0].clone()], &[a[1].clone()], None)
            .unwrap()
            .0
    });
    let (_, scan_grads) = vjp(
        &product,
        &[f32s(&[], &[2.]), f32s(&[3], &[3., 4., 5.])],
        &[f32s(&[], &[1.])],
        &[0, 1],
        &[0],
    )
    .unwrap();
    let outputs = vec![
        tangent[0].clone(),
        grads[0].clone(),
        grads[1].clone(),
        mapped[0].clone(),
        scan_grads[0].clone(),
        scan_grads[1].clone(),
    ];
    block_on(device.eval(
        &outputs,
        ExecOpts {
            transfers: TransferPolicy::AllowCopy,
        },
    ))
    .unwrap();
    assert_eq!(values(&outputs[0]), [4., 10.]);
    assert_eq!(values(&outputs[1]), [6., 16.]);
    assert_eq!(values(&outputs[2]), [2., 6.]);
    assert_eq!(values(&outputs[3]), [1., 16., 4., 25., 9., 36.]);
    assert_eq!(values(&outputs[4]), [60.]);
    assert_eq!(values(&outputs[5]), [40., 30., 24.]);
}

#[test]
fn jaxprs_keep_transformations_unless_expanded() {
    Python::initialize();
    let guard = TraceContext::enter();
    let square = region(&[info(&[])], |a| vec![apply(mul_op(), &a[0], &a[0])]);
    let x = Trace::placeholder(info(&[]));
    let (_, grad) = vjp(&square, &[x.clone()], &[x.clone()], &[0], &[0]).unwrap();
    let args = [ObjectTracer::from(x)];
    let capture = Capture::from_context(guard.context(), &args, &[grad[0].clone().into()]).unwrap();
    let expr = capture
        .expr()
        .lower(&gnx_expr::InputSpec::from_args(&[], &args))
        .unwrap()
        .expr
        .as_ref()
        .clone();
    let expanded = Pipeline::new()
        .with(ExpandTransforms)
        .run(expr.clone())
        .unwrap();
    // The unexpanded program contains `ad.vjp` (lowered by jax.vjp); the
    // expanded one only `array` operations. Both trace to one-result JAXPRs.
    assert!(expr.to_string().contains("ad::vjp"), "{expr}");
    assert!(!expanded.to_string().contains("ad::"), "{expanded}");
    Python::attach(|py| {
        for program in [&expr, &expanded] {
            let jaxpr = trace_expr(py, program).unwrap().jaxpr().unwrap();
            assert_eq!(jaxpr.outputs().unwrap().len(), 1);
        }
    });
}

#[test]
fn jax_runs_the_jit_based_builders() {
    use gnx::graph::util::dag;
    use gnx::{Array, Callable};
    use gnx_math::ops::reduce_sum_op;
    use gnx_transforms::{grad, hessian, map};

    fn array(trace: Trace) -> Array {
        <Array as gnx::graph::Leaf>::try_from_value(trace)
            .ok()
            .unwrap()
    }
    fn mul(a: &Array, b: &Array) -> Array {
        array(apply(
            mul_op(),
            a.tracer().trace_ref(),
            b.tracer().trace_ref(),
        ))
    }
    fn sum(a: &Array) -> Array {
        array(
            Invocation::invoke(
                reduce_sum_op(&[0]),
                vec![],
                vec![a.tracer().trace_ref().clone()],
                vec![info(&[])],
            )
            .pop()
            .unwrap()
            .into_trace_ref(),
        )
    }

    Python::initialize();
    let device = Python::attach(|py| gnx_jax::devices(py, Some("cpu")).unwrap().remove(0));
    let cube = |x: Array| sum(&mul(&mul(&x, &x), &x));
    let x = array(f32s(&[2], &[1., 2.]));
    let (g,) = grad(cube).invoke((x.clone(),));
    let h = hessian(cube).invoke((x.clone(),));
    let scaled = map(|x: Array, y: Array| mul(&x, &y))
        .in_axes(dag!(0, *))
        .invoke((array(f32s(&[2, 2], &[1., 2., 3., 4.])), x));
    let outputs = vec![
        g.tracer().trace_ref().clone(),
        h.tracer().trace_ref().clone(),
        scaled.tracer().trace_ref().clone(),
    ];
    block_on(device.eval(
        &outputs,
        ExecOpts {
            transfers: TransferPolicy::AllowCopy,
        },
    ))
    .unwrap();
    assert_eq!(values(&outputs[0]), [3., 12.]);
    assert_eq!(values(&outputs[1]), [6., 0., 0., 12.]);
    assert_eq!(values(&outputs[2]), [1., 4., 3., 8.]);
}

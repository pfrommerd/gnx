//! The jit-based transformation builders and macros (`map`, `jvp`, `vjp`,
//! `grad`, `value_and_grad`, `hessian`), executed by the XLA backend.

use std::sync::atomic::{AtomicUsize, Ordering};

use futures_executor::block_on;
use gnx::graph::util::dag;
use gnx::{Array, Callable, Eval, Graph, TryCallable, jit};
use gnx_expr::{Invocation, Op, VarInfo, hexpr};
use gnx_math::ops::{self, add_op, mul_op, reduction::reduce_op, zeros_op};
use gnx_math::{ArrayInfo, DType, DataHandle, ExecOpts, HostBuffer, Shape, TransferPolicy};
use gnx_transforms::{grad, hessian, jvp, map, value_and_grad, vjp};
use gnx_xla::XlaDevice;

const COPY: ExecOpts = ExecOpts {
    transfers: TransferPolicy::AllowCopy,
};

fn floats(dims: &[usize], data: &[f32]) -> Array {
    let info = ArrayInfo::new(Shape::fixed(dims.iter().copied()), DType::F32);
    let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
    Array::from_data(DataHandle::new(
        HostBuffer::copy_from_slice(info, &bytes).unwrap(),
    ))
}

fn scalar(x: f32) -> Array {
    floats(&[], &[x])
}

fn apply(op: Op, inputs: &[&Array], info: ArrayInfo) -> Array {
    Invocation::invoke(
        op,
        vec![],
        inputs.iter().map(|a| a.tracer().clone()).collect(),
        vec![VarInfo::new(info)],
    )
    .pop()
    .unwrap()
    .try_cast::<Array>()
    .ok()
    .unwrap()
    .into()
}

fn info(a: &Array) -> ArrayInfo {
    ArrayInfo::new(a.shape().clone(), a.dtype())
}

fn add(a: &Array, b: &Array) -> Array {
    apply(add_op(), &[a, b], info(a))
}

fn mul(a: &Array, b: &Array) -> Array {
    apply(mul_op(), &[a, b], info(a))
}

fn sum(a: &Array) -> Array {
    let axes: Vec<_> = (0..a.shape().dims().len()).collect();
    let scalar = ArrayInfo::new(Shape::fixed([]), a.dtype());
    let metadata = VarInfo::new(scalar.clone());
    let reducer = hexpr! {
        |x: &metadata, y: &metadata| => {
            result: &metadata = array::add x, y;
            in (result)
        }
    }
    .build()
    .unwrap();
    let initial = apply(zeros_op(), &[], scalar);
    Invocation::invoke(
        reduce_op(&axes),
        vec![(reducer.handle(), vec![])],
        vec![a.tracer().clone(), initial.tracer().clone()],
        vec![metadata],
    )
    .remove(0)
    .try_cast::<Array>()
    .ok()
    .unwrap()
    .into()
}

fn values(device: &XlaDevice, array: &Array) -> Vec<f32> {
    device
        .download(array.data().expect("evaluated array"))
        .unwrap()
        .bytes()
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn eval<G: Graph>(device: &XlaDevice, graph: &G) {
    block_on(device.eval(graph, COPY)).unwrap();
}

#[track_caller]
fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{actual:?} vs {expected:?}");
    for (a, e) in actual.iter().zip(expected) {
        assert!((a - e).abs() < 1e-4, "{actual:?} vs {expected:?}");
    }
}

#[test]
fn map_specs_select_axes_per_argument_leaf() {
    let device = XlaDevice::cpu(gnx_base::cache::Cache::in_memory()).unwrap();
    // f(x, (w, b)) = x * w + b, mapping x over rows, w over columns, sharing b.
    let affine =
        map(|x: Array, wb: (Array, Array)| add(&mul(&x, &wb.0), &wb.1)).in_axes(dag!(0, (1, *)));
    let xs = floats(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
    let w = floats(&[3, 2], &[1., 10., 2., 20., 3., 30.]);
    let b = floats(&[3], &[100., 100., 100.]);
    let out = affine.invoke((xs.clone(), (w, b)));
    assert_eq!(out.shape(), Shape::fixed([2, 3]));

    // The macro form, stacking along axis 1.
    let scaled = gnx_transforms::map!(
        |x: Array, y: Array| mul(&x, &y),
        in_axes = (0, *),
        out_axes = 1
    );
    let columns = scaled(xs, floats(&[3], &[1., 2., 3.]));
    assert_eq!(columns.shape(), Shape::fixed([3, 2]));

    eval(&device, &(out.clone(), columns.clone()));
    close(
        &values(&device, &out),
        &[101., 104., 109., 140., 200., 280.],
    );
    close(&values(&device, &columns), &[1., 4., 4., 10., 9., 18.]);
}

#[test]
fn map_rejects_specs_that_do_not_match_the_arguments() {
    let args = || (floats(&[2], &[1., 2.]), floats(&[2], &[1., 2.]));
    // One entry for two arguments.
    let f = map(|x: Array, y: Array| add(&x, &y)).in_axes(dag!([0]));
    let message = f.try_invoke(args()).err().unwrap().to_string();
    assert!(
        message.contains("in_axes: the spec has no entry for [1]"),
        "{message}"
    );
    // A struct spec for an array argument.
    let f = map(|x: Array, y: Array| add(&x, &y)).in_axes(dag!(0, { "w": 0 }));
    let message = f.try_invoke(args()).err().unwrap().to_string();
    assert!(
        message.contains("more detailed than the value at [1]"),
        "{message}"
    );
    let g = grad(|x: Array| x);
    let message = g
        .try_invoke((floats(&[2], &[1., 2.]),))
        .err()
        .unwrap()
        .to_string();
    assert!(message.contains("one floating-point scalar"), "{message}");
}

#[test]
fn array_arithmetic_differentiates_through_broadcasts() {
    let device = XlaDevice::cpu(gnx_base::cache::Cache::in_memory()).unwrap();
    let f = |x: Array, y: Array| sum(&((&x % &y) / &y - x));
    let (dx, dy) = grad(f).invoke((floats(&[2], &[5., -3.]), scalar(2.)));
    eval(&device, &(dx.clone(), dy.clone()));
    close(&values(&device, &dx), &[-0.5, -0.5]);
    close(&values(&device, &dy), &[-0.5]);
}

#[test]
fn numpy_and_lax_agree_on_bias_values_and_gradients() {
    use gnx::{lax, np};
    let device = XlaDevice::cpu(gnx_base::cache::Cache::in_memory()).unwrap();
    let x = floats(&[2, 2], &[0., 1., -1., 2.]);
    let bias = floats(&[2], &[1., -1.]);
    let t = 1f32.tanh();
    for explicit in [false, true] {
        let scores = move |x: &Array, b: &Array| {
            if explicit {
                let b = lax::broadcast_in_dim(b, x.shape(), &[1]);
                lax::reduce_sum(&lax::tanh(&lax::add(x, &b)), &[1])
            } else {
                np::sum(&np::tanh(&np::add(x, b)), -1)
            }
        };
        let y = scores(&x, &bias);
        let (_, db) = grad(move |x: Array, b: Array| np::sum(&scores(&x, &b), np::ALL))
            .invoke((x.clone(), bias.clone()));
        assert_eq!(y.shape(), Shape::fixed([2]));
        eval(&device, &(y.clone(), db.clone()));
        close(&values(&device, &y), &[t, t]);
        close(&values(&device, &db), &[2. - t * t, 2. - t * t]);
    }
}

#[test]
fn read_only_cell_gradients_follow_identity_and_snapshot_on_each_call() {
    use gnx::array::ArrayRef;
    use gnx::graph::Of;

    let device = XlaDevice::cpu(gnx_base::cache::Cache::in_memory()).unwrap();
    let a = ArrayRef::new(scalar(3.));
    let b = ArrayRef::new(scalar(5.));
    let df = value_and_grad(|a: ArrayRef, b: ArrayRef| a.get() * b.get()).wrt_arg(0);
    let result = df.invoke((a.clone(), b));
    eval(&device, &result);
    close(&values(&device, &result.0), &[15.]);
    close(&values(&device, &result.1.0.get()), &[5.]);
    close(&values(&device, &result.1.1.get()), &[0.]);

    // Same cached trace; different entry value and different alias pattern.
    a.set(scalar(4.));
    let tied = df.invoke((a.clone(), a.clone()));
    eval(&device, &tied);
    close(&values(&device, &tied.0), &[16.]);
    close(&values(&device, &tied.1.0.get()), &[8.]);
    tied.1.0.set(scalar(17.));
    close(&values(&device, &tied.1.1.get()), &[17.]);
    close(&values(&device, &a.get()), &[4.]);

    // A captured handle aliases the explicit input. Returning a read
    // directly must work too (a forwarded SSA result, not an equation).
    let captured = a.clone();
    let (dx,) = grad(|x: ArrayRef| x.get() * captured.get())
        .wrt(Of::<ArrayRef>::filter())
        .invoke((a.clone(),));
    eval(&device, &dx);
    close(&values(&device, &dx.get()), &[8.]);
    let square_grad = jit(|x: ArrayRef| grad(|z: ArrayRef| z.get() * z.get()).invoke((x,)).0.get());
    let dx = square_grad.invoke((a.clone(),));
    eval(&device, &dx);
    close(&values(&device, &dx), &[8.]);

    let (dx,) = grad(|x: ArrayRef| x.get()).invoke((a,));
    eval(&device, &dx);
    close(&values(&device, &dx.get()), &[1.]);
}

#[test]
fn indexed_reads_updates_and_gradients() {
    use gnx::{array::ArrayRef, lax, np};
    let device = XlaDevice::cpu(gnx_base::cache::Cache::in_memory()).unwrap();
    let x = floats(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
    let indices = lax::zeros([1, 2], DType::I32);
    let y = x.at((.., &indices));
    assert_eq!(y.shape(), Shape::fixed([2, 1, 2]));
    eval(&device, &y);
    close(&values(&device, &y), &[1., 1., 4., 4.]);
    let (dx,) = grad(|x: Array| np::sum(&x.at((.., &indices)), np::ALL)).invoke((x.clone(),));
    eval(&device, &dx);
    close(&values(&device, &dx), &[2., 0., 0., 2., 0., 0.]);
    assert!(x.try_at((.., 3)).is_err());
    assert!(x.try_at((&indices, &indices)).is_err());
    let cell = ArrayRef::new(x);
    assert!(cell.try_set_at((1, ..), &scalar(0.)).is_err());
    let swap = jit(|c: ArrayRef, v: Array| {
        let before = c.at((1, 1..=2));
        c.set_at((1, 1..), &v);
        before
    });
    for (new, old) in [([7., 8.], [5., 6.]), ([9., 10.], [7., 8.])] {
        let before = swap.invoke((cell.clone(), floats(&[2], &new)));
        eval(&device, &before);
        close(&values(&device, &before), &old);
    }
    let final_value = cell.get();
    eval(&device, &final_value);
    close(&values(&device, &final_value), &[1., 2., 3., 4., 9., 10.]);
}

#[test]
fn cell_grad_keeps_value_arguments_positional_and_rejects_writes() {
    use gnx::array::ArrayRef;
    let device = XlaDevice::cpu(gnx_base::cache::Cache::in_memory()).unwrap();
    let x = scalar(2.);
    let cell = ArrayRef::new(scalar(3.));
    let result = grad(|x: Array, y: Array, c: ArrayRef| x + y * c.get()).invoke((
        x.clone(),
        x,
        cell.clone(),
    ));
    eval(&device, &result);
    close(&values(&device, &result.0), &[1.]);
    close(&values(&device, &result.1), &[3.]);
    close(&values(&device, &result.2.get()), &[2.]);

    let write = grad(|x: ArrayRef| {
        x.set(scalar(99.));
        x.get()
    });
    let err = write.try_invoke((cell.clone(),)).err().unwrap().to_string();
    assert!(err.contains("cannot write"), "{err}");
    let write_capture = grad(|x: Array| {
        cell.set(x.clone());
        x
    });
    let err = write_capture
        .try_invoke((scalar(42.),))
        .err()
        .unwrap()
        .to_string();
    assert!(err.contains("cannot write"), "{err}");
    close(&values(&device, &cell.get()), &[3.]);
}

#[test]
fn affine_model_differentiates_only_params() {
    use gnx::{
        graph::Of,
        nn::{Linear, Param},
        np,
    };

    #[derive(Clone, Graph)]
    struct Model {
        head: Linear,
        scale: Array,
    }

    let device = XlaDevice::cpu(gnx_base::cache::Cache::in_memory()).unwrap();
    let model = Model {
        head: Linear::new(
            floats(&[2, 3], &[1., 2., 3., 3., 2., 1.]),
            Some(floats(&[2], &[1., -1.])),
        ),
        scale: scalar(2.),
    };
    let x = floats(&[2, 3], &[1., 0., 1., 0., 2., 1.]);
    let loss = value_and_grad(|m: Model, x: Array| np::sum(&m.head.forward(x), np::ALL) * m.scale)
        .wrt(Of::<Param>::filter());
    let result = loss.invoke((model, x));
    eval(&device, &result);
    let (loss, (dm, dx)) = result;
    close(&values(&device, &loss), &[40.]);
    close(
        &values(&device, &dm.head.weight.value()),
        &[2., 4., 4., 2., 4., 4.],
    );
    close(&values(&device, &dm.head.bias().unwrap().value()), &[4., 4.]);
    close(&values(&device, &dm.scale), &[0.]);
    close(&values(&device, &dx), &[0.; 6]);
}

#[test]
fn grad_and_value_and_grad_follow_wrt() {
    let device = XlaDevice::cpu(gnx_base::cache::Cache::in_memory()).unwrap();
    let traces = AtomicUsize::new(0);
    // f(x, y) = sum(x * x * y)
    let f = |x: Array, y: Array| {
        traces.fetch_add(1, Ordering::Relaxed);
        sum(&mul(&mul(&x, &x), &y))
    };
    let both = grad(f);
    let only_x = grad(f).wrt_arg(0);
    let (x, y) = (floats(&[2], &[1., 2.]), floats(&[2], &[3., 4.]));
    let (dx, dy) = both.invoke((x.clone(), y.clone()));
    let (dx_only, dy_zero) = only_x.invoke((x.clone(), y.clone()));
    let (dx_zero, dy_only) = grad(f).wrt_arg(1).invoke((x.clone(), y.clone()));
    eval(&device, &(dx_zero.clone(), dy_only.clone()));
    close(&values(&device, &dx_zero), &[0., 0.]);
    close(&values(&device, &dy_only), &[1., 4.]);
    // Same signature: the wrapper's trace is reused.
    let before = traces.load(Ordering::Relaxed);
    let _ = both.invoke((x.clone(), y.clone()));
    assert_eq!(traces.load(Ordering::Relaxed), before);

    let (value, (gx, _)) = value_and_grad(f).invoke((x.clone(), y.clone()));
    let macro_grad = gnx_transforms::grad!(
        |x: Array, y: Array| sum(&mul(&mul(&x, &x), &y)),
        wrt_arg = 1
    );
    let (_, dy_macro) = macro_grad(x, y);

    eval(
        &device,
        &(dx.clone(), dy.clone(), dx_only.clone(), dy_zero.clone()),
    );
    eval(&device, &(value.clone(), gx.clone(), dy_macro.clone()));
    close(&values(&device, &dx), &[6., 16.]); // 2xy
    close(&values(&device, &dy), &[1., 4.]); // x^2
    close(&values(&device, &dx_only), &[6., 16.]);
    close(&values(&device, &dy_zero), &[0., 0.]);
    close(&values(&device, &value), &[1. * 3. + 4. * 4.]);
    close(&values(&device, &gx), &[6., 16.]);
    close(&values(&device, &dy_macro), &[1., 4.]);
}

#[test]
fn jvp_vjp_and_pullback() {
    let device = XlaDevice::cpu(gnx_base::cache::Cache::in_memory()).unwrap();
    // f(x, y) = x * y + x
    let f = |x: Array, y: Array| add(&mul(&x, &y), &x);
    let (x, y) = (floats(&[2], &[2., 3.]), floats(&[2], &[5., 7.]));
    let (out, tangent) = jvp(f).invoke((
        (x.clone(), y.clone()),
        (floats(&[2], &[1., 0.5]), floats(&[2], &[-1., 2.])),
    ));
    let (_, (cx, cy)) = vjp(f).invoke(((x.clone(), y.clone()), floats(&[2], &[1., 2.])));
    let pullbacks = vjp(f).wrt_arg(1);
    let (primal, pullback) = pullbacks.pullback((x, y));
    let (zero, cy_only) = pullback(floats(&[2], &[1., 1.]));

    eval(
        &device,
        &(out.clone(), tangent.clone(), cx.clone(), cy.clone()),
    );
    eval(&device, &(primal.clone(), zero.clone(), cy_only.clone()));
    close(&values(&device, &out), &[12., 24.]);
    close(
        &values(&device, &tangent),
        &[1. * 5. - 2. + 1., 0.5 * 7. + 3. * 2. + 0.5],
    );
    close(&values(&device, &cx), &[6., 16.]);
    close(&values(&device, &cy), &[2., 6.]);
    close(&values(&device, &primal), &[12., 24.]);
    close(&values(&device, &zero), &[0., 0.]);
    close(&values(&device, &cy_only), &[2., 3.]);
}

#[test]
fn transformations_compose() {
    let device = XlaDevice::cpu(gnx_base::cache::Cache::in_memory()).unwrap();
    let cube = |x: Array| mul(&mul(&x, &x), &x);
    // d²/dx² x³ = 6x, as grad of grad.
    let second = grad(grad(cube));
    let (d2,) = second.invoke((scalar(2.),));
    // Hessian of sum(x³) is diag(6x).
    let h = hessian(|x: Array| sum(&cube(x))).invoke((floats(&[2], &[1., 2.]),));
    assert_eq!(h.shape(), Shape::fixed([2, 2]));
    // Per-example gradients of w * x with respect to the shared w.
    let per_example = map(grad(|w: Array, x: Array| mul(&w, &x)).wrt_arg(0)).in_axes(dag!(*, 0));
    let (dw, _) = per_example.invoke((scalar(3.), floats(&[4], &[1., 2., 3., 4.])));
    // jit of grad.
    let jitted = jit(grad(cube));
    let (d1,) = jitted.invoke((scalar(2.),));

    eval(&device, &(d2.clone(), h.clone(), dw.clone(), d1.clone()));
    close(&values(&device, &d2), &[12.]);
    close(&values(&device, &h), &[6., 0., 0., 12.]);
    close(&values(&device, &dw), &[1., 2., 3., 4.]);
    close(&values(&device, &d1), &[12.]);
}

#[test]
fn stop_gradient_passes_through_the_primal_and_stops_both_derivatives() {
    let device = XlaDevice::cpu(gnx_base::cache::Cache::in_memory()).unwrap();
    let f = |x: Array| mul(&x, &gnx::stop_gradient(&x));
    let (y, dy) = jvp(f).invoke(((scalar(3.),), (scalar(2.),)));
    let (_, (dx,)) = vjp(f).invoke(((scalar(3.),), scalar(1.)));
    eval(&device, &(y.clone(), dy.clone(), dx.clone()));
    close(&values(&device, &y), &[9.]);
    close(&values(&device, &dy), &[6.]);
    close(&values(&device, &dx), &[3.]);
}

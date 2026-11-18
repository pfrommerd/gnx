use std::collections::BTreeMap;
use std::future::{Future, ready};
use std::io;
use std::sync::{
    Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::task::{Context, Poll, Waker};

use gnx::array::{ArrayInfo, ArrayRef, DType, Shape};
use gnx::buffer::{DataHandle, HostBuffer};
use gnx::expr::{Expr, Invocation, Value};
use gnx::{Array, Device, Eval, EvalError, EvalPlanError, ExecOpts, Graph};

fn array(n: i32) -> Array {
    Array::from_data(buffer(n))
}

fn buffer(n: i32) -> DataHandle {
    let info = ArrayInfo::new(Shape::fixed([]), DType::I32);
    DataHandle::new(HostBuffer::new(info, n.to_le_bytes().to_vec()).unwrap())
}

fn number(data: &DataHandle) -> i32 {
    i32::from_le_bytes(data.to_host().unwrap().bytes().try_into().unwrap())
}

fn add(left: &Array, right: &Array) -> Array {
    let info = left.tracer().trace_ref().info().clone();
    let result = Invocation::invoke(
        gnx::ops::add_op(),
        vec![],
        vec![left.tracer().clone(), right.tracer().clone()],
        vec![info],
    )
    .pop()
    .unwrap();
    result.try_cast::<Array>().ok().unwrap().into()
}

fn run<T>(future: impl Future<Output = T>) -> T {
    match std::pin::pin!(future).poll(&mut Context::from_waker(Waker::noop())) {
        Poll::Ready(value) => value,
        Poll::Pending => panic!("mock device must be immediately ready"),
    }
}

#[derive(Default)]
struct MockDevice {
    calls: Mutex<Vec<(usize, usize)>>,
    fail: AtomicBool,
}

impl Device for MockDevice {
    // Device-specific iterable: it needn't be Vec<Value>.
    type Outputs = std::vec::IntoIter<Value>;

    fn execute(
        &self,
        expr: Expr,
        args: impl IntoIterator<Item = Value>,
        _: ExecOpts,
    ) -> impl Future<Output = io::Result<Self::Outputs>> + Send {
        self.calls
            .lock()
            .unwrap()
            .push((expr.eqns().len(), expr.outputs().len()));
        if self.fail.load(Ordering::Relaxed) {
            return ready(Err(io::Error::other("device refused to execute")));
        }
        let mut values: BTreeMap<_, _> = expr
            .closure()
            .iter()
            .chain(expr.inputs())
            .zip(args)
            .map(|((var, _), value)| (*var, number(value.downcast::<DataHandle>().unwrap())))
            .collect();
        for eqn in expr.eqns() {
            let [left, right] = eqn.inputs() else {
                panic!("expected two add operands");
            };
            let result = values[left] + values[right];
            values.insert(eqn.outputs()[0].0, result);
        }
        let outputs = expr
            .outputs()
            .iter()
            .map(|var| Value::new(buffer(values[var])))
            .collect::<Vec<_>>();
        ready(Ok(outputs.into_iter()))
    }
}

// Exercise the Eval API without needing to name Device in client bounds.
fn evaluate<D: Eval, G: Graph>(device: &D, roots: &G) -> Result<(), EvalError> {
    run(device.eval(roots, ExecOpts::default()))
}

#[test]
fn eval_can_be_implemented_without_a_device() {
    struct NoDevice;
    impl Eval for NoDevice {
        fn eval<G: Graph>(
            &self,
            _: &G,
            _: ExecOpts,
        ) -> impl Future<Output = Result<(), EvalError>> + Send {
            ready(Ok(()))
        }
    }
    evaluate(&NoDevice, &()).unwrap();
}

#[test]
fn expr_captures_and_prints_a_graph_without_materializing_it() {
    let (a, b) = (array(2), array(3));
    let z = add(&a, &b);
    let x = add(&z, &z);
    let cell = ArrayRef::new(x.tracer().clone());

    let capture = gnx::expr(&(cell, x.clone())).unwrap();
    assert_eq!(capture.outputs().len(), 2); // x (deduplicated), then live z
    let pretty = capture.to_string();
    assert_eq!(pretty.matches("array::add").count(), 2);
    assert_eq!(pretty, capture.expr().to_string());
    assert_eq!(format!("{capture:#?}"), pretty);
    assert!(x.data().is_none());
    assert!(z.data().is_none());
}

#[test]
fn eval_materializes_live_tracers_and_uses_a_single_device_call() {
    let device = MockDevice::default();
    let (a, b) = (array(2), array(3));
    let z = add(&a, &b);
    let x = add(&z, &z);
    evaluate(&device, &x).unwrap();
    assert_eq!(number(z.data().unwrap()), 5);
    assert_eq!(number(x.data().unwrap()), 10);
    evaluate(&device, &(x, z, a, b)).unwrap();
    assert_eq!(*device.calls.lock().unwrap(), [(2, 2)]);
}

#[test]
fn cell_roots_evaluate_their_home_snapshot_without_rebinding() {
    let device = MockDevice::default();
    let a = array(2);
    let z = add(&a, &a);
    let cell = ArrayRef::new(z.tracer().clone());
    let next = add(&a, &z);
    let future = device.eval(&cell, ExecOpts::default());
    cell.set(next.tracer().clone());
    run(future).unwrap();
    assert_eq!(number(z.data().unwrap()), 4);
    assert!(next.data().is_none());
    assert!(cell.get().data().is_none());
    evaluate(&device, &cell).unwrap();
    assert_eq!(number(cell.get().data().unwrap()), 6);
    assert_eq!(*device.calls.lock().unwrap(), [(1, 1), (1, 1)]);

    let placeholder: ArrayRef =
        gnx::expr::CellTracer::placeholder(a.tracer().trace_ref().info().clone()).into();
    assert!(matches!(
        evaluate(&device, &placeholder),
        Err(EvalError::Plan(EvalPlanError::UnboundPlaceholder))
    ));
}

#[test]
fn backend_failure_keeps_the_deferred_slice_for_retry() {
    let device = MockDevice::default();
    let a = array(2);
    let z = add(&a, &a);
    let x = add(&z, &z);
    device.fail.store(true, Ordering::Relaxed);
    assert!(matches!(evaluate(&device, &x), Err(EvalError::Backend(_))));
    assert!(x.data().is_none());
    assert!(z.data().is_none());
    device.fail.store(false, Ordering::Relaxed);
    evaluate(&device, &x).unwrap();
    assert_eq!(number(x.data().unwrap()), 8);
}

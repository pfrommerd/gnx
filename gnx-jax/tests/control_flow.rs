use futures_executor::block_on;
use gnx::Eval;
use gnx_expr::{
    Capture, CellTrace, Dialect, Eqn, Invocation, ObjectTracer, Op, Operation, Trace, TraceContext,
    Value, ValueInfo,
};
use gnx_jax::jaxpr::{self, JaxLowering, TraceContext as JaxContext, TraceValue};
use gnx_jax::{JaxDevice, register_jax_lowering};
use gnx_math::{
    ArrayInfo, DType, DataHandle, ExecOpts, HostBuffer, Shape, TransferPolicy, ops::add_op,
};
use gnx_transforms::{cond, scan, switch, while_loop};
use pyo3::prelude::*;

fn info(shape: &[usize], dtype: DType) -> ValueInfo {
    ValueInfo::new(ArrayInfo::new(Shape::fixed(shape.iter().copied()), dtype))
}

fn host(shape: &[usize], dtype: DType, bytes: Vec<u8>) -> Trace {
    let info = info(shape, dtype);
    let storage = HostBuffer::new(info.downcast::<ArrayInfo>().unwrap().clone(), bytes).unwrap();
    Trace::concrete(Value::new(DataHandle::new(storage)), info)
}

fn f32s(shape: &[usize], values: &[f32]) -> Trace {
    host(
        shape,
        DType::F32,
        values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    )
}

fn scalar(value: f32) -> Trace {
    f32s(&[], &[value])
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

fn add(lhs: &Trace, rhs: &Trace) -> Trace {
    Invocation::invoke(
        add_op(),
        vec![],
        vec![lhs.clone(), rhs.clone()],
        vec![lhs.info().clone()],
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

fn evaluate(device: &JaxDevice, outputs: Vec<Trace>) {
    block_on(device.eval(
        &outputs,
        ExecOpts {
            transfers: TransferPolicy::AllowCopy,
        },
    ))
    .unwrap();
}

struct TestDialect;
#[derive(Clone, PartialEq, Eq, Hash)]
struct LessThan;
impl Dialect for TestDialect {
    fn name(&self) -> &'static str {
        "control_flow_test"
    }
}
impl std::fmt::Display for LessThan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}::{}", Operation::dialect(self).name(), Operation::name(self))
    }
}

impl Operation for LessThan {
    fn name(&self) -> &'static str {
        "lt"
    }
    fn dialect(&self) -> &'static dyn Dialect {
        &TestDialect
    }
}
impl JaxLowering for LessThan {
    fn lower_jax<'py>(
        &self,
        ctx: &mut JaxContext<'py>,
        _: &Eqn,
        operands: &[TraceValue<'py>],
    ) -> jaxpr::Result<Vec<TraceValue<'py>>> {
        let result = ctx
            .python()
            .import("jax.lax")?
            .call_method1("lt", (operands[0].as_native(), operands[1].as_native()))?;
        Ok(vec![TraceValue::from_native(result)])
    }
}
register_jax_lowering!(LessThan);
fn less_than(lhs: Trace, rhs: Trace) -> Trace {
    Invocation::invoke(
        Op::new(LessThan),
        vec![],
        vec![lhs, rhs],
        vec![info(&[], DType::Bool)],
    )
    .pop()
    .unwrap()
    .into_trace_ref()
}

#[test]
fn jax_runs_cond_switch_while_and_scan_with_private_cell_state() {
    Python::initialize();
    let device = Python::attach(|py| gnx_jax::devices(py, Some("cpu")).unwrap().remove(0));
    let f32_scalar = info(&[], DType::F32);

    for (predicate, expected) in [(false, 1.), (true, 5.)] {
        let cell = CellTrace::new(scalar(1.));
        let on_true = region(std::slice::from_ref(&f32_scalar), |args| {
            let result = add(&cell.get(), &args[0]);
            cell.set(result.clone());
            vec![result]
        });
        let on_false = region(std::slice::from_ref(&f32_scalar), |_| vec![cell.get()]);
        let flag = host(&[], DType::Bool, vec![u8::from(predicate)]);
        let result = cond(flag, &on_true, &on_false, &[scalar(4.).into()]).unwrap();
        assert_eq!(result.len(), 1);
        evaluate(&device, vec![result[0].clone(), cell.get()]);
        assert_eq!(values(&result[0]), [expected]);
        assert_eq!(values(&cell.get()), [expected]);
    }

    for (predicate, expected) in [(false, [1., 7.]), (true, [5., 2.])] {
        let a = CellTrace::new(scalar(1.));
        let b = CellTrace::new(scalar(2.));
        let on_true = region(&[], |_| {
            a.set(scalar(5.));
            vec![]
        });
        let on_false = region(&[], |_| {
            b.set(scalar(7.));
            vec![]
        });
        let flag = host(&[], DType::Bool, vec![u8::from(predicate)]);
        assert!(cond(flag, &on_true, &on_false, &[]).unwrap().is_empty());
        evaluate(&device, vec![a.get(), b.get()]);
        assert_eq!(values(&a.get()), [expected[0]]);
        assert_eq!(values(&b.get()), [expected[1]]);
    }

    let first = region(std::slice::from_ref(&f32_scalar), |arg| arg.to_vec());
    let second = region(std::slice::from_ref(&f32_scalar), |arg| {
        vec![add(&arg[0], &arg[0])]
    });
    for (index, expected) in [(-9i32, 4.), (1, 8.), (9, 8.)] {
        let index = host(&[], DType::I32, index.to_le_bytes().to_vec());
        let result = switch(index, &[&first, &second], &[scalar(4.).into()]).unwrap();
        evaluate(&device, result.clone());
        assert_eq!(values(&result[0]), [expected]);
    }

    for (initial_state, expected) in [(0., [3., 6.]), (3., [0., 0.])] {
        let cell = CellTrace::new(scalar(initial_state));
        let carry = [f32_scalar.clone(), f32_scalar.clone()];
        let predicate = region(&carry, |_| vec![less_than(cell.get(), scalar(3.))]);
        let body = region(&carry, |args| {
            cell.set(add(&cell.get(), &scalar(1.)));
            vec![add(&args[0], &scalar(1.)), add(&args[1], &scalar(2.))]
        });
        // Equal initial SSA values do not mean the two carries stay equal.
        let zero = scalar(0.);
        let result = while_loop(&predicate, &body, &[zero.clone(), zero]).unwrap();
        assert_eq!(result.len(), 2);
        let mut roots = result.clone();
        roots.push(cell.get());
        evaluate(&device, roots);
        assert_eq!(values(&result[0]), [expected[0]]);
        assert_eq!(values(&result[1]), [expected[1]]);
        assert_eq!(values(&cell.get()), [3.]);
    }

    for (xs, carry_expected, ys_expected, state_expected) in [
        (vec![1., 2., 3.], 6., vec![11., 14., 20.], 20.),
        (vec![], 0., vec![], 10.),
    ] {
        let cell = CellTrace::new(scalar(10.));
        let step = region(&[f32_scalar.clone(), f32_scalar.clone()], |args| {
            let carry = add(&args[0], &args[1]);
            let state = add(&cell.get(), &carry);
            cell.set(state.clone());
            vec![carry, state]
        });
        let (carry, ys) = scan(&step, &[scalar(0.)], &[f32s(&[xs.len()], &xs)], None).unwrap();
        assert_eq!((carry.len(), ys.len()), (1, 1));
        assert_eq!(ys[0].info(), &info(&[xs.len()], DType::F32));
        evaluate(&device, vec![carry[0].clone(), ys[0].clone(), cell.get()]);
        assert_eq!(values(&carry[0]), [carry_expected]);
        if !xs.is_empty() {
            assert_eq!(values(&ys[0]), ys_expected);
        }
        assert_eq!(values(&cell.get()), [state_expected]);
    }

    let step = region(&[f32_scalar], |args| {
        let carry = add(&args[0], &scalar(1.));
        vec![carry.clone(), carry]
    });
    let (carry, ys) = scan(&step, &[scalar(0.)], &[], Some(2)).unwrap();
    evaluate(&device, vec![carry[0].clone(), ys[0].clone()]);
    assert_eq!(values(&carry[0]), [2.]);
    assert_eq!(values(&ys[0]), [1., 2.]);
}

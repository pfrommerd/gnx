#![cfg(feature = "native")]

use futures_executor::block_on;
use gnx::Eval;
use gnx_expr::{
    Capture, Dialect, Eqn, Expr, Invocation, ObjectTracer, Op, Operation, SubExpr, Trace,
    TraceContext, Value, ValueInfo, VarScope,
};
use gnx_math::{
    ArrayInfo, DType, DataHandle, ExecOpts, HostBuffer, HostId, LocalAllocations, MemoryLocation,
    Shape, ShardStorage, TransferPolicy, ops::add_op,
};
use gnx_transforms::{cond, control_flow::call_op, scan, switch, while_loop};
use gnx_xla::{LoweringContext, XlaDevice, XlaLowering, XlaOp, register_xla_lowering};

const COPY: ExecOpts = ExecOpts {
    transfers: TransferPolicy::AllowCopy,
};

fn array_info(dims: &[usize], dtype: DType) -> ArrayInfo {
    ArrayInfo::new(Shape::fixed(dims.iter().copied()), dtype)
}
fn host(dims: &[usize], dtype: DType, bytes: Vec<u8>) -> DataHandle {
    DataHandle::new(HostBuffer::new(array_info(dims, dtype), bytes).unwrap())
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
fn compile_nested_call_reuse_outputs_and_forward_remote_refs() {
    let device = XlaDevice::cpu().unwrap();
    let array = array_info(&[2], DType::F32);
    let ty = ValueInfo::new(array.clone());
    let mut vars = VarScope::new();
    let [a, b, r, sum] = std::array::from_fn(|_| vars.create_var());
    let child = Expr::new(
        vec![(a, ty.clone())],
        vec![(b, ty.clone())],
        vec![Eqn::new(add_op(), vec![a, b], vec![(sum, ty.clone())])],
        vec![sum, sum],
    );
    let [c, d, dead] = std::array::from_fn(|_| vars.create_var());
    let program = Expr::new(
        vec![(a, ty.clone())],
        vec![(b, ty.clone()), (r, ty.clone())],
        vec![
            Eqn::new_with_subexprs(
                call_op(),
                vec![SubExpr::new(child.into(), vec![a])],
                vec![b],
                vec![(c, ty.clone()), (d, ty.clone())],
            ),
            Eqn::new(add_op(), vec![r, r], vec![(dead, ty)]),
        ],
        vec![d, r, c, d],
    );
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
    assert_eq!(
        compiled
            .run(
                &[left.clone(), right.clone(), remote.clone()],
                ExecOpts::default()
            )
            .unwrap_err()
            .kind(),
        std::io::ErrorKind::Unsupported
    );
    let outputs = compiled.run(&[left, right, remote.clone()], COPY).unwrap();
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
    let info = ValueInfo::new(data.info());
    Trace::concrete(Value::new(data), info)
}
fn scalar(n: f32) -> Trace {
    trace(floats(&[], &[n]))
}
fn add(a: &Trace, b: &Trace) -> Trace {
    invoke(add_op(), a, b, a.info().clone())
}
fn invoke(op: Op, a: &Trace, b: &Trace, info: ValueInfo) -> Trace {
    Invocation::invoke(op, vec![], vec![a.clone(), b.clone()], vec![info])
        .pop()
        .unwrap()
        .into_trace_ref()
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

struct LessThan;
struct TestDialect;
impl Dialect for TestDialect {
    fn name(&self) -> &'static str {
        "xla_test"
    }
    fn create(&self, _: &'static str) -> Option<&'static dyn Operation> {
        None
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
register_xla_lowering!(LessThan);

#[test]
fn runs_structured_regions() {
    let device = XlaDevice::cpu().unwrap();
    let f32_scalar = ValueInfo::new(array_info(&[], DType::F32));
    let bool_scalar = ValueInfo::new(array_info(&[], DType::Bool));
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
        vec![invoke(Op::new(&LessThan), &a[0], &scalar(5.), bool_scalar)]
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

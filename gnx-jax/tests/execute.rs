use std::fmt;
use std::io;
use std::rc::Rc;
use std::sync::Arc;

use futures_executor::block_on;
use gnx::Eval;
use gnx_expr::{
    Dialect, Eqn, Expr, Kind, Op, Operation, Ownership, SubExpr, Value, ValueInfo, VarScope,
    builtins::call_op,
};
use gnx_jax::jaxpr::{self, JaxLowering, JaxLoweringError, TraceContext, TraceValue};
use gnx_jax::{JaxBuffer, dlpack, register_jax_lowering};
use gnx_math::{
    ArrayInfo, BufferInfo, BufferLayout, DType, DataHandle, DataImpl, Device, DlpackDevice,
    DlpackRequest, DlpackTensor, ExecOpts, HostBuffer, MemoryLocation, Shape, TransferPolicy,
    ops::add_op,
};
use pyo3::prelude::*;

fn info() -> ArrayInfo {
    ArrayInfo::new(Shape::fixed([2]), DType::F32)
}

fn host(values: [f32; 2]) -> Value {
    let bytes = values.into_iter().flat_map(f32::to_le_bytes).collect();
    Value::new(DataHandle::new(HostBuffer::new(info(), bytes).unwrap()))
}

fn unpack(data: &DataHandle) -> Vec<f32> {
    data.to_host()
        .unwrap()
        .bytes()
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect()
}

fn nested() -> Expr {
    use Ownership::{Borrow, Move};
    let ty = ValueInfo::new(info());
    let mut child = VarScope::new();
    let bias = child.create_var();
    let input = child.create_var();
    let sum = child.create_var();
    let child = Expr::new(
        vec![(bias, Borrow, ty.clone())],
        vec![(input, Borrow, ty.clone())],
        vec![Eqn::new(
            add_op(),
            vec![(input, Borrow), (bias, Borrow)],
            vec![(sum, Kind::Value(ty.clone()))],
        )],
        vec![sum],
    );
    let mut parent = VarScope::new();
    let bias = parent.create_var();
    let input = parent.create_var();
    let sum = parent.create_var();
    let doubled = parent.create_var();
    Expr::new(
        vec![(bias, Borrow, ty.clone())],
        vec![(input, Borrow, ty.clone())],
        vec![
            Eqn::new_with_subexprs(
                call_op(),
                vec![SubExpr::new(Arc::new(child), vec![(bias, Borrow)])],
                vec![(input, Borrow)],
                vec![(sum, Kind::Value(ty.clone()))],
            ),
            Eqn::new(
                Op::new(&DOUBLE),
                vec![(sum, Move)],
                vec![(doubled, Kind::Value(ty))],
            ),
        ],
        vec![doubled],
    )
}

struct TestDialect;
static TEST: TestDialect = TestDialect;
static DOUBLE: Double = Double;
static UNSUPPORTED: Unsupported = Unsupported;
static DUPLICATE: Duplicate = Duplicate;

impl Dialect for TestDialect {
    fn name(&self) -> &'static str {
        "test"
    }

    fn create(&self, _: &'static str) -> Option<&'static dyn Operation> {
        None
    }
}

struct Double;
impl Operation for Double {
    fn name(&self) -> &'static str {
        "double"
    }
    fn dialect(&self) -> &'static dyn Dialect {
        &TEST
    }
}
impl JaxLowering for Double {
    fn lower_jax<'py>(
        &self,
        ctx: &mut TraceContext<'py>,
        _: &Eqn,
        operands: &[TraceValue<'py>],
    ) -> jaxpr::Result<Vec<TraceValue<'py>>> {
        Ok(vec![ctx.lax()?.add(&operands[0], &operands[0])?])
    }
}
register_jax_lowering!(Double);

struct Unsupported;
impl Operation for Unsupported {
    fn name(&self) -> &'static str {
        "unsupported"
    }
    fn dialect(&self) -> &'static dyn Dialect {
        &TEST
    }
}

struct Duplicate;
impl Operation for Duplicate {
    fn name(&self) -> &'static str {
        "duplicate"
    }
    fn dialect(&self) -> &'static dyn Dialect {
        &TEST
    }
}
impl JaxLowering for Duplicate {
    fn lower_jax<'py>(
        &self,
        _: &mut TraceContext<'py>,
        _: &Eqn,
        _: &[TraceValue<'py>],
    ) -> jaxpr::Result<Vec<TraceValue<'py>>> {
        panic!("duplicate lowering must not be called")
    }
}
register_jax_lowering!(Duplicate);
register_jax_lowering!(Duplicate);

fn one_rule(op: Op, info: ArrayInfo) -> Expr {
    let mut vars = VarScope::new();
    let x = vars.create_var();
    let result = vars.create_var();
    let info = ValueInfo::new(info);
    Expr::new(
        vec![],
        vec![(x, Ownership::Borrow, info.clone())],
        vec![Eqn::new(
            op,
            vec![(x, Ownership::Borrow)],
            vec![(result, Kind::Value(info))],
        )],
        vec![result],
    )
}

#[test]
fn jax_cpu_traces_executes_and_transfers() {
    Python::initialize();
    let expr = nested();
    let device = Python::attach(|py| {
        let device = gnx_jax::devices(py, Some("cpu")).unwrap().remove(0);
        let closed = jaxpr::trace_expr(py, &expr).unwrap();
        let jaxpr = closed.jaxpr().unwrap();
        assert_eq!(jaxpr.inputs().unwrap().len(), 2);
        assert_eq!(jaxpr.outputs().unwrap().len(), 1);
        assert!(closed.consts().unwrap().is_empty());
        assert_eq!(
            jaxpr
                .equations()
                .unwrap()
                .iter()
                .map(|eqn| eqn.primitive().unwrap())
                .collect::<Vec<_>>(),
            ["add", "add"]
        );
        for (operation, message) in [
            (
                &UNSUPPORTED as &dyn Operation,
                "no JAX lowering for test.unsupported",
            ),
            (
                &DUPLICATE,
                "multiple JAX lowerings registered for test.duplicate",
            ),
        ] {
            let error = jaxpr::trace_expr(py, &one_rule(Op::new(operation), info())).unwrap_err();
            assert!(matches!(error, JaxLoweringError::Equation { index: 0, .. }));
            assert!(error.to_string().contains(message));
        }
        let dynamic = ArrayInfo::new([gnx_math::Dim::Dynamic].into_iter().collect(), DType::F32);
        assert!(
            jaxpr::trace_expr(py, &one_rule(Op::new(&DOUBLE), dynamic))
                .unwrap_err()
                .to_string()
                .contains("fixed dimensions")
        );
        let jax = py.import("jax").unwrap();
        if !jax
            .getattr("config")
            .unwrap()
            .getattr("jax_enable_x64")
            .unwrap()
            .extract::<bool>()
            .unwrap()
        {
            let f64 = ArrayInfo::new(Shape::fixed([2]), DType::F64);
            assert!(
                jaxpr::trace_expr(py, &one_rule(Op::new(&DOUBLE), f64))
                    .unwrap_err()
                    .to_string()
                    .contains("would change dtype float64")
            );
        }
        device
    });
    let opts = ExecOpts {
        transfers: TransferPolicy::AllowCopy,
    };
    for (bias, expected) in [([1.0, 2.0], [8.0, 12.0]), ([5.0, 6.0], [16.0, 20.0])] {
        // A non-Send iterator is permitted even though the returned future is Send.
        let non_send = Rc::new(());
        let args = [host(bias), host([3.0, 4.0])]
            .into_iter()
            .map(move |value| {
                let _ = Rc::strong_count(&non_send);
                value
            });
        let result = block_on(device.execute(expr.clone(), args, opts)).unwrap();
        let buffer = result[0].downcast::<DataHandle>().unwrap();
        assert_eq!(unpack(buffer), expected);
        let storage = buffer.downcast_ref::<JaxBuffer>().unwrap();
        assert_eq!(storage.device(), &device);
        assert_eq!(
            buffer.buffer_info().location,
            MemoryLocation::Device {
                runtime: "jax",
                device_id: device.uuid().to_string()
            }
        );
    }
    assert_eq!(
        block_on(device.execute(expr, [host([1., 2.]), host([3., 4.])], ExecOpts::default()))
            .unwrap_err()
            .kind(),
        io::ErrorKind::Unsupported
    );
    Python::attach(|py| {
        test_dlpack(py, &device);
    });
    test_eval(&device);
}

fn test_eval(device: &gnx_jax::JaxDevice) {
    fn add(left: &gnx::Array, right: &gnx::Array) -> gnx::Array {
        let info = left.tracer().trace_ref().info().clone();
        let result = gnx_expr::Invocation::invoke(
            add_op(),
            vec![],
            vec![left.tracer().clone(), right.tracer().clone()],
            vec![info],
        )
        .pop()
        .unwrap();
        result.try_cast::<gnx::Array>().ok().unwrap().into()
    }

    let a = gnx::Array::from_data(host([1., 2.]).downcast_into::<DataHandle>().unwrap());
    let b = gnx::Array::from_data(host([3., 4.]).downcast_into::<DataHandle>().unwrap());
    let z = add(&a, &b);
    let x = add(&z, &z);
    block_on(device.eval(
        &x,
        ExecOpts {
            transfers: TransferPolicy::AllowCopy,
        },
    ))
    .unwrap();
    assert_eq!(unpack(x.data().unwrap()), [8., 12.]);
    assert_eq!(unpack(z.data().unwrap()), [4., 6.]);

    // Retaining buffers by capture must allow native JAX inputs with no copy.
    let next = add(&x, &z);
    block_on(device.eval(&next, ExecOpts::default())).unwrap();
    assert_eq!(unpack(next.data().unwrap()), [12., 18.]);
    assert_eq!(unpack(a.data().unwrap()), [1., 2.]);

    // Eval materializes the home tracers without changing either cell.
    use gnx::array::ArrayRef;
    use gnx_expr::{Capture, CellTracer, TraceContext};
    let traced = {
        let scope = TraceContext::enter();
        let cell = ArrayRef::from(CellTracer::placeholder(ValueInfo::new(info())));
        let sum = add(&cell.get(), &z);
        cell.set(sum.tracer().clone());
        let fresh = ArrayRef::new(sum.into_tracer());
        Capture::from_context(
            scope.context(),
            &[cell.into_tracer().into()],
            &[fresh.into_tracer().into()],
        )
        .unwrap()
    };
    let cell = ArrayRef::new(x.tracer().clone());
    let fresh = traced.call(&[cell.tracer().clone().into()]).pop().unwrap();
    assert!(cell.get().data().is_none());
    block_on(device.eval(&cell, ExecOpts::default())).unwrap();
    assert_eq!(unpack(cell.get().data().unwrap()), [12., 18.]);
    let fresh = fresh.resolve();
    assert_eq!(
        unpack(
            fresh
                .try_concrete()
                .unwrap()
                .downcast::<DataHandle>()
                .unwrap()
        ),
        [12., 18.]
    );
}

#[derive(Debug)]
struct NumpyBuffer(Py<PyAny>);
impl fmt::Display for NumpyBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("NumPy CPU allocation")
    }
}
impl DataImpl for NumpyBuffer {
    fn info(&self) -> ArrayInfo {
        info()
    }
    fn buffer_info(&self) -> BufferInfo {
        BufferInfo {
            location: MemoryLocation::Host,
            memory_kind: None,
            layout: BufferLayout::Packed,
        }
    }
    fn dlpack_device(&self) -> Option<DlpackDevice> {
        Some(DlpackDevice(1, 0))
    }
    fn export_dlpack(&self, request: DlpackRequest) -> io::Result<DlpackTensor> {
        Python::attach(|py| {
            // SAFETY: owned native NumPy allocation with no writer after export.
            unsafe { dlpack::export(self.0.bind(py), request) }
        })
        .map_err(io::Error::other)
    }
}

fn test_dlpack(py: Python<'_>, device: &gnx_jax::JaxDevice) {
    let numpy = py.import("numpy").unwrap();
    let owner = numpy.call_method1("zeros", (256, "uint8")).unwrap();
    let addr = owner
        .getattr("ctypes")
        .unwrap()
        .getattr("data")
        .unwrap()
        .extract::<usize>()
        .unwrap();
    let offset = addr.next_multiple_of(64) - addr;
    let array = numpy
        .call_method1("frombuffer", (&owner, "float32", 2, offset))
        .unwrap();
    array.set_item(0, 4.0).unwrap();
    array.set_item(1, 6.0).unwrap();
    let addr = addr + offset;
    drop(owner);
    let native = device
        .import_buffer(
            py,
            DataHandle::new(NumpyBuffer(array.unbind())),
            TransferPolicy::RequireZeroCopy,
        )
        .unwrap();
    let actual_addr = native
        .native(py)
        .call_method0("unsafe_buffer_pointer")
        .unwrap()
        .extract::<usize>()
        .unwrap();
    assert_eq!(actual_addr, addr);
    assert_eq!(
        unpack(&DataHandle::new(
            JaxBuffer::from_native(native.native(py).clone()).unwrap()
        )),
        [4., 6.]
    );
    let producer =
        dlpack::producer(py, DataHandle::new(native).into_dlpack().ok().unwrap()).unwrap();
    let restored = numpy.call_method1("from_dlpack", (&producer,)).unwrap();
    let restored_addr = restored
        .getattr("ctypes")
        .unwrap()
        .getattr("data")
        .unwrap()
        .extract::<usize>()
        .unwrap();
    assert_eq!(restored_addr, addr);
    assert_eq!(
        restored
            .call_method0("tolist")
            .unwrap()
            .extract::<Vec<f32>>()
            .unwrap(),
        [4., 6.]
    );
    assert!(numpy.call_method1("from_dlpack", (producer,)).is_err());
}

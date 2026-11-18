use gnx_expr::{
    Capture, CellKey, HInstr, Invocation, ObjectTracer, TraceContext, Tracer, TracerKey, ValueInfo,
};
use gnx_math::{Array, ArrayInfo, ArrayRef, DType, Shape};
use gnx_transforms::{Callable, jit};
use std::cell::Cell;
use std::sync::Arc;

fn array() -> Array {
    Tracer::<Array>::placeholder(ValueInfo::new(ArrayInfo::new(
        Shape::fixed([4]),
        DType::F32,
    )))
    .into()
}
fn same(a: &Array, b: &Array) {
    assert_eq!(
        TracerKey::from(a.tracer().trace_ref()),
        TracerKey::from(b.tracer().trace_ref())
    );
}
fn same_cell(a: &ArrayRef, b: &ArrayRef) {
    assert_eq!(
        CellKey::from(a.tracer().cell_ref()),
        CellKey::from(b.tracer().cell_ref())
    );
}

#[test]
fn rebuilds_typed_results_static_fields_and_shared_graphs() {
    let value = array();
    let shared = Arc::new(value.clone());
    let calls = Cell::new(0);
    let function = jit(|a: Array, b: Array, shared: Arc<Array>, label: String| {
        calls.set(calls.get() + 1);
        same(&a, &b);
        assert!(a.tracer().trace_ref().context_id().is_current());
        (a, b, shared.clone(), shared, label, 42u32)
    });
    let result = function.invoke((value.clone(), value.clone(), shared, "static".to_string()));
    same(&result.0, &value);
    same(&result.1, &value);
    same(&result.2, &value);
    assert!(Arc::ptr_eq(&result.2, &result.3));
    assert_eq!(result.4, "static");
    assert_eq!(result.5, 42);
    assert_eq!(calls.get(), 1);
}

#[test]
fn replays_aliasing_reads_and_writes_on_root_cells() {
    let initial = array();
    let replacement = array();
    let target = ArrayRef::new(initial.into_tracer());
    let calls = Cell::new(0);
    let function = jit(|a: ArrayRef, b: ArrayRef, value: Array| {
        calls.set(calls.get() + 1);
        same_cell(&a, &b);
        a.set(value.into_tracer());
        (b.get(), a, b)
    });
    let result = function.invoke((target.clone(), target.clone(), replacement.clone()));
    same(&result.0, &replacement);
    same(&target.get(), &replacement);
    same_cell(&result.1, &target);
    same_cell(&result.2, &target);
    assert_eq!(calls.get(), 1);
}

#[test]
fn captured_cells_are_only_mutated_by_replay_and_can_alias_arguments() {
    let initial = array();
    let replacement = array();
    let target = ArrayRef::new(initial.clone().into_tracer());
    let function = jit(|argument: ArrayRef, value: Array| {
        target.set(value.into_tracer());
        // The temporary trace must not mutate the caller's cell.
        assert_eq!(
            TracerKey::from(&target.tracer().cell_ref().read_home()),
            TracerKey::from(initial.tracer().trace_ref())
        );
        (argument.get(), target.clone())
    });
    let result = function.invoke((target.clone(), replacement.clone()));
    same(&result.0, &replacement);
    same(&target.get(), &replacement);
    same_cell(&result.1, &target);
}

#[test]
fn fresh_returned_cells_keep_aliases_but_are_fresh_per_call() {
    let value = array();
    let function = jit(|value: Array| {
        let local = ArrayRef::new(value.into_tracer());
        (local.clone(), local)
    });
    let first = function.invoke((value.clone(),));
    let second = function.invoke((value.clone(),));
    same_cell(&first.0, &first.1);
    same_cell(&second.0, &second.1);
    assert_ne!(
        CellKey::from(first.0.tracer().cell_ref()),
        CellKey::from(second.0.tracer().cell_ref())
    );
    same(&first.0.get(), &value);
}

#[test]
fn nested_jit_replays_into_the_enclosing_capture() {
    let outer = TraceContext::enter();
    let target = ArrayRef::from(gnx_expr::CellTracer::<Array>::placeholder(ValueInfo::new(
        ArrayInfo::new(Shape::fixed([4]), DType::F32),
    )));
    let replacement = array();
    let inner = jit(|cell: ArrayRef, value: Array| {
        cell.set(value.into_tracer());
        cell.get()
    });
    let function = jit(|cell: ArrayRef, value: Array| inner.invoke((cell, value)));
    let result = function.invoke((target.clone(), replacement.clone()));
    let inputs = [
        ObjectTracer::Cell(target.into_tracer().into()),
        ObjectTracer::Ref(replacement.into_tracer().into_trace_ref()),
    ];
    let capture = Capture::from_context(
        outer.context(),
        &inputs,
        &[ObjectTracer::Ref(result.into_tracer().into_trace_ref())],
    )
    .unwrap();
    assert_eq!(capture.expr().instructions().len(), 1);
    assert!(matches!(
        capture.expr().instructions()[0],
        HInstr::WriteCell { .. }
    ));
    // The read after the write forwards the assigned value at the boundary;
    // it does not require reading the original cell contents.
}

struct TestDialect;
struct TestOperation;
impl gnx_expr::Dialect for TestDialect {
    fn name(&self) -> &'static str {
        "jit_test"
    }
    fn create(&self, name: &'static str) -> Option<&'static dyn gnx_expr::Operation> {
        (name == "identity").then_some(&TestOperation as &dyn gnx_expr::Operation)
    }
}
impl gnx_expr::Operation for TestOperation {
    fn name(&self) -> &'static str {
        "identity"
    }
    fn dialect(&self) -> &'static dyn gnx_expr::Dialect {
        &TestDialect
    }
}

#[test]
fn operation_results_are_replayed_with_actual_inputs_not_placeholders() {
    let value = array();
    let captured_value = array();
    let calls = Cell::new(0);
    let function = jit(|input: Array| {
        calls.set(calls.get() + 1);
        let result = Invocation::invoke(
            gnx_expr::Op::new(&TestOperation),
            vec![],
            vec![
                input.into_tracer().into_trace_ref(),
                captured_value.tracer().trace_ref().clone(),
            ],
            vec![captured_value.tracer().trace_ref().info().clone()],
        )
        .pop()
        .unwrap();
        let tracer: Tracer<gnx_expr::Generic> = result.into();
        Array::from(tracer.try_cast::<Array>().ok().unwrap())
    });
    for _ in 0..2 {
        let result = function.invoke((value.clone(),));
        let trace = result.tracer().trace_ref();
        assert!(trace.context_id().is_current());
        let (invocation, _) = trace.producer_with_index().unwrap();
        let Invocation::Apply { op, .. } = invocation.as_ref() else {
            panic!("expected operation")
        };
        assert_eq!(op.to_string(), "builtins.call");
        assert_eq!(invocation.subexprs().len(), 1);
        let body = &invocation.subexprs()[0].0;
        body.check().unwrap();
        assert_eq!(body.eqns().len(), 1);
        assert_eq!(body.eqns()[0].op().to_string(), "jit_test.identity");
        assert_eq!(body.closure().len(), 1);
        assert_eq!(body.inputs().len(), 1);
        assert_eq!(invocation.inputs().len(), 1);
        assert_eq!(invocation.subexprs()[0].1.len(), 1);
        assert_eq!(
            TracerKey::from(&invocation.subexprs()[0].1[0]),
            TracerKey::from(captured_value.tracer().trace_ref())
        );
        assert_eq!(
            TracerKey::from(&invocation.inputs()[0]),
            TracerKey::from(value.tracer().trace_ref())
        );
    }
    assert_eq!(calls.get(), 2, "this stage deliberately does not cache");
}

#[test]
fn nested_computation_calls_keep_both_function_boundaries() {
    let input = array();
    let inner = jit(|value: Array| {
        let info = value.tracer().trace_ref().info().clone();
        let result = Invocation::invoke(
            gnx_expr::Op::new(&TestOperation),
            vec![],
            vec![value.into_tracer()],
            vec![info],
        )
        .pop()
        .unwrap();
        let tracer: Tracer<gnx_expr::Generic> = result.into();
        Array::from(tracer.try_cast::<Array>().ok().unwrap())
    });
    let outer = jit(|value: Array| inner.invoke((value,)));
    let result = outer.invoke((input.clone(),));
    let (outer_call, _) = result.tracer().trace_ref().producer_with_index().unwrap();
    let outer_body = &outer_call.subexprs()[0].0;
    assert_eq!(outer_body.eqns().len(), 1);
    assert_eq!(outer_body.eqns()[0].op().to_string(), "builtins.call");
    let inner_body = outer_body.eqns()[0].subexprs()[0].expr();
    assert_eq!(inner_body.eqns().len(), 1);
    assert_eq!(inner_body.eqns()[0].op().to_string(), "jit_test.identity");
    outer_body.check().unwrap();
    assert_eq!(
        TracerKey::from(&outer_call.inputs()[0]),
        TracerKey::from(input.tracer().trace_ref())
    );
}

#[test]
fn update_only_computations_retain_the_call_and_update_the_original_cell() {
    let input = array();
    let target = ArrayRef::new(array().into_tracer());
    let function = jit(|cell: ArrayRef, value: Array| {
        let info = value.tracer().trace_ref().info().clone();
        let result = Invocation::invoke(
            gnx_expr::Op::new(&TestOperation),
            vec![],
            vec![value.into_tracer()],
            vec![info],
        )
        .pop()
        .unwrap();
        let tracer: Tracer<gnx_expr::Generic> = result.into();
        cell.set(tracer.try_cast::<Array>().ok().unwrap());
    });
    function.invoke((target.clone(), input.clone()));
    let home = target.tracer().cell_ref().read_home();
    let (call, index) = home.producer_with_index().unwrap();
    assert_eq!(index, 0);
    let Invocation::Apply { op, .. } = call.as_ref() else {
        panic!("expected call")
    };
    assert_eq!(op.to_string(), "builtins.call");
    assert_eq!(call.subexprs()[0].0.eqns().len(), 1);
    assert_eq!(
        TracerKey::from(&call.inputs()[1]),
        TracerKey::from(input.tracer().trace_ref())
    );
}

#[test]
fn borrowed_array_arguments_use_owned_placeholder_graphs() {
    let value = array();
    let function = gnx_transforms::jittable!(fn(&Array) -> Array, |x: &Array| {
        assert!(x.tracer().trace_ref().context_id().is_current());
        x.clone()
    });
    let result = jit(function).invoke((&value,));
    same(&result, &value);
}

#[test]
fn writes_are_replayed_even_with_no_dynamic_outputs() {
    let value = array();
    let target = ArrayRef::new(array().into_tracer());
    let function = jit(|cell: ArrayRef, value: Array| cell.set(value.into_tracer()));
    function.invoke((target.clone(), value.clone()));
    same(&target.get(), &value);
}

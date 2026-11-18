use gnx_expr::{
    Capture, CellKey, HInstr, Invocation, ObjectTracer, TraceContext, Tracer, TracerKey, ValueInfo,
};
use gnx_math::{Array, ArrayInfo, ArrayRef, DType, Shape};
use gnx_transforms::{Callable, jit};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};

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
        assert_ne!(
            TracerKey::from(a.tracer().trace_ref()),
            TracerKey::from(b.tracer().trace_ref())
        );
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
fn flat_results_keep_order_after_a_shared_child_on_cache_hits() {
    let calls = Cell::new(0);
    let function = jit(|a: Array, b: Array, tail: Array| {
        calls.set(calls.get() + 1);
        let shared = Arc::new(b);
        ((a, shared.clone()), ("label".to_owned(), shared, tail))
    });
    for _ in 0..2 {
        let (a, b, tail) = (array(), array(), array());
        let ((first, shared), (label, repeated, last)) =
            function.invoke((a.clone(), b.clone(), tail.clone()));
        same(&first, &a);
        same(shared.as_ref(), &b);
        assert!(Arc::ptr_eq(&shared, &repeated));
        assert_eq!(label, "label");
        same(&last, &tail);
    }
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
        assert_ne!(
            CellKey::from(a.tracer().cell_ref()),
            CellKey::from(b.tracer().cell_ref())
        );
        a.set(value.into_tracer());
        (b.get(), a, b)
    });
    let result = function.invoke((target.clone(), target.clone(), replacement.clone()));
    assert_same_call(&result.0, &target.get());
    same(&argument(call(&result.0).1, 2), &replacement);
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
    assert_same_call(&result.0, &target.get());
    same(&argument(call(&result.0).1, 1), &replacement);
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
    same(&argument(call(&first.0.get()).1, 0), &value);
}

#[test]
fn nested_jit_records_a_high_level_call_in_the_enclosing_capture() {
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
    let eqn = capture.expr().instructions()[0].as_apply().unwrap();
    assert_eq!(eqn.op(), &gnx_expr::builtins::call_op());
    let nested = eqn.subexprs()[0].expr().instructions()[0]
        .as_apply()
        .unwrap();
    assert!(matches!(
        nested.subexprs()[0].expr().instructions()[0],
        HInstr::WriteCell { .. }
    ));
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

fn compute(value: Array) -> Array {
    let info = value.tracer().trace_ref().info().clone();
    let result = Invocation::invoke(
        gnx_expr::Op::new(&TestOperation),
        vec![],
        vec![value.into_tracer()],
        vec![info],
    )
    .pop()
    .unwrap();
    Array::from(result.try_cast::<Array>().ok().unwrap())
}

fn call(value: &Array) -> (&Arc<Invocation>, &gnx_expr::DeferredCall) {
    let (invocation, _) = value.tracer().trace_ref().producer_with_index().unwrap();
    let Invocation::DeferredCall(call) = invocation.as_ref() else {
        panic!("expected a high-level call")
    };
    (invocation, call)
}

fn argument(call: &gnx_expr::DeferredCall, index: usize) -> Array {
    let ObjectTracer::Ref(value) = &call.inputs()[index] else {
        panic!("expected value argument")
    };
    Array::from(
        Tracer::from(value.clone())
            .try_cast::<Array>()
            .ok()
            .unwrap(),
    )
}

fn cell_key(object: &ObjectTracer) -> CellKey {
    let ObjectTracer::Cell(cell) = object else {
        panic!("expected cell binding")
    };
    CellKey::from(cell)
}

fn assert_same_call(a: &Array, b: &Array) {
    assert!(Arc::ptr_eq(call(a).0, call(b).0));
}

#[test]
fn threads_share_a_cold_trace_across_alias_patterns() {
    let traces = AtomicUsize::new(0);
    let start = Barrier::new(6);
    let f = jit(|a: ArrayRef, b: ArrayRef, x: Array| {
        traces.fetch_add(1, Ordering::Relaxed);
        a.set(compute(x).into_tracer());
        b.get()
    });
    let bodies = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..6)
            .map(|i| {
                let f = &f;
                let start = &start;
                scope.spawn(move || {
                    let a = ArrayRef::new(array().into_tracer());
                    let old_b = array();
                    let b = if i % 2 == 0 {
                        a.clone()
                    } else {
                        ArrayRef::new(old_b.clone().into_tracer())
                    };
                    start.wait();
                    let result = f.invoke((a.clone(), b.clone(), array()));
                    let invocation = call(&result).1;
                    assert_eq!(
                        cell_key(&invocation.inputs()[0]) == cell_key(&invocation.inputs()[1]),
                        i % 2 == 0
                    );
                    assert_same_call(&a.get(), &result);
                    if i % 2 != 0 {
                        same(&b.get(), &old_b);
                    }
                    invocation.body().clone()
                })
            })
            .collect();
        workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .collect::<Vec<_>>()
    });
    assert_eq!(traces.load(Ordering::Relaxed), 1);
    assert!(bodies.iter().all(|body| Arc::ptr_eq(body, &bodies[0])));
}

#[test]
fn high_level_call_preserves_aliases_between_captures_and_inputs() {
    let captured = ArrayRef::new(array().into_tracer());
    let calls = Cell::new(0);
    let f = jit(|input: ArrayRef, x: Array| {
        calls.set(calls.get() + 1);
        captured.set(compute(x).into_tracer());
        input.get()
    });
    let first = f.invoke((ArrayRef::new(array().into_tracer()), array()));
    let first_call = call(&first).1;
    assert_ne!(
        cell_key(&first_call.captures()[0]),
        cell_key(&first_call.inputs()[0])
    );
    let second = f.invoke((captured.clone(), array()));
    let second_call = call(&second).1;
    assert_eq!(
        cell_key(&second_call.captures()[0]),
        cell_key(&second_call.inputs()[0])
    );
    assert!(Arc::ptr_eq(first_call.body(), second_call.body()));
    assert_eq!(calls.get(), 1);
}

#[test]
fn nested_jit_threads_local_cell_state_and_retains_update_only_calls() {
    let update = jit(|cell: ArrayRef, value: Array| cell.set(compute(value).into_tracer()));
    let outer = jit(|external: ArrayRef, value: Array| {
        let local = ArrayRef::new(value.clone().into_tracer());
        update.invoke((local.clone(), value));
        let result = local.get();
        update.invoke((external, result.clone()));
        (local.clone(), local, result)
    });
    let external = ArrayRef::new(array().into_tracer());
    let (first, second, local_value) = outer.invoke((external.clone(), array()));
    same_cell(&first, &second);
    assert_same_call(&first.get(), &local_value);
    assert_same_call(&external.get(), &local_value);
    let body = call(&local_value).1.body();
    assert_eq!(
        body.instructions()
            .iter()
            .filter(|instr| instr.as_apply().is_some())
            .count(),
        2
    );
}

#[test]
fn trace_cache_distinguishes_static_fields_and_ordered_leaf_metadata() {
    fn sized(n: usize) -> Array {
        Tracer::<Array>::placeholder(ValueInfo::new(ArrayInfo::new(
            Shape::fixed([n]),
            DType::F32,
        )))
        .into()
    }
    let calls = Cell::new(0);
    let f = jit(|a: Array, b: Array, label: u32| {
        calls.set(calls.get() + 1);
        (a, b, label)
    });
    let invoke = |a, b, label| {
        let (out_a, out_b, out_label) = f.invoke((sized(a), sized(b), label));
        assert_eq!(out_a.shape(), &Shape::fixed([a]));
        assert_eq!(out_b.shape(), &Shape::fixed([b]));
        assert_eq!(out_label, label);
    };
    invoke(2, 3, 1);
    invoke(2, 3, 1);
    invoke(3, 2, 1);
    invoke(2, 3, 2);
    invoke(2, 3, 1);
    assert_eq!(calls.get(), 3);
}

#[test]
fn trace_cache_preserves_graph_sharing_without_using_allocation_addresses() {
    let calls = Cell::new(0);
    let f = jit(|a: Arc<Array>, b: Arc<Array>| {
        calls.set(calls.get() + 1);
        Arc::ptr_eq(&a, &b)
    });
    for _ in 0..2 {
        let new_shared = Arc::new(array());
        assert!(f.invoke((new_shared.clone(), new_shared)));
        assert!(!f.invoke((Arc::new(array()), Arc::new(array()))));
    }
    assert_eq!(calls.get(), 2);
}

#[test]
fn trace_cache_checks_builder_equality_after_a_hash_match() {
    // The derived hash_builder for an opaque builder only hashes its type.
    #[derive(
        Clone, gnx_graph::Leaf, gnx_graph::GraphPartialEq, gnx_graph::GraphEq, gnx_graph::GraphHash,
    )]
    struct Static(u32);

    let calls = Cell::new(0);
    let f = jit(|value: Static, input: Array| {
        calls.set(calls.get() + 1);
        (value.0, input)
    });
    for label in [1, 2, 1] {
        let input = array();
        let (output_label, output) = f.invoke((Static(label), input.clone()));
        assert_eq!(output_label, label);
        same(&output, &input);
    }
    assert_eq!(calls.get(), 2);
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
        let invocation = call(&result).1;
        let body = invocation.body();
        assert_eq!(
            body.instructions()[0].as_apply().unwrap().op().to_string(),
            "jit_test.identity"
        );
        let [ObjectTracer::Ref(captured)] = invocation.captures() else {
            panic!("expected captured value")
        };
        assert_eq!(
            TracerKey::from(captured),
            TracerKey::from(captured_value.tracer().trace_ref())
        );
        same(&argument(invocation, 0), &value);
    }
    assert_eq!(calls.get(), 1, "the second call reuses the trace");
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
    let outer_call = call(&result).1;
    let outer_body = outer_call.body();
    assert_eq!(outer_body.instructions().len(), 1);
    assert_eq!(
        outer_body.instructions()[0]
            .as_apply()
            .unwrap()
            .op()
            .to_string(),
        "builtins.call"
    );
    let inner_body = outer_body.instructions()[0].as_apply().unwrap().subexprs()[0].expr();
    assert_eq!(inner_body.instructions().len(), 1);
    assert_eq!(
        inner_body.instructions()[0]
            .as_apply()
            .unwrap()
            .op()
            .to_string(),
        "jit_test.identity"
    );
    same(&argument(outer_call, 0), &input);
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
    let home = target.get();
    let update = call(&home).1;
    assert!(
        update
            .body()
            .instructions()
            .iter()
            .any(|instr| matches!(instr, HInstr::WriteCell { .. }))
    );
    same(&argument(update, 1), &input);
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
    same(&argument(call(&target.get()).1, 1), &value);
}

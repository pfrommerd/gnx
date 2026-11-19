use gnx_expr::{
    Capture, CellTrace, Dialect, Generic, InputId, InputSpec, Invocation, ObjectTracer, Op,
    Operation, Trace, TraceContext, Tracer, TracerKey, ValueInfo,
};
use gnx_transforms::cf::call_lowered;

fn info() -> ValueInfo {
    ValueInfo::new(0usize)
}
fn value() -> Trace {
    Trace::placeholder(info())
}
fn same(a: &Trace, b: &Trace) {
    assert_eq!(TracerKey::from(a), TracerKey::from(b));
}
struct TestDialect;
struct Compute;
impl Dialect for TestDialect {
    fn name(&self) -> &'static str {
        "parameters"
    }
    fn create(&self, _: &'static str) -> Option<&'static dyn Operation> {
        Some(&Compute)
    }
}
impl Operation for Compute {
    fn name(&self) -> &'static str {
        "compute"
    }
    fn dialect(&self) -> &'static dyn Dialect {
        &TestDialect
    }
}

#[test]
fn input_ids_are_original_offsets_and_explicit_inputs_take_precedence() {
    let a = value();
    let b = value();
    let c = value();
    let d = value();
    let closure = [
        b.clone().into(),
        c.clone().into(),
        c.into(),
        a.clone().into(),
        d.into(),
    ];
    let inputs = [a.clone().into(), a.into(), b.into()];
    let spec = InputSpec::from_args(&closure, &inputs);
    assert_eq!(
        spec.closure.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
        vec![
            InputId::Input(2),
            InputId::Closure(1),
            InputId::Closure(1),
            InputId::Input(0),
            InputId::Closure(4),
        ]
    );
    assert_eq!(
        spec.inputs.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
        vec![InputId::Input(0), InputId::Input(0), InputId::Input(2),]
    );
}

#[test]
fn closure_only_values_stay_captured_and_shared_values_are_regular_inputs() {
    let captured_only = value();
    let shared = value();
    let explicit_only = value();
    let captured = {
        let guard = TraceContext::enter();
        let a = value();
        let b = value();
        let c = value();
        let result: Tracer<Generic> = Invocation::invoke(
            Op::new(&Compute),
            vec![],
            vec![
                captured_only.clone(),
                shared.clone(),
                a.clone(),
                b.clone(),
                c.clone(),
            ],
            vec![info()],
        )
        .pop()
        .unwrap();
        Capture::from_context(
            guard.context(),
            &[a.into(), b.into(), c.into()],
            &[result.into()],
        )
        .unwrap()
    };
    let arguments = [
        shared.clone().into(),
        explicit_only.clone().into(),
        shared.clone().into(),
    ];
    let spec = InputSpec::from_args(captured.closure(), &arguments);
    let lowered = captured.expr().lower(&spec).unwrap();
    assert_eq!(lowered.closure, vec![InputId::Closure(0)]);
    assert_eq!(lowered.inputs, vec![InputId::Input(0), InputId::Input(1)]);
    assert_eq!(lowered.expr.closure().len(), 1);
    assert_eq!(lowered.expr.inputs().len(), 2);
    let operands = lowered.expr.eqns()[0].inputs();
    assert_eq!(operands[0], lowered.expr.closure()[0].0);
    assert_eq!(operands[1], lowered.expr.inputs()[0].0);
    assert_eq!(operands[2], operands[1]);
    assert_eq!(operands[3], lowered.expr.inputs()[1].0);
    assert_eq!(operands[4], operands[1]);
    let results = call_lowered(&lowered, captured.closure(), &arguments).unwrap();
    let ObjectTracer::Ref(result) = &results[0] else {
        panic!("expected value")
    };
    let (call, _) = result.producer_with_index().unwrap();
    assert_eq!(call.inputs().len(), 2);
    assert_eq!(call.subexprs()[0].1.len(), 1);
    assert!(std::sync::Arc::ptr_eq(
        &call.subexprs()[0].0,
        lowered.trace_body()
    ));
    let cloned = lowered.clone();
    assert!(std::sync::Arc::ptr_eq(&cloned.expr, &lowered.expr));
    let repeated = call_lowered(&cloned, captured.closure(), &arguments).unwrap();
    let ObjectTracer::Ref(repeated) = &repeated[0] else {
        panic!("expected value")
    };
    let (repeated_call, _) = repeated.producer_with_index().unwrap();
    assert!(std::sync::Arc::ptr_eq(
        &repeated_call.subexprs()[0].0,
        lowered.trace_body()
    ));
    same(&call.inputs()[0], &shared);
    same(&call.inputs()[1], &explicit_only);
    same(&call.subexprs()[0].1[0], &captured_only);
}

#[test]
fn captured_cells_share_updates_with_inputs_while_closure_only_cells_stay_captured() {
    let initial = value();
    let closure_only = CellTrace::new(initial.clone());
    let shared = CellTrace::new(value());
    let replacement = value();
    let captured = {
        let guard = TraceContext::enter();
        let a = CellTrace::placeholder(info());
        let x = value();
        let old = closure_only.get();
        shared.set(x.clone());
        let after = a.get();
        Capture::from_context(
            guard.context(),
            &[a.into(), x.into()],
            &[old.into(), after.into()],
        )
        .unwrap()
    };
    let arguments = [shared.clone().into(), replacement.clone().into()];
    let spec = InputSpec::from_args(captured.closure(), &arguments);
    let lowered = captured.expr().lower(&spec).unwrap();
    assert_eq!(lowered.closure.len(), 1);
    assert!(matches!(lowered.closure[0], InputId::Closure(_)));
    assert_eq!(lowered.inputs, vec![InputId::Input(0), InputId::Input(1)]);
    assert_eq!(lowered.updates.len(), 1);
    assert_eq!(lowered.updates[0].0, InputId::Input(0));
    let results = call_lowered(&lowered, captured.closure(), &arguments).unwrap();
    let ObjectTracer::Ref(old) = &results[0] else {
        panic!("expected value")
    };
    let ObjectTracer::Ref(after) = &results[1] else {
        panic!("expected value")
    };
    same(old, &initial);
    same(after, &replacement);
    same(&shared.get(), &replacement);
}

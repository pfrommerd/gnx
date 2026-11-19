use std::fmt;

use gnx_expr::{
    Capture, CaptureError, CellTrace, ConcreteValue, Dialect, EvalPlanError, Invocation, Op,
    Operation, Trace, TraceContext, TracerKey, Value, ValueInfo,
};

struct TestOp;
static TEST: TestOp = TestOp;

impl Dialect for TestOp {
    fn name(&self) -> &'static str {
        "test"
    }
    fn create(&self, _: &'static str) -> Option<&'static dyn Operation> {
        Some(&TEST)
    }
}
impl Operation for TestOp {
    fn name(&self) -> &'static str {
        "compute"
    }
    fn dialect(&self) -> &'static dyn Dialect {
        &TEST
    }
}

#[derive(Debug)]
struct Number(i64);
impl fmt::Display for Number {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}
impl ConcreteValue for Number {
    fn to_info(&self) -> ValueInfo {
        info()
    }
}

#[derive(Debug)]
struct WrongType;
impl fmt::Display for WrongType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("wrong type")
    }
}
impl ConcreteValue for WrongType {
    fn to_info(&self) -> ValueInfo {
        ValueInfo::new("wrong")
    }
}

fn info() -> ValueInfo {
    ValueInfo::new("number")
}
fn value(n: i64) -> Value {
    Value::new(Number(n))
}
fn concrete(n: i64) -> Trace {
    Trace::concrete(value(n), info())
}
fn number(trace: &Trace) -> i64 {
    trace
        .try_concrete()
        .unwrap()
        .downcast::<Number>()
        .unwrap()
        .0
}

fn op(inputs: Vec<Trace>, outputs: usize) -> Vec<Trace> {
    Invocation::invoke(Op::new(&TEST), vec![], inputs, vec![info(); outputs])
        .into_iter()
        .map(|value| value.into_trace_ref())
        .collect()
}

#[test]
fn live_requested_and_sibling_results_are_published_in_place() {
    let a = concrete(2);
    let [z, sibling]: [Trace; 2] = op(vec![a.clone()], 2).try_into().unwrap();
    let x = op(vec![z.clone(), z.clone()], 1).pop().unwrap();
    let alias = z.clone();
    let producer = std::sync::Arc::downgrade(&x.producer().unwrap());
    let capture = Capture::from_roots([&x, &x]).unwrap();
    assert_eq!(capture.expr().instructions().len(), 2);
    let selected = capture.outputs();
    assert_eq!(
        selected
            .iter()
            .map(|out| TracerKey::from(&out.resolve()))
            .collect::<Vec<_>>(),
        [&x, &z, &sibling].map(TracerKey::from)
    );

    let (expr, args, commit) = capture.prepare().unwrap().into_parts();
    assert_eq!(expr.outputs().len(), 3);
    assert!(std::ptr::eq(
        args[0].downcast::<Number>().unwrap(),
        a.try_concrete().unwrap().downcast::<Number>().unwrap()
    ));
    let old_key = TracerKey::from(&z);
    commit.publish([value(12), value(6), value(5)]).unwrap();
    assert_eq!(
        [number(&x), number(&z), number(&alias), number(&sibling)],
        [12, 6, 6, 5]
    );
    assert_eq!(TracerKey::from(&z), old_key);
    assert!(z.producer().is_none());
    assert!(x.producer().is_none());
    assert!(producer.upgrade().is_none());
    assert!(Capture::from_roots([&x, &z]).unwrap().outputs().is_empty());
}

#[test]
fn internal_edges_do_not_make_an_intermediate_an_output() {
    let z = op(vec![concrete(2)], 1).pop().unwrap();
    let x = op(vec![z.clone(), z.clone()], 1).pop().unwrap();
    // Even a dropped consumer should not pin this intermediate as an output.
    drop(op(vec![z], 1));
    let capture = Capture::from_roots([&x]).unwrap();
    assert_eq!(capture.outputs().len(), 1);
    assert_eq!(capture.expr().instructions().len(), 2);
    let (expr, _, _) = capture.prepare().unwrap().into_parts();
    assert_eq!(expr.outputs().len(), 1);
}

#[test]
fn another_deferred_consumer_keeps_its_input_live_but_is_not_evaluated() {
    let z = op(vec![concrete(2)], 1).pop().unwrap();
    let x = op(vec![z.clone()], 1).pop().unwrap();
    let y = op(vec![z], 1).pop().unwrap();
    let capture = Capture::from_roots([&x]).unwrap();
    assert_eq!(capture.outputs().len(), 2);
    assert_eq!(capture.expr().instructions().len(), 2);
    let (_, _, commit) = capture.prepare().unwrap().into_parts();
    commit.publish([value(16), value(8)]).unwrap();
    assert!(y.try_concrete().is_none());
    let later = Capture::from_roots([&y]).unwrap();
    assert_eq!(later.expr().instructions().len(), 1);
    let (_, args, commit) = later.prepare().unwrap().into_parts();
    assert_eq!(args[0].downcast::<Number>().unwrap().0, 8);
    commit.publish([value(9)]).unwrap();
    assert_eq!(number(&y), 9);
}

#[test]
fn context_capture_retains_bindings_in_expression_order() {
    let explicit = concrete(4);
    let closed = concrete(6);
    let (capture, output) = {
        let scope = TraceContext::enter();
        let output = op(vec![explicit.clone(), closed.clone()], 1).pop().unwrap();
        let capture = Capture::from_context(
            scope.context(),
            &[explicit.clone().into()],
            &[output.clone().into(), explicit.clone().into()],
        )
        .unwrap();
        (capture, output)
    };
    assert_eq!(
        TracerKey::from(&capture.closure()[0].resolve()),
        TracerKey::from(&closed)
    );
    assert_eq!(
        TracerKey::from(&capture.inputs()[0].resolve()),
        TracerKey::from(&explicit)
    );
    assert_eq!(
        TracerKey::from(&capture.outputs()[0].resolve()),
        TracerKey::from(&output)
    );
    assert_eq!(
        TracerKey::from(&capture.outputs()[1].resolve()),
        TracerKey::from(&explicit)
    );
    let (expr, args, commit) = capture.prepare().unwrap().into_parts();
    assert_eq!(expr.outputs().len(), 1);
    assert_eq!(
        args.iter()
            .map(|arg| arg.downcast::<Number>().unwrap().0)
            .collect::<Vec<_>>(),
        [6, 4]
    );
    commit.publish([value(10)]).unwrap();
    assert_eq!(number(&output), 10);
    assert_eq!(number(&explicit), 4);
}

#[test]
fn pure_traced_calls_keep_only_outside_results_live() {
    let traced = {
        let guard = TraceContext::enter();
        let input = Trace::placeholder(info());
        let output = op(vec![input.clone()], 1).pop().unwrap();
        gnx_expr::Capture::from_context(guard.context(), &[input.into()], &[output.into()]).unwrap()
    };
    let z = op(vec![concrete(1)], 1).pop().unwrap();
    let gnx_expr::ObjectTracer::Ref(x) = gnx_transforms::cf::call(&traced, &[z.into()])
        .pop()
        .unwrap()
    else {
        panic!("expected one value result");
    };
    let capture = Capture::from_roots([&x]).unwrap();
    assert_eq!(capture.outputs().len(), 1);
    assert_eq!(capture.expr().instructions().len(), 2);
    let (expr, _, commit) = capture.prepare().unwrap().into_parts();
    assert_eq!(expr.eqns().len(), 2);
    commit.publish([value(2)]).unwrap();
    assert_eq!(number(&x), 2);
}

#[test]
fn invalid_output_does_not_publish_and_first_success_wins() {
    let [x, y]: [Trace; 2] = op(vec![concrete(2)], 2).try_into().unwrap();
    let prepare = || {
        Capture::from_roots([&x, &y])
            .unwrap()
            .prepare()
            .unwrap()
            .into_parts()
            .2
    };
    assert!(matches!(
        prepare().publish([value(1)]),
        Err(EvalPlanError::OutputArity { .. })
    ));
    assert!(matches!(
        prepare().publish([value(1), Value::new(WrongType)]),
        Err(EvalPlanError::OutputInfo { index: 1, .. })
    ));
    assert!(x.try_concrete().is_none());
    assert!(y.try_concrete().is_none());
    let first = prepare();
    let second = prepare();
    first.publish([value(1), value(2)]).unwrap();
    second.publish([value(11), value(22)]).unwrap();
    assert_eq!([number(&x), number(&y)], [1, 2]);
}

#[test]
fn capture_can_inspect_placeholders_but_prepare_requires_bindings() {
    let _ctx = TraceContext::enter();
    let missing = Trace::placeholder(info());
    let pending = op(vec![missing], 1).pop().unwrap();
    let captured = Capture::from_roots([&pending]).unwrap();
    assert_eq!(captured.expr().instructions().len(), 1);
    assert!(captured.inputs().is_empty());
    assert!(captured.closure()[0].resolve().is_placeholder());
    assert!(matches!(
        captured.prepare(),
        Err(EvalPlanError::UnboundPlaceholder)
    ));
}

#[test]
fn captures_one_scope_at_a_time_and_stops_at_concrete_inputs() {
    let input = concrete(2);
    let pending = op(vec![input.clone()], 1).pop().unwrap();
    let (first, second) = {
        let _ctx = TraceContext::enter();
        let first = op(vec![input], 1).pop().unwrap();
        let second = op(vec![first.clone()], 1).pop().unwrap();
        let foreign = op(vec![pending.clone()], 1).pop().unwrap();
        assert!(matches!(
            Capture::from_roots([&pending, &first]),
            Err(CaptureError::MixedScopes)
        ));
        assert!(matches!(
            Capture::from_roots([&foreign]),
            Err(CaptureError::MixedScopes)
        ));
        assert_eq!(Capture::from_roots([&second]).unwrap().outputs().len(), 2);
        (first, second)
    };
    let (expr, _, commit) = Capture::from_roots([&second])
        .unwrap()
        .prepare()
        .unwrap()
        .into_parts();
    assert_eq!(expr.eqns().len(), 2);
    commit.publish([value(10), value(5)]).unwrap();
    assert_eq!([number(&first), number(&second)], [5, 10]);
}

#[test]
fn nested_cell_updates_and_cell_results_lower_to_value_results() {
    let traced = {
        let scope = TraceContext::enter();
        let input = CellTrace::placeholder(info());
        let result = op(vec![input.get()], 1).pop().unwrap();
        input.set(result.clone());
        let fresh = CellTrace::new(result);
        Capture::from_context(scope.context(), &[input.into()], &[fresh.into()]).unwrap()
    };
    let original = CellTrace::new(op(vec![concrete(2)], 1).pop().unwrap());
    let result = gnx_transforms::cf::call(&traced, &[original.clone().into()])
        .pop()
        .unwrap();
    let fresh_home = result.resolve();
    let original_home = original.read_home();
    let capture = Capture::from_roots([&fresh_home]).unwrap();
    assert_eq!(capture.outputs().len(), 2);
    let (expr, args, commit) = capture.prepare().unwrap().into_parts();
    assert_eq!(expr.eqns().len(), 2);
    assert_eq!(expr.outputs().len(), 1);
    assert_eq!(args[0].downcast::<Number>().unwrap().0, 2);
    commit.publish([value(3)]).unwrap();
    assert_eq!(number(&original_home), 3);
    assert_eq!(number(&fresh_home), 3);
    assert_eq!(
        TracerKey::from(&original.read_home()),
        TracerKey::from(&original_home)
    );
    assert_eq!(
        TracerKey::from(&result.resolve()),
        TracerKey::from(&fresh_home)
    );
}

#[test]
fn read_only_outer_cells_use_concrete_contents() {
    let cell = CellTrace::new(concrete(2));
    let _scope = TraceContext::enter();
    let read = cell.get();
    let capture = Capture::from_roots([&read]).unwrap();
    let (expr, args, commit) = capture.prepare().unwrap().into_parts();
    assert!(expr.eqns().is_empty());
    assert!(expr.outputs().is_empty());
    assert_eq!(args[0].downcast::<Number>().unwrap().0, 2);
    commit.publish([]).unwrap();
    assert_eq!(number(&read), 2);
}

#[test]
fn outer_cell_writes_and_cell_outputs_cannot_be_prepared() {
    let outer = CellTrace::new(concrete(2));
    let scope = TraceContext::enter();
    outer.set(concrete(3));
    let read = outer.get();
    let capture = Capture::from_roots([&read]).unwrap();
    assert!(matches!(capture.prepare(), Err(EvalPlanError::CellEffect)));
    assert!(read.try_concrete().is_none());
    assert_eq!(number(&outer.read_home()), 2);

    // A fresh cell at the outer boundary would require reconstruction.
    drop(scope);
    let scope = TraceContext::enter();
    let local = CellTrace::new(concrete(4));
    let capture = Capture::from_context(scope.context(), [], &[local.into()]).unwrap();
    assert!(matches!(capture.prepare(), Err(EvalPlanError::CellEffect)));
}

#[test]
fn cell_placeholders_can_be_inspected_but_not_evaluated() {
    let cell_placeholder = CellTrace::placeholder(info());
    assert!(matches!(
        Capture::from_roots([&cell_placeholder.get()])
            .unwrap()
            .prepare(),
        Err(EvalPlanError::UnboundPlaceholder)
    ));
}

#[test]
fn retained_values_only_move_out_when_unique() {
    let original = value(2);
    let clone = original.clone();
    let original = original.downcast_into::<Number>().unwrap_err();
    assert!(std::ptr::eq(
        original.downcast::<Number>().unwrap(),
        clone.downcast::<Number>().unwrap()
    ));
    drop(clone);
    assert_eq!(original.downcast_into::<Number>().unwrap().0, 2);
}

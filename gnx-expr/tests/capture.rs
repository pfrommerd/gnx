use std::fmt::Display;

use gnx_expr::trace::{
    CellTrace, Generic, Invocation, ObjectTracer, TraceContext, Tracer, ValueInfo,
};
use gnx_expr::{Dialect, Operation};
use gnx_expr::{HInstr, Op, Capture};

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct SampleOperation {
    name: &'static str,
}
pub struct SampleDialect;

impl Dialect for SampleDialect {
    fn name(&self) -> &'static str {
        "sample"
    }
}

impl std::fmt::Display for SampleOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}::{}", Operation::dialect(self).name(), Operation::name(self))
    }
}

impl Operation for SampleOperation {
    fn name(&self) -> &'static str {
        self.name
    }
    fn dialect(&self) -> &'static dyn Dialect {
        &SampleDialect
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct InfoType;

impl Display for InfoType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "InfoType")
    }
}

fn sample_info() -> ValueInfo {
    ValueInfo::new(InfoType)
}

fn op(name: &'static str) -> Op {
    Op::new(SampleOperation { name })
}

#[test]
fn capture_linear_chain() {
    let guard = TraceContext::enter();
    let ctx = guard.context();
    let a = Tracer::<Generic>::placeholder(sample_info());
    let b = Tracer::<Generic>::placeholder(sample_info());
    let [t1]: [Tracer<Generic>; 1] =
        Invocation::invoke(op("sin"), vec![], vec![b.clone()], vec![sample_info()])
            .try_into()
            .expect("expected exactly one output");
    let [t2]: [Tracer<Generic>; 1] =
        Invocation::invoke(op("mul"), vec![], vec![t1, b.clone()], vec![sample_info()])
            .try_into()
            .expect("expected exactly one output");
    let t3 = Invocation::invoke(op("add"), vec![], vec![a.clone(), t2], vec![sample_info()]);

    let cap = Capture::from_context(
        ctx,
        &[ObjectTracer::from(a.clone()), ObjectTracer::from(b.clone())],
        &[ObjectTracer::from(t3[0].clone())],
    )
    .unwrap();
    let ex = cap.expr();
    assert_eq!(ex.closure().len(), 0);
    assert_eq!(ex.inputs().len(), 2);
    assert_eq!(ex.instructions().len(), 3);
    assert_eq!(ex.outputs().len(), 1);

    let s = ex.to_string();
    assert!(s.contains("sin"));
    assert!(s.contains("mul"));
    assert!(s.contains("add"));
}

#[test]
fn capture_diamond_distinct_intermediate_vars() {
    let guard = TraceContext::enter();
    let ctx = guard.context();
    let x = Tracer::<Generic>::placeholder(sample_info());
    let [u] = Invocation::invoke(op("abs"), vec![], vec![x.clone()], vec![sample_info()])
        .try_into()
        .expect("expected exactly one output");
    let [v] = Invocation::invoke(op("neg"), vec![], vec![x.clone()], vec![sample_info()])
        .try_into()
        .expect("expected exactly one output");
    let [w] = Invocation::invoke(
        op("add"),
        vec![],
        vec![u.clone(), v.clone()],
        vec![sample_info()],
    )
    .try_into()
    .expect("expected exactly one output");

    let cap = Capture::from_context(
        ctx,
        &[ObjectTracer::from(x.clone())],
        &[ObjectTracer::from(w.clone())],
    )
    .unwrap();
    assert_eq!(cap.expr().instructions().len(), 3);
    let add = cap
        .expr()
        .instructions()
        .last()
        .unwrap()
        .as_apply()
        .unwrap();
    assert_eq!(add.inputs().len(), 2);
    assert_ne!(&add.inputs()[0], &add.inputs()[1]);
}

#[test]
fn capture_reuses_var_when_same_tracer_used_twice() {
    let guard = TraceContext::enter();
    let ctx = guard.context();
    let x = Tracer::<Generic>::placeholder(sample_info());
    let [w]: [Tracer<Generic>; 1] = Invocation::invoke(
        op("add"),
        vec![],
        vec![x.clone(), x.clone()],
        vec![sample_info()],
    )
    .try_into()
    .expect("expected exactly one output");
    let cap = Capture::from_context(
        ctx,
        &[ObjectTracer::from(x.clone())],
        &[ObjectTracer::from(w.clone())],
    )
    .unwrap();
    let add = cap
        .expr()
        .instructions()
        .last()
        .unwrap()
        .as_apply()
        .unwrap();
    assert_eq!(&add.inputs()[0], &add.inputs()[1]);
}

#[test]
fn capture_placeholder_only() {
    let guard = TraceContext::enter();
    let ctx = guard.context();
    let x = Tracer::<Generic>::placeholder(sample_info());
    let cap = Capture::from_context(
        ctx,
        &[ObjectTracer::from(x.clone())],
        &[ObjectTracer::from(x.clone())],
    )
    .unwrap();
    let ex = cap.expr();
    assert!(ex.instructions().is_empty());
    assert_eq!(ex.inputs().len(), 1);
    assert_eq!(
        ex.outputs(),
        ex.inputs().iter().map(|i| i.0).collect::<Vec<_>>()
    );
}

#[test]
fn capture_foreign_context_tracer_is_closure() {
    let outer = TraceContext::enter();
    let b = Tracer::<Generic>::placeholder(sample_info());

    let inner = TraceContext::enter();
    let ctx = inner.context();
    let a = Tracer::<Generic>::placeholder(sample_info());
    let [w]: [Tracer<Generic>; 1] = Invocation::invoke(
        op("add"),
        vec![],
        vec![a.clone(), b.clone()],
        vec![sample_info()],
    )
    .try_into()
    .expect("expected exactly one output");

    let cap = Capture::from_context(
        ctx,
        &[ObjectTracer::from(a.clone())],
        &[ObjectTracer::from(w.clone())],
    )
    .unwrap();
    assert_eq!(cap.expr().inputs().len(), 1);
    assert_eq!(cap.expr().closure().len(), 1);
    assert_eq!(cap.closure().len(), 1);
    let add = cap
        .expr()
        .instructions()
        .last()
        .unwrap()
        .as_apply()
        .unwrap();
    assert_ne!(
        &add.inputs()[0],
        &add.inputs()[1],
        "a and outer b must be distinct vars"
    );
    drop(inner);
    drop(outer);
}

#[test]
fn capture_escaping_aliased_write_is_preserved() {
    // A write into a foreign (maybe-aliased) cell is an escaping side effect:
    // it must be retained in the captured expr even when no output depends on
    // it (seeded from the aliased-write chain).
    let outer = TraceContext::enter();
    let _outer_ctx = outer.context();
    let value = Tracer::<Generic>::placeholder(sample_info());
    let cell = CellTrace::new(value.trace_ref().clone());

    let inner = TraceContext::enter();
    let inner_ctx = inner.context();
    let new_val = Tracer::<Generic>::placeholder(sample_info());
    cell.set(new_val.trace_ref().clone());

    let cap = Capture::from_context(inner_ctx, &[ObjectTracer::from(new_val)], &[]).unwrap();
    assert_eq!(
        cap.expr().instructions().len(),
        1,
        "escaping write into a maybe-aliased cell is preserved"
    );
    assert!(matches!(
        cap.expr().instructions()[0],
        HInstr::WriteCell { .. }
    ));
    drop(inner);
    drop(outer);
}

#[test]
fn capture_aliased_update_preserves_value_and_producer() {
    let guard = TraceContext::enter();
    let ctx = guard.context();
    let cell = CellTrace::placeholder(sample_info());
    let base = Tracer::<Generic>::placeholder(sample_info());
    let [value]: [Tracer<Generic>; 1] =
        Invocation::invoke(op("neg"), vec![], vec![base.clone()], vec![sample_info()])
            .try_into()
            .unwrap();
    cell.set(value.trace_ref().clone());

    let cap = Capture::from_context(
        ctx,
        &[ObjectTracer::from(cell), ObjectTracer::from(base)],
        &[],
    )
    .unwrap();
    let expr = cap.expr();
    assert_eq!(
        expr.instructions().len(),
        2,
        "the assigned value's producer must survive"
    );
    let producer = expr.instructions()[0].as_apply().unwrap();
    let HInstr::WriteCell { cell, value } = &expr.instructions()[1] else {
        panic!("expected structural write");
    };
    assert_eq!(producer.op().to_string(), "sample::neg");
    assert_eq!(cell, &expr.inputs()[0].0);
    assert_eq!(expr.cell_access(cell), Some(gnx_expr::CellAccess::Write));
    assert_eq!(expr.cell_access(&expr.inputs()[1].0), None);
    assert_eq!(value, &producer.outputs()[0].0);
}

#[test]
fn capture_foreign_cell_update_preserves_assigned_closure_value() {
    let outer = TraceContext::enter();
    let base = Tracer::<Generic>::placeholder(sample_info());
    let cell = CellTrace::new(base.trace_ref().clone());
    let value = Tracer::<Generic>::placeholder(sample_info());

    let inner = TraceContext::enter();
    cell.set(value.trace_ref().clone());
    let cap = Capture::from_context(inner.context(), vec![], vec![]).unwrap();
    let expr = cap.expr();
    assert_eq!(expr.instructions().len(), 1);
    assert_eq!(
        expr.closure().len(),
        2,
        "capture both target and assigned value"
    );
    let HInstr::WriteCell {
        cell,
        value: assigned,
    } = &expr.instructions()[0]
    else {
        panic!("expected structural write");
    };
    assert_eq!(cell, &expr.closure()[0].0);
    assert_eq!(assigned, &expr.closure()[1].0);
    let ObjectTracer::Ref(captured_value) = &cap.closure()[1] else {
        panic!("assigned value must be an immutable trace");
    };
    assert!(
        gnx_expr::trace::TracerKey::from(captured_value)
            == gnx_expr::trace::TracerKey::from(value.trace_ref()),
        "capture must retain the exact assigned value",
    );
    drop(inner);
    drop(outer);
}

#[test]
fn capture_nonaliased_dead_store_eliminated() {
    // Two writes to an in-context (non-aliased) cell with no read between them:
    // the first store is dead and should not appear in the capture.
    let guard = TraceContext::enter();
    let ctx = guard.context();
    let base = Tracer::<Generic>::placeholder(sample_info());
    let cell = CellTrace::new(base.trace_ref().clone());

    let [v1]: [Tracer<Generic>; 1] =
        Invocation::invoke(op("abs"), vec![], vec![base.clone()], vec![sample_info()])
            .try_into()
            .unwrap();
    cell.set(v1.trace_ref().clone()); // dead store

    let [v2]: [Tracer<Generic>; 1] =
        Invocation::invoke(op("neg"), vec![], vec![base.clone()], vec![sample_info()])
            .try_into()
            .unwrap();
    cell.set(v2.trace_ref().clone()); // live store (escapes via output)

    let out = ObjectTracer::Ref(cell.get());
    let cap =
        Capture::from_context(ctx, &[ObjectTracer::from(base.clone())], &[out]).unwrap();
    let ops: Vec<String> = cap
        .expr()
        .instructions()
        .iter()
        .map(|e| e.as_apply().unwrap().op().to_string())
        .collect();
    assert_eq!(ops, vec!["sample::neg"], "dead store should be eliminated");
}

#[test]
fn capture_aliased_read_write_ordering_preserved() {
    // An input (maybe-aliased) cell read twice, then written: the reads must be
    // ordered before the write (WAR), though the two reads may appear in either
    // order (RAR).
    let guard = TraceContext::enter();
    let ctx = guard.context();
    let cell = CellTrace::placeholder(sample_info());

    let _r1 = cell.get();
    let _r2 = cell.get();
    let w = Tracer::<Generic>::placeholder(sample_info());
    cell.set(w.trace_ref().clone());

    let cap =
        Capture::from_context(ctx, &[ObjectTracer::from(cell), ObjectTracer::from(w)], &[])
            .unwrap();
    let instructions = cap.expr().instructions();
    assert_eq!(instructions.len(), 3);
    assert!(matches!(instructions[0], HInstr::ReadCell { .. }));
    assert!(matches!(instructions[1], HInstr::ReadCell { .. }));
    assert!(matches!(instructions[2], HInstr::WriteCell { .. }));
}

#[test]
fn capture_trace_cell_in_context_override_no_update_eqn() {
    let guard = TraceContext::enter();
    let ctx = guard.context();
    let base = Tracer::<Generic>::placeholder(sample_info());
    let cell = CellTrace::new(base.trace_ref().clone());
    let replacement = Tracer::<Generic>::placeholder(sample_info());
    cell.set(replacement.trace_ref().clone());

    let _ = ctx;
    assert_eq!(
        cell.get().context_id(),
        replacement.context_id(),
        "in-context set updates the cell value"
    );
}

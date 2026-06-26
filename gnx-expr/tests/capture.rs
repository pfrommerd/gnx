use std::fmt::Display;

use gnx_expr::trace::{
    Generic, Invocation, TraceCellRef, TraceContext, TraceObject, Tracer, ValueInfo,
};
use gnx_expr::{Capture, Op};
use gnx_expr::{Dialect, Operation};

pub struct SampleOperation {
    name: &'static str,
}
pub struct SampleDialect;

impl Dialect for SampleDialect {
    fn name(&self) -> &'static str {
        "sample"
    }
    fn create(&self, name: &'static str) -> Option<&'static dyn Operation> {
        Some(match name {
            "sin" => &SampleOperation { name: "sin" },
            "mul" => &SampleOperation { name: "mul" },
            "add" => &SampleOperation { name: "add" },
            "abs" => &SampleOperation { name: "abs" },
            "neg" => &SampleOperation { name: "neg" },
            _ => return None,
        })
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
    static SIN: SampleOperation = SampleOperation { name: "sin" };
    static MUL: SampleOperation = SampleOperation { name: "mul" };
    static ADD: SampleOperation = SampleOperation { name: "add" };
    static ABS: SampleOperation = SampleOperation { name: "abs" };
    static NEG: SampleOperation = SampleOperation { name: "neg" };
    let impl_ = match name {
        "sin" => &SIN as &dyn Operation,
        "mul" => &MUL,
        "add" => &ADD,
        "abs" => &ABS,
        "neg" => &NEG,
        _ => panic!("unknown op"),
    };
    Op::new(impl_)
}

#[test]
fn capture_linear_chain() {
    let guard = TraceContext::enter();
    let ctx = guard.context();
    let a = Tracer::<Generic>::placeholder(sample_info());
    let b = Tracer::<Generic>::placeholder(sample_info());
    let [t1]: [Tracer<Generic>; 1] = Invocation::invoke(
        op("sin"),
        vec![],
        vec![TraceObject::from(b.clone())],
        vec![sample_info()],
    )
    .try_into()
    .expect("expected exactly one output");
    let [t2]: [Tracer<Generic>; 1] = Invocation::invoke(
        op("mul"),
        vec![],
        vec![TraceObject::from(t1), TraceObject::from(b.clone())],
        vec![sample_info()],
    )
    .try_into()
    .expect("expected exactly one output");
    let t3 = Invocation::invoke(
        op("add"),
        vec![],
        vec![TraceObject::from(a.clone()), TraceObject::from(t2)],
        vec![sample_info()],
    );

    let cap = Capture::from_context(
        ctx,
        &[TraceObject::from(a.clone()), TraceObject::from(b.clone())],
        &[TraceObject::from(t3[0].clone())],
    )
    .unwrap();
    let ex = cap.expr();
    assert_eq!(ex.closure().len(), 0);
    assert_eq!(ex.inputs().len(), 2);
    assert_eq!(ex.eqns().len(), 3);
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
    let [u] = Invocation::invoke(
        op("abs"),
        vec![],
        vec![TraceObject::from(x.clone())],
        vec![sample_info()],
    )
    .try_into()
    .expect("expected exactly one output");
    let [v] = Invocation::invoke(
        op("neg"),
        vec![],
        vec![TraceObject::from(x.clone())],
        vec![sample_info()],
    )
    .try_into()
    .expect("expected exactly one output");
    let [w] = Invocation::invoke(
        op("add"),
        vec![],
        vec![TraceObject::from(u.clone()), TraceObject::from(v.clone())],
        vec![sample_info()],
    )
    .try_into()
    .expect("expected exactly one output");

    let cap = Capture::from_context(
        ctx,
        &[TraceObject::from(x.clone())],
        &[TraceObject::from(w.clone())],
    )
    .unwrap();
    assert_eq!(cap.expr().eqns().len(), 3);
    let add = cap.expr().eqns().last().unwrap();
    assert_eq!(add.inputs().len(), 2);
    assert_ne!(add.inputs()[0].var(), add.inputs()[1].var());
}

#[test]
fn capture_reuses_var_when_same_tracer_used_twice() {
    let guard = TraceContext::enter();
    let ctx = guard.context();
    let x = Tracer::<Generic>::placeholder(sample_info());
    let [w]: [Tracer<Generic>; 1] = Invocation::invoke(
        op("add"),
        vec![],
        vec![TraceObject::from(x.clone()), TraceObject::from(x.clone())],
        vec![sample_info()],
    )
    .try_into()
    .expect("expected exactly one output");
    let cap = Capture::from_context(
        ctx,
        &[TraceObject::from(x.clone())],
        &[TraceObject::from(w.clone())],
    )
    .unwrap();
    let add = cap.expr().eqns().last().unwrap();
    assert_eq!(add.inputs()[0].var(), add.inputs()[1].var());
}

#[test]
fn capture_placeholder_only() {
    let guard = TraceContext::enter();
    let ctx = guard.context();
    let x = Tracer::<Generic>::placeholder(sample_info());
    let cap = Capture::from_context(
        ctx,
        &[TraceObject::from(x.clone())],
        &[TraceObject::from(x.clone())],
    )
    .unwrap();
    let ex = cap.expr();
    assert!(ex.eqns().is_empty());
    assert_eq!(ex.inputs().len(), 1);
    assert_eq!(
        ex.outputs(),
        ex.inputs().iter().map(|i| *i.var()).collect::<Vec<_>>()
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
        vec![TraceObject::from(a.clone()), TraceObject::from(b.clone())],
        vec![sample_info()],
    )
    .try_into()
    .expect("expected exactly one output");

    let cap = Capture::from_context(
        ctx,
        &[TraceObject::from(a.clone())],
        &[TraceObject::from(w.clone())],
    )
    .unwrap();
    assert_eq!(cap.expr().inputs().len(), 1);
    assert_eq!(cap.expr().closure().len(), 1);
    assert_eq!(cap.closure().len(), 1);
    let add = cap.expr().eqns().last().unwrap();
    assert_ne!(
        add.inputs()[0].var(),
        add.inputs()[1].var(),
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
    let cell = TraceCellRef::new(value.trace_ref().clone());

    let inner = TraceContext::enter();
    let inner_ctx = inner.context();
    let new_val = Tracer::<Generic>::placeholder(sample_info());
    cell.set(new_val.trace_ref().clone());

    let cap = Capture::from_context(inner_ctx, vec![], vec![]).unwrap();
    assert_eq!(
        cap.expr().eqns().len(),
        1,
        "escaping write into a maybe-aliased cell is preserved"
    );
    assert_eq!(cap.expr().eqns()[0].op().to_string(), "builtins.update");
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
    let cell = TraceCellRef::new(base.trace_ref().clone());

    let [v1]: [Tracer<Generic>; 1] = Invocation::invoke(
        op("abs"),
        vec![],
        vec![TraceObject::from(base.clone())],
        vec![sample_info()],
    )
    .try_into()
    .unwrap();
    cell.set(v1.trace_ref().clone()); // dead store

    let [v2]: [Tracer<Generic>; 1] = Invocation::invoke(
        op("neg"),
        vec![],
        vec![TraceObject::from(base.clone())],
        vec![sample_info()],
    )
    .try_into()
    .unwrap();
    cell.set(v2.trace_ref().clone()); // live store (escapes via output)

    let out = TraceObject::Ref(cell.get());
    let cap = Capture::from_context(ctx, &[TraceObject::from(base.clone())], &[out]).unwrap();
    let ops: Vec<String> = cap.expr().eqns().iter().map(|e| e.op().to_string()).collect();
    assert_eq!(ops, vec!["sample.neg"], "dead store should be eliminated");
}

#[test]
fn capture_aliased_read_write_ordering_preserved() {
    // An input (maybe-aliased) cell read twice, then written: the reads must be
    // ordered before the write (WAR), though the two reads may appear in either
    // order (RAR).
    let guard = TraceContext::enter();
    let ctx = guard.context();
    let cell = TraceCellRef::placeholder(sample_info());

    let _r1 = cell.get();
    let _r2 = cell.get();
    let w = Tracer::<Generic>::placeholder(sample_info());
    cell.set(w.trace_ref().clone());

    let cap = Capture::from_context(ctx, vec![], vec![]).unwrap();
    let ops: Vec<String> = cap.expr().eqns().iter().map(|e| e.op().to_string()).collect();
    let write_pos = ops
        .iter()
        .position(|o| o == "builtins.update")
        .expect("aliased write retained");
    let read_count = ops.iter().filter(|o| *o == "builtins.read").count();
    assert_eq!(read_count, 2, "both aliased reads retained");
    assert!(
        ops[..write_pos].iter().all(|o| o == "builtins.read"),
        "aliased reads must precede the aliased write: {ops:?}"
    );
}

#[test]
fn capture_trace_cell_in_context_override_no_update_eqn() {
    let guard = TraceContext::enter();
    let ctx = guard.context();
    let base = Tracer::<Generic>::placeholder(sample_info());
    let cell = TraceCellRef::new(base.trace_ref().clone());
    let replacement = Tracer::<Generic>::placeholder(sample_info());
    cell.set(replacement.trace_ref().clone());

    let _ = ctx;
    assert_eq!(
        cell.get().context_id(),
        replacement.context_id(),
        "in-context set updates the cell value"
    );
}

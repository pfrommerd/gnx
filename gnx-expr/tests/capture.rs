use gnx_expr::array::{ArrayInfo, DType, Shape};
use gnx_expr::expr::{AttrMap, Capture, Effect, Op};
use gnx_expr::trace::{
    Generic, Invocation, TraceCell, TraceContext, TraceOperand, Tracer, ValueInfo,
};
use gnx_expr::expr::{Dialect, Operation};

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

fn f32x4_info() -> ValueInfo {
    ValueInfo::Array(ArrayInfo::new(Shape::fixed([4]), DType::F32))
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
    Op::new(impl_, AttrMap::default())
}

#[test]
fn capture_linear_chain() {
    let guard = TraceContext::enter();
    let ctx = guard.context();
    let a = Tracer::<Generic>::placeholder(f32x4_info());
    let b = Tracer::<Generic>::placeholder(f32x4_info());
    let t1 = Invocation::invoke(
        op("sin"),
        vec![],
        vec![TraceOperand::from_ref(&b)],
        vec![f32x4_info()],
    );
    let t2 = Invocation::invoke(
        op("mul"),
        vec![],
        vec![
            TraceOperand::from_ref(&t1[0]),
            TraceOperand::from_ref(&b),
        ],
        vec![f32x4_info()],
    );
    let t3 = Invocation::invoke(
        op("add"),
        vec![],
        vec![
            TraceOperand::from_ref(&a),
            TraceOperand::from_ref(&t2[0]),
        ],
        vec![f32x4_info()],
    );

    let cap = Capture::from_context(ctx, &[&a, &b], &[&t3[0]]).unwrap();
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
    let x = Tracer::<Generic>::placeholder(f32x4_info());
    let u = Invocation::invoke(
        op("abs"),
        vec![],
        vec![TraceOperand::from_ref(&x)],
        vec![f32x4_info()],
    );
    let v = Invocation::invoke(
        op("neg"),
        vec![],
        vec![TraceOperand::from_ref(&x)],
        vec![f32x4_info()],
    );
    let w = Invocation::invoke(
        op("add"),
        vec![],
        vec![
            TraceOperand::from_ref(&u[0]),
            TraceOperand::from_ref(&v[0]),
        ],
        vec![f32x4_info()],
    );

    let cap = Capture::from_context(ctx, &[&x], &[&w[0]]).unwrap();
    assert_eq!(cap.expr().eqns().len(), 3);
    let add = cap.expr().eqns().last().unwrap();
    assert_eq!(add.inputs().len(), 2);
    assert_ne!(add.inputs()[0].var(), add.inputs()[1].var());
}

#[test]
fn capture_reuses_var_when_same_tracer_used_twice() {
    let guard = TraceContext::enter();
    let ctx = guard.context();
    let x = Tracer::<Generic>::placeholder(f32x4_info());
    let w = Invocation::invoke(
        op("add"),
        vec![],
        vec![
            TraceOperand::from_ref(&x),
            TraceOperand::from_ref(&x),
        ],
        vec![f32x4_info()],
    );
    let cap = Capture::from_context(ctx, &[&x], &[&w[0]]).unwrap();
    let add = cap.expr().eqns().last().unwrap();
    assert_eq!(add.inputs()[0].var(), add.inputs()[1].var());
}

#[test]
fn capture_placeholder_only() {
    let guard = TraceContext::enter();
    let ctx = guard.context();
    let x = Tracer::<Generic>::placeholder(f32x4_info());
    let cap = Capture::from_context(ctx, &[&x], &[&x]).unwrap();
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
    let b = Tracer::<Generic>::placeholder(f32x4_info());

    let inner = TraceContext::enter();
    let ctx = inner.context();
    let a = Tracer::<Generic>::placeholder(f32x4_info());
    let w = Invocation::invoke(
        op("add"),
        vec![],
        vec![
            TraceOperand::from_ref(&a),
            TraceOperand::from_ref(&b),
        ],
        vec![f32x4_info()],
    );

    let cap = Capture::from_context(ctx, &[&a], &[&w[0]]).unwrap();
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
fn capture_trace_cell_cross_context_update_emits_gnx_update() {
    let outer = TraceContext::enter();
    let _outer_ctx = outer.context();
    let value = Tracer::<Generic>::placeholder(f32x4_info());
    let cell = TraceCell::new(value.trace_ref().clone());

    let inner = TraceContext::enter();
    let inner_ctx = inner.context();
    let new_val = Tracer::<Generic>::placeholder(f32x4_info());
    cell.set(new_val.trace_ref().clone());

    assert_eq!(inner_ctx.updates().len(), 1);

    let cap = Capture::from_context(
        inner_ctx,
        std::iter::empty::<&Tracer<Generic>>(),
        std::iter::empty::<&Tracer<Generic>>(),
    )
    .unwrap();
    assert_eq!(cap.expr().eqns().len(), 1);
    let update_eqn = &cap.expr().eqns()[0];
    assert!(update_eqn.op().to_string().contains("update"));
    assert_eq!(update_eqn.inputs().len(), 2);
    assert_eq!(update_eqn.inputs()[0].effect(), Effect::Write);
    assert_eq!(update_eqn.inputs()[1].effect(), Effect::Read);
    drop(inner);
    drop(outer);
}

#[test]
fn capture_trace_cell_in_context_override_no_update_eqn() {
    let guard = TraceContext::enter();
    let ctx = guard.context();
    let base = Tracer::<Generic>::placeholder(f32x4_info());
    let cell = TraceCell::new(base.trace_ref().clone());
    let replacement = Tracer::<Generic>::placeholder(f32x4_info());
    cell.set(replacement.trace_ref().clone());

    assert!(ctx.updates().is_empty());
    assert_eq!(
        cell.get().context_id(),
        replacement.context_id(),
        "in-context set records an override"
    );
}

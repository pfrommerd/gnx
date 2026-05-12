use gnx_expr::array::{ArrayInfo, DType, Shape};
use gnx_expr::expr::{AttrMap, Capture, Effect, Op, OpString};
use gnx_expr::trace::{Generic, Invocation, Tracer, ValueInfo};

fn sample_op(name: &'static str) -> Op {
    Op {
        dialect: OpString::Static("test"),
        name: OpString::Static(name),
        attrs: AttrMap::default(),
    }
}

fn f32x4_info() -> ValueInfo {
    ValueInfo::Array(ArrayInfo::new(
        Shape::from_dims([4]),
        DType::F32,
    ))
}

#[test]
fn capture_linear_chain() {
    let a = Tracer::<Generic>::placeholder(f32x4_info());
    let b = Tracer::<Generic>::placeholder(f32x4_info());
    let t1 = Invocation::invoke(
        sample_op("sin"),
        vec![],
        vec![(b.clone(), Effect::Read)],
        vec![f32x4_info()],
    );
    let t2 = Invocation::invoke(
        sample_op("mul"),
        vec![],
        vec![(t1[0].clone(), Effect::Read), (b.clone(), Effect::Read)],
        vec![f32x4_info()],
    );
    let t3 = Invocation::invoke(
        sample_op("add"),
        vec![],
        vec![(a.clone(), Effect::Read), (t2[0].clone(), Effect::Read)],
        vec![f32x4_info()],
    );

    let cap = Capture::from_trace_refs(
        &[a.trace_ref(), b.trace_ref()],
        &[t3[0].trace_ref()],
    ).unwrap();
    let ex = cap.expr();
    assert_eq!(ex.closure().len(), 0);
    assert_eq!(ex.inputs().len(), 2, "a and b (b used twice → one Var)");
    assert_eq!(ex.eqns().len(), 3);
    assert_eq!(ex.outputs().len(), 1);

    let s = ex.to_string();
    assert!(s.contains("sin"));
    assert!(s.contains("mul"));
    assert!(s.contains("add"));
}

#[test]
fn capture_diamond_distinct_intermediate_vars() {
    let x = Tracer::<Generic>::placeholder(f32x4_info());
    let u = Invocation::invoke(
        sample_op("abs"),
        vec![],
        vec![(x.clone(), Effect::Read)],
        vec![f32x4_info()],
    );
    let v = Invocation::invoke(
        sample_op("neg"),
        vec![],
        vec![(x.clone(), Effect::Read)],
        vec![f32x4_info()],
    );
    let w = Invocation::invoke(
        sample_op("add"),
        vec![],
        vec![(u[0].clone(), Effect::Read), (v[0].clone(), Effect::Read)],
        vec![f32x4_info()],
    );

    let cap = Capture::from_trace_refs(&[x.trace_ref()], &[w[0].trace_ref()]).unwrap();
    assert_eq!(cap.expr().eqns().len(), 3);
    let add = cap.expr().eqns().last().unwrap();
    assert_eq!(add.inputs().len(), 2);
    assert_ne!(
        add.inputs()[0], add.inputs()[1],
        "abs and neg outputs are distinct tracers → distinct vars"
    );
}

#[test]
fn capture_reuses_var_when_same_tracer_used_twice() {
    let x = Tracer::<Generic>::placeholder(f32x4_info());
    let w = Invocation::invoke(
        sample_op("add"),
        vec![],
        vec![(x.clone(), Effect::Read), (x.clone(), Effect::Read)],
        vec![f32x4_info()],
    );
    let cap = Capture::from_trace_refs(&[x.trace_ref()], &[w[0].trace_ref()]).unwrap();
    let add = cap.expr().eqns().last().unwrap();
    assert_eq!(add.inputs()[0], add.inputs()[1]);
}

#[test]
fn capture_placeholder_only() {
    let x = Tracer::<Generic>::placeholder(f32x4_info());
    let cap = Capture::from_trace_refs(&[x.trace_ref()], &[x.trace_ref()]).unwrap();
    let ex = cap.expr();
    assert!(ex.eqns().is_empty());
    assert_eq!(ex.inputs().len(), 1);
    assert_eq!(ex.outputs(), ex.inputs().iter().map(|i| i.var().clone()).collect::<Vec<_>>());
}

#[test]
fn capture_unlisted_leaf_tracer_is_closure() {
    let a = Tracer::<Generic>::placeholder(f32x4_info());
    let b = Tracer::<Generic>::placeholder(f32x4_info());
    let w = Invocation::invoke(
        sample_op("add"),
        vec![],
        vec![(a.clone(), Effect::Read), (b.clone(), Effect::Read)],
        vec![f32x4_info()],
    );
    let cap = Capture::from_trace_refs(&[a.trace_ref()], &[w[0].trace_ref()]).unwrap();
    assert_eq!(cap.expr().inputs().len(), 1);
    assert_eq!(cap.expr().closure().len(), 1);
    assert_eq!(cap.closure().len(), 1);
    assert!(std::ptr::eq(
        std::sync::Arc::as_ptr(&cap.closure()[0]),
        std::sync::Arc::as_ptr(&b.trace_ref()),
    ));
}

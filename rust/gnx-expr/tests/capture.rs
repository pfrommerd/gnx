use gnx_expr::array::{ArrayInfo, DType, Shape};
use gnx_expr::expr::{AttrMap, Capture, Op, OpString};
use gnx_expr::trace::{Generic, Tracer, ValueInfo};

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
    let t1 = Tracer::<Generic>::invoke(
        sample_op("sin"),
        vec![b.clone().generic()],
        vec![f32x4_info()],
    );
    let t2 = Tracer::<Generic>::invoke(
        sample_op("mul"),
        vec![t1[0].clone(), b.clone().generic()],
        vec![f32x4_info()],
    );
    let t3 = Tracer::<Generic>::invoke(
        sample_op("add"),
        vec![a.clone().generic(), t2[0].clone()],
        vec![f32x4_info()],
    );

    let cap = Capture::from_tracers([&a, &b], [&t3[0]]);
    let ex = cap.expr();
    assert_eq!(ex.closure_inputs().len(), 0);
    assert_eq!(
        ex.explicit_inputs().len(),
        2,
        "a and b (b used twice → one Var)"
    );
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
    let u = Tracer::<Generic>::invoke(
        sample_op("abs"),
        vec![x.clone().generic()],
        vec![f32x4_info()],
    );
    let v = Tracer::<Generic>::invoke(
        sample_op("neg"),
        vec![x.clone().generic()],
        vec![f32x4_info()],
    );
    let w = Tracer::<Generic>::invoke(
        sample_op("add"),
        vec![u[0].clone(), v[0].clone()],
        vec![f32x4_info()],
    );

    let cap = Capture::from_tracers([&x], [&w[0]]);
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
    let w = Tracer::<Generic>::invoke(
        sample_op("add"),
        vec![x.clone().generic(), x.clone().generic()],
        vec![f32x4_info()],
    );
    let cap = Capture::from_tracers([&x], [&w[0]]);
    let add = cap.expr().eqns().last().unwrap();
    assert_eq!(add.inputs()[0], add.inputs()[1]);
}

#[test]
fn capture_placeholder_only() {
    let x = Tracer::<Generic>::placeholder(f32x4_info());
    let cap = Capture::from_tracers([&x], [&x]);
    let ex = cap.expr();
    assert!(ex.eqns().is_empty());
    assert_eq!(ex.explicit_inputs().len(), 1);
    assert_eq!(ex.outputs(), ex.explicit_inputs());
}

#[test]
fn capture_unlisted_leaf_tracer_is_closure() {
    let a = Tracer::<Generic>::placeholder(f32x4_info());
    let b = Tracer::<Generic>::placeholder(f32x4_info());
    let w = Tracer::<Generic>::invoke(
        sample_op("add"),
        vec![a.clone().generic(), b.clone().generic()],
        vec![f32x4_info()],
    );
    let cap = Capture::from_tracers([&a], [&w[0]]);
    assert_eq!(cap.expr().explicit_inputs().len(), 1);
    assert_eq!(cap.expr().closure_inputs().len(), 1);
    assert_eq!(cap.closure().len(), 1);
    assert_eq!(cap.closure()[0].addr(), b.addr());
}

use gnx_expr::array::{ArrayInfo, DType, Shape};
use gnx_expr::trace::{Generic, Tracer, ValueInfo};

fn f32x4_info() -> ValueInfo {
    ValueInfo::Array(ArrayInfo::new(Shape::fixed([4]), DType::F32))
}

#[test]
fn clone_tracer_retains_only_top_frame() {
    let mut stacked = Tracer::<Generic>::placeholder(f32x4_info());
    let overlay = Tracer::<Generic>::placeholder(f32x4_info());
    stacked.push(overlay.trace_ref());
    assert_eq!(stacked.stack_depth(), 2);
    let cloned = stacked.clone();
    assert_eq!(cloned.stack_depth(), 1);
    assert!(std::ptr::eq(
        std::sync::Arc::as_ptr(&cloned.trace_ref()),
        std::sync::Arc::as_ptr(&stacked.trace_ref()),
    ));
}

#[test]
fn pop_restores_outer_frame() {
    let mut stacked = Tracer::<Generic>::placeholder(f32x4_info());
    let base_ptr = std::sync::Arc::as_ptr(&stacked.trace_ref());
    stacked.push(Tracer::<Generic>::placeholder(f32x4_info()).trace_ref());
    assert_eq!(stacked.stack_depth(), 2);
    stacked.pop();
    assert_eq!(stacked.stack_depth(), 1);
    assert_eq!(
        std::sync::Arc::as_ptr(&stacked.trace_ref()),
        base_ptr
    );
}
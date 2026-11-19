use gnx_array::{Array, ArrayInfo, ArrayRef, DType, Shape};
use gnx_expr::trace::{CellKey, CellTrace, TraceContext, Trace, Tracer, TracerKey};
use gnx_expr::ValueInfo;
use gnx_graph::{Filter, Graph, GraphContext, GraphSource, Of, TypedGraph};

fn f32x4_info() -> ValueInfo {
    ValueInfo::new(ArrayInfo::new(Shape::fixed([4]), DType::F32))
}

#[test]
fn of_trace_ref_extracts_array_inner_trace() {
    let _guard = TraceContext::enter();
    let tracer = Tracer::<Array>::placeholder(f32x4_info());
    let inner = tracer.trace_ref().clone();
    let array = Array::from(tracer);

    let extracted: Trace = array.as_source().leaf(()).unwrap();
    assert!(TracerKey::from(&extracted) == TracerKey::from(&inner));

    assert!(Of::<Trace>::filter().matches_ref(&array).is_err());
    assert!(
        Of::<Trace>::filter()
            .matches_ref(array.tracer().trace_ref())
            .is_ok()
    );
}

#[test]
fn of_trace_ref_replace_array_inner_trace() {
    let _guard = TraceContext::enter();
    let array = Array::from(Tracer::<Array>::placeholder(f32x4_info()));
    let replacement = Tracer::<Array>::placeholder(f32x4_info())
        .trace_ref()
        .clone();

    let replaced = array
        .replace(
            Of::<Trace>::filter(),
            replacement.as_source(),
            &mut GraphContext::new(),
        )
        .unwrap();

    let got: Trace = replaced.as_source().leaf(()).unwrap();
    assert!(TracerKey::from(&got) == TracerKey::from(&replacement));
}

#[test]
fn of_trace_cell_ref_extracts_array_ref_inner_cell() {
    let _guard = TraceContext::enter();
    let array_ref = ArrayRef::new(Tracer::<Array>::placeholder(f32x4_info()));
    let inner = array_ref.tracer().cell_ref().clone();

    let extracted: CellTrace = array_ref.as_source().leaf(()).unwrap();
    assert_eq!(CellKey::from(&extracted), CellKey::from(&inner));

    assert!(
        Of::<CellTrace>::filter()
            .matches_ref(&array_ref)
            .is_err()
    );
    assert!(
        Of::<CellTrace>::filter()
            .matches_ref(array_ref.tracer().cell_ref())
            .is_ok()
    );
}

#[test]
fn of_trace_cell_ref_replace_array_ref_inner_cell() {
    let _guard = TraceContext::enter();
    let array_ref = ArrayRef::new(Tracer::<Array>::placeholder(f32x4_info()));
    let replacement = CellTrace::new(
        Tracer::<Array>::placeholder(f32x4_info())
            .trace_ref()
            .clone(),
    );

    let replaced = array_ref
        .replace(
            Of::<CellTrace>::filter(),
            replacement.as_source(),
            &mut GraphContext::new(),
        )
        .unwrap();

    let got: CellTrace = replaced.as_source().leaf(()).unwrap();
    assert_eq!(CellKey::from(&got), CellKey::from(&replacement));
}

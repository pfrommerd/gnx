use gnx_base::{Array, ArrayInfo, ArrayRef, DType, Shape};
use gnx_expr::{
    Capture, CellKey, CellTrace, HInstr, ObjectTracer, ObjectTracerRef, Trace, TraceContext,
    Tracer, TracerKey, VarInfo,
};
use gnx_graph::util::GraphExt;
use gnx_graph::{
    Bound, Builder, Filter, Graph, GraphConsumer, GraphContext, GraphId, GraphSource, Leaf, Node,
    Nothing, NullSource, Of, SourceError, TypedGraph, View,
};

fn info() -> VarInfo {
    VarInfo::new(ArrayInfo::new(Shape::fixed([4]), DType::F32))
}
fn trace() -> Trace {
    Tracer::<Array>::placeholder(info()).into_trace_ref()
}
fn cell() -> CellTrace {
    CellTrace::placeholder(info())
}
fn array() -> Array {
    Array::from(Tracer::<Array>::placeholder(info()))
}
fn non_array_trace() -> Trace {
    Trace::placeholder(VarInfo::new(0u32))
}
fn array_ref() -> ArrayRef {
    <ArrayRef as Leaf>::try_from_value(cell()).unwrap()
}

fn collect<G: Graph>(graph: &G) -> Vec<ObjectTracer> {
    let mut objects = Vec::new();
    graph.to_dag(Of::<ObjectTracer>::filter(), |reference| {
        objects.push(reference.to_owned());
    });
    objects
}
fn assert_same(left: &ObjectTracer, right: &ObjectTracer) {
    match (left, right) {
        (ObjectTracer::Ref(a), ObjectTracer::Ref(b)) => {
            assert_eq!(TracerKey::from(a), TracerKey::from(b))
        }
        (ObjectTracer::Cell(a), ObjectTracer::Cell(b)) => {
            assert_eq!(CellKey::from(a), CellKey::from(b))
        }
        _ => panic!("different trace-object variants"),
    }
}
struct TakeLeaf;
impl<G: Graph, L: Leaf> GraphConsumer<G, L> for TakeLeaf {
    type Output = Option<L>;
    fn consume_leaf(self, value: L) -> Self::Output {
        Some(value)
    }
    fn consume_static<S: Leaf>(self, _: S) -> Self::Output {
        None
    }
    fn consume_node<N: Node, F: Filter<L>>(self, _: Bound<N, F>) -> Self::Output {
        panic!("expected leaf")
    }
    fn consume_shared<S: Graph, F: Filter<L>>(self, _: GraphId, _: View<'_, S, F>) -> Self::Output {
        panic!("expected unshared leaf")
    }
}

#[test]
fn borrowed_views_are_copyable_and_clone_only_handles() {
    let _guard = TraceContext::enter();
    let value = ObjectTracer::Ref(trace());
    let mutable = ObjectTracer::Cell(cell());
    for object in [&value, &mutable] {
        let view = object.as_ref();
        let copied = view;
        assert_same(object, &view.to_owned());
        assert_same(object, &copied.to_owned());
        assert_same(object, &<ObjectTracer as Leaf>::clone_ref(object.as_ref()));
    }
    assert!(matches!(value.as_ref(), ObjectTracerRef::Ref(_)));
    assert!(matches!(mutable.as_ref(), ObjectTracerRef::Cell(_)));
}

#[test]
fn raw_handles_convert_to_and_from_the_common_leaf() {
    let _guard = TraceContext::enter();
    let value = trace();
    let mutable = cell();
    let value_object = <ObjectTracer as Leaf>::try_from_value(value.clone()).unwrap();
    let cell_object = <ObjectTracer as Leaf>::try_from_value(mutable.clone()).unwrap();
    let borrowed_value = <ObjectTracer as Leaf>::try_from_ref(&value).unwrap();
    let borrowed_cell = <ObjectTracer as Leaf>::try_from_ref(&mutable).unwrap();
    assert_same(&value_object, &borrowed_value.to_owned());
    assert_same(&cell_object, &borrowed_cell.to_owned());
    let got = <ObjectTracer as Leaf>::try_into_value::<Trace>(value_object).unwrap();
    assert_eq!(TracerKey::from(&got), TracerKey::from(&value));
    let got = <ObjectTracer as Leaf>::try_into_value::<CellTrace>(cell_object).unwrap();
    assert_eq!(CellKey::from(&got), CellKey::from(&mutable));
    assert_eq!(
        <ObjectTracer as Leaf>::try_from_value(17u32).err().unwrap(),
        17
    );
}

#[test]
fn failed_variant_conversions_preserve_the_original_object() {
    let _guard = TraceContext::enter();
    let original = ObjectTracer::Cell(cell());
    let failed = <ObjectTracer as Leaf>::try_into_value::<Trace>(original.clone())
        .err()
        .unwrap();
    assert_same(&original, &failed);
    let original = ObjectTracer::Ref(trace());
    let failed = <ObjectTracer as Leaf>::try_into_value::<CellTrace>(original.clone())
        .err()
        .unwrap();
    assert_same(&original, &failed);
}

#[test]
fn common_filter_visits_each_object_once_and_selective_filters_keep_variants() {
    let _guard = TraceContext::enter();
    let value = ObjectTracer::Ref(trace());
    let mutable = ObjectTracer::Cell(cell());
    let objects = collect(&(value.clone(), mutable.clone()));
    assert_eq!(objects.len(), 2);
    assert_same(&objects[0], &value);
    assert_same(&objects[1], &mutable);
    let got: Trace = value.as_source().leaf(()).unwrap();
    assert_same(&ObjectTracer::Ref(got), &value);
    let got: CellTrace = mutable.as_source().leaf(()).unwrap();
    assert_same(&ObjectTracer::Cell(got), &mutable);
    assert!(Of::<Trace>::filter().matches_ref(&mutable).is_err());
    assert!(Of::<CellTrace>::filter().matches_ref(&value).is_err());
}

#[test]
fn owned_visits_support_both_common_and_selective_filters() {
    let _guard = TraceContext::enter();
    let value = ObjectTracer::Ref(trace());
    let mutable = ObjectTracer::Cell(cell());
    for object in [&value, &mutable] {
        let consumed = object
            .clone()
            .visit_into(Of::<ObjectTracer>::filter(), TakeLeaf)
            .unwrap();
        assert_same(object, &consumed);
    }
    let got = value
        .clone()
        .visit_into(Of::<Trace>::filter(), TakeLeaf)
        .unwrap();
    assert_same(&value, &ObjectTracer::Ref(got));
    let got = mutable
        .clone()
        .visit_into(Of::<CellTrace>::filter(), TakeLeaf)
        .unwrap();
    assert_same(&mutable, &ObjectTracer::Cell(got));
    assert!(
        mutable
            .visit_into(Of::<Trace>::filter(), TakeLeaf)
            .is_none()
    );
}

#[test]
fn raw_handle_graphs_can_be_replaced_and_built_from_trace_objects() {
    let _guard = TraceContext::enter();
    let original = (trace(), cell());
    let replacement = (ObjectTracer::Ref(trace()), ObjectTracer::Cell(cell()));
    let replaced = original
        .replace(
            Of::<ObjectTracer>::filter(),
            replacement.as_source(),
            &mut GraphContext::new(),
        )
        .unwrap();
    let builder = original
        .builder::<ObjectTracer, _, SourceError>(
            Of::<ObjectTracer>::filter(),
            &mut GraphContext::new(),
        )
        .unwrap();
    let rebuilt =
        Builder::<ObjectTracer>::build(builder, replacement.as_source(), &mut GraphContext::new())
            .unwrap();
    for result in [replaced, rebuilt] {
        let objects = collect(&result);
        assert_same(&objects[0], &replacement.0);
        assert_same(&objects[1], &replacement.1);
    }
}

#[test]
fn typed_mixed_graphs_roundtrip_through_one_trace_object_filter() {
    let _guard = TraceContext::enter();
    let original = (array(), non_array_trace(), array_ref(), 42u32);
    let replacement = (
        ObjectTracer::Ref(trace()),
        ObjectTracer::Ref(non_array_trace()),
        ObjectTracer::Cell(cell()),
        99u32,
    );
    let extracted = collect(&original);
    assert_eq!(extracted.len(), 3);
    let replaced = original
        .replace(
            Of::<ObjectTracer>::filter(),
            replacement.as_source(),
            &mut GraphContext::new(),
        )
        .unwrap();
    let builder = original
        .builder::<ObjectTracer, _, SourceError>(
            Of::<ObjectTracer>::filter(),
            &mut GraphContext::new(),
        )
        .unwrap();
    let rebuilt =
        Builder::<ObjectTracer>::build(builder, replacement.as_source(), &mut GraphContext::new())
            .unwrap();
    for result in [replaced, rebuilt] {
        assert_eq!(
            result.3, 42,
            "static fields come from the original graph/builder"
        );
        let objects = collect(&result);
        assert_eq!(objects.len(), 3);
        assert_same(&objects[0], &replacement.0);
        assert_same(&objects[1], &replacement.1);
        assert_same(&objects[2], &replacement.2);
    }
}

#[test]
fn repeated_value_and_cell_handles_keep_their_identities() {
    let _guard = TraceContext::enter();
    let value = array();
    let mutable = array_ref();
    let original = (value.clone(), value, mutable.clone(), mutable);
    let objects = collect(&original);
    assert_same(&objects[0], &objects[1]);
    assert_same(&objects[2], &objects[3]);
    let source = (
        objects[0].clone(),
        objects[1].clone(),
        objects[2].clone(),
        objects[3].clone(),
    );
    let builder = original
        .builder::<ObjectTracer, _, SourceError>(
            Of::<ObjectTracer>::filter(),
            &mut GraphContext::new(),
        )
        .unwrap();
    let rebuilt =
        Builder::<ObjectTracer>::build(builder, source.as_source(), &mut GraphContext::new())
            .unwrap();
    let objects = collect(&rebuilt);
    assert_same(&objects[0], &objects[1]);
    assert_same(&objects[2], &objects[3]);
}

#[test]
fn typed_conversions_reject_wrong_variants_and_metadata_families() {
    let _guard = TraceContext::enter();
    let array_value = ObjectTracer::Ref(trace());
    let array_cell = ObjectTracer::Cell(cell());
    let other_value = ObjectTracer::Ref(non_array_trace());
    let other_cell = ObjectTracer::Cell(CellTrace::placeholder(VarInfo::new(0u32)));
    assert_same(
        &array_cell,
        &<Array as Leaf>::try_from_value(array_cell.clone())
            .err()
            .unwrap(),
    );
    assert_same(
        &array_value,
        &<ArrayRef as Leaf>::try_from_value(array_value.clone())
            .err()
            .unwrap(),
    );
    assert_same(
        &other_value,
        &<Array as Leaf>::try_from_value(other_value.clone())
            .err()
            .unwrap(),
    );
    assert_same(
        &other_cell,
        &<ArrayRef as Leaf>::try_from_value(other_cell.clone())
            .err()
            .unwrap(),
    );
    assert!(<Array as Leaf>::try_from_ref(&array_cell).is_err());
    assert!(<ArrayRef as Leaf>::try_from_ref(&array_value).is_err());
    assert!(<Array as Leaf>::try_from_ref(&other_value).is_err());
    assert!(<ArrayRef as Leaf>::try_from_ref(&other_cell).is_err());
    let original = array();
    assert!(
        original
            .replace(
                Of::<ObjectTracer>::filter(),
                array_cell.as_source(),
                &mut GraphContext::new()
            )
            .is_err()
    );
    let builder = original
        .builder::<ObjectTracer, _, SourceError>(
            Of::<ObjectTracer>::filter(),
            &mut GraphContext::new(),
        )
        .unwrap();
    assert!(
        Builder::<ObjectTracer>::build(builder, array_cell.as_source(), &mut GraphContext::new())
            .is_err()
    );
}

#[test]
fn selective_replacement_keeps_unmatched_cells_static() {
    let _guard = TraceContext::enter();
    let original = ObjectTracer::Cell(cell());
    let replacement = trace();
    let replaced = original
        .replace(
            Of::<Trace>::filter(),
            replacement.as_source(),
            &mut GraphContext::new(),
        )
        .unwrap();
    assert_same(&original, &replaced);
    let builder = original
        .builder::<Trace, _, SourceError>(Of::<Trace>::filter(), &mut GraphContext::new())
        .unwrap();
    let rebuilt = Builder::<Trace>::build(builder, NullSource, &mut GraphContext::new()).unwrap();
    assert_same(&original, &rebuilt);
    let builder = original
        .builder::<ObjectTracer, _, SourceError>(
            Nothing::<ObjectTracer>::filter(),
            &mut GraphContext::new(),
        )
        .unwrap();
    assert_same(
        &original,
        &Builder::<ObjectTracer>::build(builder, NullSource, &mut GraphContext::new()).unwrap(),
    );
}

#[test]
fn whole_object_replacement_can_change_the_erased_variant() {
    let _guard = TraceContext::enter();
    let original = ObjectTracer::Ref(trace());
    let replacement = ObjectTracer::Cell(cell());
    let result = original
        .replace(
            Of::<ObjectTracer>::filter(),
            replacement.as_source(),
            &mut GraphContext::new(),
        )
        .unwrap();
    assert_same(&result, &replacement);
}

#[test]
fn typed_filters_can_also_rebuild_an_erased_object() {
    let _guard = TraceContext::enter();
    let original = ObjectTracer::Ref(trace());
    let replacement = array();
    let result = original
        .replace(
            Of::<Array>::filter(),
            replacement.as_source(),
            &mut GraphContext::new(),
        )
        .unwrap();
    assert_same(&result, &collect(&replacement)[0]);
    let original = ObjectTracer::Cell(cell());
    let replacement = array_ref();
    let result = original
        .replace(
            Of::<ArrayRef>::filter(),
            replacement.as_source(),
            &mut GraphContext::new(),
        )
        .unwrap();
    assert_same(&result, &collect(&replacement)[0]);
}

#[test]
fn visiting_and_rebuilding_cells_records_no_reads_or_writes() {
    let guard = TraceContext::enter();
    let mutable = cell();
    let object = ObjectTracer::Cell(mutable.clone());
    let typed = <ArrayRef as Leaf>::try_from_value(object.clone()).unwrap();
    collect(&(object.clone(), typed.clone()));
    object
        .clone()
        .visit_into(Of::<ObjectTracer>::filter(), TakeLeaf)
        .unwrap();
    typed
        .clone()
        .visit_into(Of::<ObjectTracer>::filter(), TakeLeaf)
        .unwrap();
    let builder = typed
        .builder::<ObjectTracer, _, SourceError>(
            Of::<ObjectTracer>::filter(),
            &mut GraphContext::new(),
        )
        .unwrap();
    let rebuilt =
        Builder::<ObjectTracer>::build(builder, object.as_source(), &mut GraphContext::new())
            .unwrap();
    assert_same(&collect(&rebuilt)[0], &object);
    assert!(guard.context().last_aliased_write().is_none());
    // Any accidental reads would become ordering predecessors of this write.
    let assigned = trace();
    mutable.set(assigned.clone());
    let captured = Capture::from_context(guard.context(), &[object, assigned.into()], &[]).unwrap();
    assert_eq!(captured.expr().instrs().len(), 1);
    assert!(matches!(
        captured.expr().instrs().next().unwrap(),
        HInstr::WriteCell { .. }
    ));
}

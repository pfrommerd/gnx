use gnx_expr::{
    CallError, CellKey, CellTrace, Generic, HExpr, InputSpec, ObjectTracer, Ownership,
    TraceContext, Capture, Tracer, TracerKey, ValueInfo, hexpr,
};
fn info() -> ValueInfo {
    ValueInfo::new(0usize)
}
fn value() -> gnx_expr::Trace {
    Tracer::<Generic>::placeholder(info()).into_trace_ref()
}
fn cell() -> CellTrace {
    CellTrace::placeholder(info())
}
fn lower(body: &HExpr, closure: &[ObjectTracer], inputs: &[ObjectTracer]) -> gnx_expr::LoweredExpr {
    let closure = closure
        .iter()
        .cloned()
        .map(|v| (v, Ownership::Borrow))
        .collect::<Vec<_>>();
    let inputs = inputs
        .iter()
        .cloned()
        .map(|v| (v, Ownership::Borrow))
        .collect::<Vec<_>>();
    body.lower(&InputSpec::from_args(&closure, &inputs))
        .unwrap()
}
fn as_value(object: &ObjectTracer) -> &gnx_expr::Trace {
    let ObjectTracer::Ref(value) = object else {
        panic!("expected value")
    };
    value
}
fn as_cell(object: &ObjectTracer) -> &CellTrace {
    let ObjectTracer::Cell(cell) = object else {
        panic!("expected cell")
    };
    cell
}
#[test]
fn returning_only_an_input_cell_emits_no_read_or_write() {
    let body = hexpr!(info(); {
        lambda a[:]: in (a[:])
    });
    let guard = TraceContext::enter();
    let a = cell();
    let inputs = [ObjectTracer::Cell(a.clone())];
    let result = lower(&body, &[], &inputs).call(&[], &inputs).unwrap();
    assert_eq!(CellKey::from(as_cell(&result[0])), CellKey::from(&a));
    let caller = Capture::from_context(guard.context(), &inputs, &result).unwrap();
    assert!(caller.expr().instructions().is_empty());
}
#[test]
fn snapshots_precede_all_updates_and_new_cells_keep_the_old_state() {
    let body = hexpr!(info(); {
        lambda a[:], b[:]:
        old_a <- a[:];
        old_b <- b[:];
        fresh[:] = old_a;
        a[:] <- old_b;
        b[:] <- old_a;
        in (fresh[:], fresh[:])
    });
    let a_value = value();
    let b_value = value();
    let a = CellTrace::new(a_value.clone());
    let b = CellTrace::new(b_value.clone());
    let inputs = [ObjectTracer::Cell(a.clone()), ObjectTracer::Cell(b.clone())];
    let lowered = lower(&body, &[], &inputs);
    let result = lowered.call(&[], &inputs).unwrap();
    assert_eq!(
        result.len(),
        2,
        "update-source descriptors must not escape as results"
    );
    assert_eq!(TracerKey::from(&a.get()), TracerKey::from(&b_value));
    assert_eq!(TracerKey::from(&b.get()), TracerKey::from(&a_value));
    assert_eq!(
        TracerKey::from(&as_cell(&result[0]).get()),
        TracerKey::from(&a_value)
    );
    assert_eq!(
        CellKey::from(as_cell(&result[0])),
        CellKey::from(as_cell(&result[1]))
    );
}
#[test]
fn boundary_reuses_stored_sources_with_new_handles_of_the_same_specialization() {
    let body = hexpr!(info(); {
        lambda a[:], b[:]:
        old_b <- b[:];
        a[:] <- old_b;
        in ()
    });
    let a = CellTrace::new(value());
    let b = CellTrace::new(value());
    let distinct = [ObjectTracer::Cell(a.clone()), ObjectTracer::Cell(b)];
    let alias = [ObjectTracer::Cell(a.clone()), ObjectTracer::Cell(a)];
    let distinct_body = lower(&body, &[], &distinct);
    let aliased_body = lower(&body, &[], &alias);
    let new_a = CellTrace::new(value());
    let new_value = value();
    let new_b = CellTrace::new(new_value.clone());
    distinct_body
        .call(&[], &[new_a.clone().into(), new_b.into()])
        .unwrap();
    assert_eq!(TracerKey::from(&new_a.get()), TracerKey::from(&new_value));
    aliased_body
        .call(&[], &[new_a.clone().into(), new_a.clone().into()])
        .unwrap();
    assert_eq!(TracerKey::from(&new_a.get()), TracerKey::from(&new_value));
}
#[test]
fn repeated_value_results_are_forwarded_without_owned_call_returns() {
    let body = hexpr!(info(); {
        lambda x: in (x, x)
    });
    let x = value();
    let inputs = [ObjectTracer::Ref(x.clone())];
    let lowered = lower(&body, &[], &inputs);
    assert!(lowered.expr.outputs().is_empty());
    let results = lowered.call(&[], &inputs).unwrap();
    for result in &results {
        assert_eq!(TracerKey::from(as_value(result)), TracerKey::from(&x));
    }
}
#[test]
fn boundary_validates_variants_and_metadata_before_updates() {
    let body = hexpr!(info(); {
        lambda a[:], x:
        a[:] <- x;
        in ()
    });
    let a = CellTrace::new(value());
    let x = value();
    let inputs = [ObjectTracer::Cell(a.clone()), ObjectTracer::Ref(x.clone())];
    let lowered = lower(&body, &[], &inputs);
    assert!(matches!(
        lowered.call(&[], &[x.clone().into(), a.clone().into()]),
        Err(CallError::ArgumentType(_))
    ));
    let wrong = gnx_expr::Trace::placeholder(ValueInfo::new(1usize));
    assert!(matches!(
        lowered.call(&[], &[a.into(), wrong.into()]),
        Err(CallError::ArgumentType(_))
    ));
    assert!(matches!(lowered.call(&[], &[]), Err(CallError::Arity)));
}

use gnx_expr::{
    Capture, CellTrace, Dialect, Generic, HExpr, InputId, InputSpec, Invocation, Kind,
    LoweringError, Op, Operation, TraceContext, Tracer, ValueInfo, lower,
};

fn distinct_parameters(expr: &HExpr) -> InputSpec {
    let bind = |params: &[(gnx_expr::Var, Kind, ValueInfo)], source: fn(usize) -> InputId| {
        params
            .iter()
            .enumerate()
            .map(|(index, param)| (source(index), param.1, param.2.clone()))
            .collect()
    };
    InputSpec {
        closure: bind(expr.closure(), InputId::Closure),
        inputs: bind(expr.inputs(), InputId::Input),
    }
}
struct TestDialect;
#[derive(Clone, PartialEq, Eq, Hash)]
struct Compute;
static DIALECT: TestDialect = TestDialect;
impl Dialect for TestDialect {
    fn name(&self) -> &'static str {
        "boundary_test"
    }
}
impl std::fmt::Display for Compute {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}::{}",
            Operation::dialect(self).name(),
            Operation::name(self)
        )
    }
}

impl Operation for Compute {
    fn name(&self) -> &'static str {
        "compute"
    }
    fn dialect(&self) -> &'static dyn Dialect {
        &DIALECT
    }
}
fn info() -> ValueInfo {
    ValueInfo::new(0usize)
}
fn value() -> Tracer<Generic> {
    Tracer::placeholder(info())
}
fn compute(inputs: Vec<Tracer<Generic>>) -> Tracer<Generic> {
    Invocation::invoke(Op::new(Compute), vec![], inputs, vec![info()])
        .pop()
        .unwrap()
}

#[test]
fn boundary_results_reconstruct_values_and_both_kinds_of_cells() {
    let guard = TraceContext::enter();
    let borrowed = value();
    let moved = value();
    let input_cell = CellTrace::placeholder(info());
    let computed = compute(vec![moved.clone()]);
    let from_input = CellTrace::new(borrowed.trace_ref().clone());
    let from_output = CellTrace::new(computed.trace_ref().clone());
    let captured = Capture::from_context(
        guard.context(),
        &[
            borrowed.clone().into(),
            moved.into(),
            input_cell.clone().into(),
        ],
        &[
            computed.into(),
            borrowed.into(),
            input_cell.into(),
            from_input.into(),
            from_output.into(),
        ],
    )
    .unwrap();
    let spec = InputSpec {
        closure: vec![],
        inputs: vec![
            (InputId::Input(10), Kind::Value, info()),
            (InputId::Input(20), Kind::Value, info()),
            (InputId::Input(30), Kind::Cell, info()),
        ],
    };
    let lowered = captured.expr().lower(&spec).unwrap();
    assert_eq!(
        lowered.inputs,
        vec![InputId::Input(10), InputId::Input(20), InputId::Input(30)]
    );
    assert_eq!(
        lowered.results(),
        &[
            lower::Output::Value { result_idx: 0 },
            lower::Output::Value { result_idx: 1 },
            lower::Output::InputCell {
                input_id: InputId::Input(30)
            },
            lower::Output::NewCell {
                id: 0,
                result_idx: 2
            },
            lower::Output::NewCell {
                id: 1,
                result_idx: 3
            },
        ]
    );
    let computed = lowered.expr.eqns()[0].outputs()[0].0;
    let input = lowered.expr.inputs()[0].0;
    assert_eq!(lowered.expr.outputs(), &[computed, input, input, computed]);
    assert_eq!(lowered.trace_body().outputs(), &[computed]);
    assert!(lowered.updates.is_empty());
}

#[test]
fn cell_updates_index_immutable_results() {
    let guard = TraceContext::enter();
    let cell = CellTrace::placeholder(info());
    let x = value();
    cell.set(x.trace_ref().clone());
    let captured = Capture::from_context(
        guard.context(),
        &[cell.into(), x.clone().into()],
        &[x.clone().into(), x.into()],
    )
    .unwrap();
    let spec = InputSpec {
        closure: vec![],
        inputs: vec![
            (InputId::Input(7), Kind::Cell, info()),
            (InputId::Input(91), Kind::Value, info()),
        ],
    };
    let lowered = captured.expr().lower(&spec).unwrap();
    assert_eq!(lowered.expr.outputs(), &[lowered.expr.inputs()[1].0; 3]);
    assert_eq!(lowered.results().len(), 2);
    assert_eq!(lowered.updates, vec![(InputId::Input(7), 2)]);
    assert!(lowered.trace_body().outputs().is_empty());
}

#[test]
fn update_only_sources_are_not_exposed_as_user_results() {
    let guard = TraceContext::enter();
    let cell = CellTrace::placeholder(info());
    let x = value();
    cell.set(compute(vec![x.clone()]).into_trace_ref());
    let captured = Capture::from_context(guard.context(), &[cell.into(), x.into()], &[]).unwrap();
    let spec = InputSpec {
        closure: vec![],
        inputs: vec![
            (InputId::Input(7), Kind::Cell, info()),
            (InputId::Input(9), Kind::Value, info()),
        ],
    };
    let lowered = captured.expr().lower(&spec).unwrap();
    assert!(lowered.results().is_empty());
    assert_eq!(lowered.updates, vec![(InputId::Input(7), 0)]);
    assert_eq!(
        lowered.expr.outputs(),
        &[lowered.expr.eqns()[0].outputs()[0].0]
    );
}

#[test]
fn new_input_cell_uses_the_original_input_state_snapshot() {
    let guard = TraceContext::enter();
    let cell = CellTrace::placeholder(info());
    let x = value();
    let fresh = CellTrace::new(cell.get());
    cell.set(x.trace_ref().clone());
    let captured =
        Capture::from_context(guard.context(), &[cell.into(), x.into()], &[fresh.into()]).unwrap();
    let spec = InputSpec {
        closure: vec![],
        inputs: vec![
            (InputId::Input(7), Kind::Cell, info()),
            (InputId::Input(9), Kind::Value, info()),
        ],
    };
    let lowered = captured.expr().lower(&spec).unwrap();
    assert_eq!(
        lowered.results(),
        &[lower::Output::NewCell {
            id: 0,
            result_idx: 0
        }]
    );
    assert_eq!(
        lowered.expr.outputs(),
        &[lowered.expr.inputs()[0].0, lowered.expr.inputs()[1].0]
    );
    assert_eq!(lowered.updates, vec![(InputId::Input(7), 1)]);
}

#[test]
fn distinct_new_cells_can_share_an_initializer_without_sharing_identity() {
    let guard = TraceContext::enter();
    let x = value();
    let computed = compute(vec![x.clone()]);
    let a = CellTrace::new(computed.trace_ref().clone());
    let b = CellTrace::new(computed.into_trace_ref());
    let captured =
        Capture::from_context(guard.context(), &[x.into()], &[a.into(), b.into()]).unwrap();
    let lowered = captured
        .expr()
        .lower(&distinct_parameters(captured.expr()))
        .unwrap();
    let computed = lowered.expr.eqns()[0].outputs()[0].0;
    assert_eq!(lowered.expr.outputs(), &[computed; 2]);
    assert_eq!(lowered.trace_body().outputs(), &[computed]);
    assert_eq!(
        lowered.results(),
        &[
            lower::Output::NewCell {
                id: 0,
                result_idx: 0
            },
            lower::Output::NewCell {
                id: 1,
                result_idx: 1
            },
        ]
    );
}

#[test]
fn one_id_cannot_identify_both_a_value_and_a_cell() {
    let guard = TraceContext::enter();
    let x = value();
    let cell = CellTrace::placeholder(info());
    let captured = Capture::from_context(guard.context(), &[x.into(), cell.into()], &[]).unwrap();
    let spec = InputSpec {
        closure: vec![],
        inputs: vec![
            (InputId::Input(0), Kind::Value, info()),
            (InputId::Input(0), Kind::Cell, info()),
        ],
    };
    assert!(matches!(
        captured.expr().lower(&spec),
        Err(LoweringError::AliasedInputKindMismatch(InputId::Input(0)))
    ));
}

#[test]
fn cell_state_and_user_result_share_one_computation() {
    let guard = TraceContext::enter();
    let cell = CellTrace::placeholder(info());
    let x = value();
    cell.set(compute(vec![x.clone()]).into_trace_ref());
    let read = cell.get();
    let captured =
        Capture::from_context(guard.context(), &[cell.into(), x.into()], &[read.into()]).unwrap();
    let spec = InputSpec {
        closure: vec![],
        inputs: vec![
            (InputId::Input(0), Kind::Cell, info()),
            (InputId::Input(1), Kind::Value, info()),
        ],
    };
    let lowered = captured.expr().lower(&spec).unwrap();
    let computed = lowered.expr.eqns()[0].outputs()[0].0;
    assert_eq!(lowered.expr.outputs(), &[computed; 2]);
    assert_eq!(lowered.trace_body().outputs(), &[computed]);
    assert_eq!(lowered.results(), &[lower::Output::Value { result_idx: 0 }]);
    assert_eq!(lowered.updates, vec![(InputId::Input(0), 1)]);
}

#[test]
fn captured_values_become_region_parameters() {
    let _outer = TraceContext::enter();
    let x = value();
    let inner = TraceContext::enter();
    let y = compute(vec![x]);
    let captured = Capture::from_context(inner.context(), &[], &[y.into()]).unwrap();
    let lowered = captured
        .expr()
        .lower(&distinct_parameters(captured.expr()))
        .unwrap();
    assert_eq!(lowered.closure, vec![InputId::Closure(0)]);
    assert!(lowered.inputs.is_empty());
    assert_eq!(
        lowered.expr.eqns()[0].inputs(),
        &[lowered.expr.closure()[0].0]
    );
}

#[test]
fn input_spec_validates_kinds_metadata_and_arity() {
    let guard = TraceContext::enter();
    let x = value();
    let captured = Capture::from_context(guard.context(), &[x.into()], &[]).unwrap();
    for (kind, metadata) in [(Kind::Cell, info()), (Kind::Value, ValueInfo::new(1usize))] {
        let spec = InputSpec {
            closure: vec![],
            inputs: vec![(InputId::Input(0), kind, metadata)],
        };
        assert_eq!(
            captured.expr().lower(&spec).unwrap_err(),
            LoweringError::KindMismatch {
                closure: false,
                index: 0
            }
        );
    }
    assert!(matches!(
        captured.expr().lower(&InputSpec::default()),
        Err(LoweringError::InputSpecArity { .. })
    ));
}

#[test]
fn input_spec_validates_capture_kinds_and_arity() {
    let _outer = TraceContext::enter();
    let x = value();
    let inner = TraceContext::enter();
    let out = compute(vec![x]);
    let captured = Capture::from_context(inner.context(), &[], &[out.into()]).unwrap();
    let spec = InputSpec {
        closure: vec![(InputId::Input(0), Kind::Cell, info())],
        inputs: vec![],
    };
    assert_eq!(
        captured.expr().lower(&spec).unwrap_err(),
        LoweringError::KindMismatch {
            closure: true,
            index: 0
        }
    );
    assert!(matches!(
        captured.expr().lower(&InputSpec::default()),
        Err(LoweringError::InputSpecArity { closure: true, .. })
    ));
}

#[test]
fn input_spec_from_args_shares_value_and_cell_identities_across_both_lists() {
    let _guard = TraceContext::enter();
    let x = value();
    let cell = CellTrace::placeholder(info());
    let spec = InputSpec::from_args(
        &[x.clone().into(), cell.clone().into()],
        &[cell.into(), x.into()],
    );
    assert_eq!(
        spec.closure,
        vec![
            (InputId::Input(1), Kind::Value, info()),
            (InputId::Input(0), Kind::Cell, info()),
        ]
    );
    assert_eq!(
        spec.inputs,
        vec![
            (InputId::Input(0), Kind::Cell, info()),
            (InputId::Input(1), Kind::Value, info()),
        ]
    );
}

#[test]
fn a_call_that_only_forwards_an_input_needs_no_equation() {
    let body = gnx_expr::hexpr!(gnx_expr::ValueInfo::new(0usize); { lambda formal: in (formal) });
    let guard = TraceContext::enter();
    let x = value();
    let result = gnx_transforms::cf::call_immutable(
        std::sync::Arc::new(body),
        Vec::<Tracer<Generic>>::new(),
        [x.clone()],
        vec![info()],
    );
    let captured = Capture::from_context(
        guard.context(),
        &[x.into()],
        &result.into_iter().map(Into::into).collect::<Vec<_>>(),
    )
    .unwrap();
    let lowered = captured
        .expr()
        .lower(&distinct_parameters(captured.expr()))
        .unwrap();
    assert!(lowered.expr.eqns().is_empty());
    assert_eq!(lowered.expr.outputs(), &[lowered.expr.inputs()[0].0]);
}

#[test]
fn repeated_results_can_forward_the_same_input() {
    let guard = TraceContext::enter();
    let x = value();
    let captured = Capture::from_context(
        guard.context(),
        &[x.clone().into()],
        &[x.clone().into(), x.into()],
    )
    .unwrap();
    let lowered = captured
        .expr()
        .lower(&distinct_parameters(captured.expr()))
        .unwrap();
    assert_eq!(lowered.expr.outputs(), &[lowered.expr.inputs()[0].0; 2]);
    assert!(lowered.trace_body().outputs().is_empty());
}

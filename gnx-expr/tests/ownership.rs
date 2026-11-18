use gnx_expr::{
    Capture, CellTrace, Dialect, Generic, HExpr, InputId, InputSpec, Invocation, Kind,
    LoweringError, Op, Operation, Ownership, TraceContext, Tracer, ValueInfo, lower,
};

fn all_values(expr: &HExpr, ownership: Ownership) -> InputSpec {
    let bind = |params: &[(gnx_expr::Var, Kind)], source: fn(usize) -> InputId| {
        params
            .iter()
            .enumerate()
            .map(|(index, param)| (source(index), param.1.clone(), ownership))
            .collect()
    };
    InputSpec {
        closure: bind(expr.closure(), InputId::Closure),
        inputs: bind(expr.inputs(), InputId::Input),
    }
}
struct TestDialect;
struct Compute;
static DIALECT: TestDialect = TestDialect;
static COMPUTE: Compute = Compute;
impl Dialect for TestDialect {
    fn name(&self) -> &'static str {
        "ownership_test"
    }
    fn create(&self, _: &'static str) -> Option<&'static dyn Operation> {
        Some(&COMPUTE)
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
    Invocation::invoke(Op::new(&COMPUTE), vec![], inputs, vec![info()])
        .pop()
        .unwrap()
}

#[test]
fn boundary_descriptors_cover_all_five_output_kinds() {
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
            (InputId::Input(10), Kind::Value(info()), Ownership::Borrow),
            (InputId::Input(20), Kind::Value(info()), Ownership::Move),
            (InputId::Input(30), Kind::Cell(info()), Ownership::Borrow),
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
            lower::Output::OutputValue { output_idx: 0 },
            lower::Output::InputValue {
                input_id: InputId::Input(10)
            },
            lower::Output::InputCell {
                input_id: InputId::Input(30)
            },
            lower::Output::NewInputCell {
                id: 0,
                input_id: InputId::Input(10)
            },
            lower::Output::NewOutputCell {
                id: 1,
                output_idx: 0
            },
        ]
    );
    assert_eq!(lowered.expr.outputs().len(), 1);
    assert!(lowered.updates.is_empty());
}

#[test]
fn update_indices_address_descriptors_not_owned_return_slots() {
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
            (InputId::Input(7), Kind::Cell(info()), Ownership::Borrow),
            (InputId::Input(91), Kind::Value(info()), Ownership::Borrow),
        ],
    };
    let lowered = captured.expr().lower(&spec).unwrap();
    assert!(lowered.expr.outputs().is_empty());
    assert_eq!(lowered.results().len(), 2);
    assert_eq!(lowered.updates, vec![(InputId::Input(7), 2)]);
    assert_eq!(
        lowered.outputs,
        vec![
            lower::Output::InputValue {
                input_id: InputId::Input(91)
            };
            3
        ]
    );
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
            (InputId::Input(7), Kind::Cell(info()), Ownership::Borrow),
            (InputId::Input(9), Kind::Value(info()), Ownership::Move),
        ],
    };
    let lowered = captured.expr().lower(&spec).unwrap();
    assert!(lowered.results().is_empty());
    assert_eq!(
        lowered.outputs,
        vec![lower::Output::OutputValue { output_idx: 0 }]
    );
    assert_eq!(lowered.updates, vec![(InputId::Input(7), 0)]);
    assert_eq!(lowered.expr.outputs().len(), 1);
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
            (InputId::Input(7), Kind::Cell(info()), Ownership::Borrow),
            (InputId::Input(9), Kind::Value(info()), Ownership::Borrow),
        ],
    };
    let lowered = captured.expr().lower(&spec).unwrap();
    assert_eq!(
        lowered.results(),
        &[lower::Output::NewInputCell {
            id: 0,
            input_id: InputId::Input(7)
        }]
    );
    assert_eq!(
        lowered.outputs[lowered.updates[0].1],
        lower::Output::InputValue {
            input_id: InputId::Input(9)
        }
    );
    assert!(lowered.expr.outputs().is_empty());
}

#[test]
fn distinct_new_cells_can_share_one_owned_initializer_without_sharing_identity() {
    let guard = TraceContext::enter();
    let x = value();
    let computed = compute(vec![x.clone()]);
    let a = CellTrace::new(computed.trace_ref().clone());
    let b = CellTrace::new(computed.into_trace_ref());
    let captured =
        Capture::from_context(guard.context(), &[x.into()], &[a.into(), b.into()]).unwrap();
    let lowered = captured
        .expr()
        .lower(&all_values(captured.expr(), Ownership::Borrow))
        .unwrap();
    assert_eq!(lowered.expr.outputs().len(), 1);
    assert_eq!(
        lowered.results(),
        &[
            lower::Output::NewOutputCell {
                id: 0,
                output_idx: 0
            },
            lower::Output::NewOutputCell {
                id: 1,
                output_idx: 0
            },
        ]
    );
}

#[test]
fn borrowed_input_is_never_moved_but_computed_result_is_owned() {
    let guard = TraceContext::enter();
    let x = value();
    let y = compute(vec![x.clone()]);
    let captured = Capture::from_context(guard.context(), &[x.into()], &[y.into()]).unwrap();
    let lowered = captured
        .expr()
        .lower(&all_values(captured.expr(), Ownership::Borrow))
        .unwrap();
    assert_eq!(lowered.expr.inputs()[0].1, Ownership::Borrow);
    assert_eq!(lowered.expr.eqns()[0].inputs()[0].1, Ownership::Borrow);
    assert_eq!(
        lowered.expr.outputs(),
        &[lowered.expr.eqns()[0].outputs()[0].0]
    );
    lowered.expr.check().unwrap();
}

#[test]
fn moved_input_is_borrowed_until_its_last_use() {
    let guard = TraceContext::enter();
    let x = value();
    let a = compute(vec![x.clone()]);
    let b = compute(vec![x.clone()]);
    let captured =
        Capture::from_context(guard.context(), &[x.into()], &[a.into(), b.into()]).unwrap();
    let lowered = captured
        .expr()
        .lower(&all_values(captured.expr(), Ownership::Move))
        .unwrap();
    assert_eq!(lowered.expr.eqns()[0].inputs()[0].1, Ownership::Borrow);
    assert_eq!(lowered.expr.eqns()[1].inputs()[0].1, Ownership::Move);
    lowered.expr.check().unwrap();
}

#[test]
fn ownership_is_chosen_independently_per_input() {
    let guard = TraceContext::enter();
    let x = value();
    let y = value();
    let out = compute(vec![x.clone(), y.clone()]);
    let captured =
        Capture::from_context(guard.context(), &[x.into(), y.into()], &[out.into()]).unwrap();
    let spec = InputSpec {
        closure: vec![],
        inputs: vec![
            (InputId::Input(0), Kind::Value(info()), Ownership::Borrow),
            (InputId::Input(1), Kind::Value(info()), Ownership::Move),
        ],
    };
    let lowered = captured.expr().lower(&spec).unwrap();
    assert_eq!(lowered.expr.eqns()[0].inputs()[0].1, Ownership::Borrow);
    assert_eq!(lowered.expr.eqns()[0].inputs()[1].1, Ownership::Move);
}

#[test]
fn intermediate_values_can_be_moved_even_when_inputs_are_borrowed() {
    let guard = TraceContext::enter();
    let x = value();
    let middle = compute(vec![x.clone()]);
    let out = compute(vec![middle]);
    let captured = Capture::from_context(guard.context(), &[x.into()], &[out.into()]).unwrap();
    let lowered = captured
        .expr()
        .lower(&all_values(captured.expr(), Ownership::Borrow))
        .unwrap();
    assert_eq!(lowered.expr.eqns()[0].inputs()[0].1, Ownership::Borrow);
    assert_eq!(lowered.expr.eqns()[1].inputs()[0].1, Ownership::Move);
}

#[test]
fn returning_a_moved_input_delays_its_move_until_return() {
    let guard = TraceContext::enter();
    let x = value();
    let y = compute(vec![x.clone()]);
    let captured =
        Capture::from_context(guard.context(), &[x.clone().into()], &[x.into(), y.into()]).unwrap();
    let lowered = captured
        .expr()
        .lower(&all_values(captured.expr(), Ownership::Move))
        .unwrap();
    assert_eq!(lowered.expr.eqns()[0].inputs()[0].1, Ownership::Borrow);
    assert_eq!(lowered.expr.outputs()[0], lowered.expr.inputs()[0].0);
}

#[test]
fn borrowed_input_passthrough_is_removed_from_low_level_outputs() {
    let guard = TraceContext::enter();
    let x = value();
    let captured =
        Capture::from_context(guard.context(), &[x.clone().into()], &[x.into()]).unwrap();
    let lowered = captured
        .expr()
        .lower(&all_values(captured.expr(), Ownership::Borrow))
        .unwrap();
    assert!(lowered.expr.outputs().is_empty());
    assert!(
        lowered.expr.eqns().is_empty(),
        "no implicit clone operation"
    );
    assert_eq!(
        lowered.results(),
        vec![lower::Output::InputValue {
            input_id: InputId::Input(0)
        }]
    );
    lowered.expr.check().unwrap();
}

#[test]
fn repeated_operand_is_moved_only_once() {
    let guard = TraceContext::enter();
    let x = value();
    let y = compute(vec![x.clone(), x.clone()]);
    let captured = Capture::from_context(guard.context(), &[x.into()], &[y.into()]).unwrap();
    let lowered = captured
        .expr()
        .lower(&all_values(captured.expr(), Ownership::Move))
        .unwrap();
    assert_eq!(lowered.expr.eqns()[0].inputs()[0].1, Ownership::Borrow);
    assert_eq!(lowered.expr.eqns()[0].inputs()[1].1, Ownership::Move);
}

#[test]
fn the_same_value_id_can_be_moved_into_two_different_arguments() {
    let guard = TraceContext::enter();
    let x = value();
    let y = value();
    let out = compute(vec![x.clone(), y.clone()]);
    let captured =
        Capture::from_context(guard.context(), &[x.into(), y.into()], &[out.into()]).unwrap();
    let binding = (InputId::Input(8), Kind::Value(info()), Ownership::Move);
    let lowered = captured
        .expr()
        .lower(&InputSpec {
            closure: vec![],
            inputs: vec![binding.clone(), binding],
        })
        .unwrap();
    assert_eq!(lowered.inputs, vec![InputId::Input(8)]);
    assert_eq!(lowered.expr.inputs().len(), 1);
    let uses = lowered.expr.eqns()[0].inputs();
    assert_eq!(uses[0].0, uses[1].0);
    assert_eq!(uses[0].1, Ownership::Borrow);
    assert_eq!(uses[1].1, Ownership::Move);
    lowered.expr.check().unwrap();
}

#[test]
fn value_identity_is_shared_across_captures_and_inputs() {
    let _outer = TraceContext::enter();
    let captured_value = value();
    let inner = TraceContext::enter();
    let argument = value();
    let out = compute(vec![captured_value, argument.clone()]);
    let captured =
        Capture::from_context(inner.context(), &[argument.into()], &[out.into()]).unwrap();
    let binding = (InputId::Input(0), Kind::Value(info()), Ownership::Move);
    let lowered = captured
        .expr()
        .lower(&InputSpec {
            closure: vec![binding.clone()],
            inputs: vec![binding],
        })
        .unwrap();
    assert_eq!(lowered.expr.inputs().len(), 1);
    assert_eq!(
        lowered.expr.eqns()[0].inputs()[0].0,
        lowered.expr.eqns()[0].inputs()[1].0
    );
    assert_eq!(lowered.expr.eqns()[0].inputs()[1].1, Ownership::Move);
}

#[test]
fn aliases_cannot_disagree_on_ownership() {
    let guard = TraceContext::enter();
    for is_cell in [false, true] {
        let params: Vec<gnx_expr::ObjectTracer> = if is_cell {
            vec![
                CellTrace::placeholder(info()).into(),
                CellTrace::placeholder(info()).into(),
            ]
        } else {
            vec![value().into(), value().into()]
        };
        let captured = Capture::from_context(guard.context(), &params, &[]).unwrap();
        let kind = if is_cell {
            Kind::Cell(info())
        } else {
            Kind::Value(info())
        };
        let spec = InputSpec {
            closure: vec![],
            inputs: vec![
                (InputId::Input(0), kind.clone(), Ownership::Borrow),
                (InputId::Input(0), kind, Ownership::Move),
            ],
        };
        assert!(matches!(
            captured.expr().lower(&spec),
            Err(LoweringError::AliasedInputOwnershipMismatch(
                InputId::Input(0)
            ))
        ));
    }
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
            (InputId::Input(0), Kind::Value(info()), Ownership::Borrow),
            (InputId::Input(0), Kind::Cell(info()), Ownership::Borrow),
        ],
    };
    assert!(matches!(
        captured.expr().lower(&spec),
        Err(LoweringError::AliasedInputKindMismatch(InputId::Input(0)))
    ));
}

#[test]
fn cell_state_and_user_result_share_one_owned_output() {
    let guard = TraceContext::enter();
    let cell = CellTrace::placeholder(info());
    let x = value();
    cell.set(x.trace_ref().clone());
    let read = cell.get();
    let captured =
        Capture::from_context(guard.context(), &[cell.into(), x.into()], &[read.into()]).unwrap();
    let spec = InputSpec {
        closure: vec![],
        inputs: vec![
            (InputId::Input(0), Kind::Cell(info()), Ownership::Move),
            (InputId::Input(1), Kind::Value(info()), Ownership::Move),
        ],
    };
    let lowered = captured.expr().lower(&spec).unwrap();
    assert_eq!(lowered.expr.outputs(), &[lowered.expr.inputs()[1].0]);
    assert_eq!(
        lowered.outputs[lowered.updates[0].1],
        lower::Output::OutputValue { output_idx: 0 }
    );
    assert_eq!(
        lowered.results()[0],
        lower::Output::OutputValue { output_idx: 0 }
    );
    assert_eq!(lowered.results().len(), 1);
    assert_eq!(lowered.outputs.len(), 2);
}

#[test]
fn borrowed_cell_state_and_value_passthroughs_never_become_returns() {
    let guard = TraceContext::enter();
    let cell = CellTrace::placeholder(info());
    let x = value();
    cell.set(x.trace_ref().clone());
    let read = cell.get();
    let captured =
        Capture::from_context(guard.context(), &[cell.into(), x.into()], &[read.into()]).unwrap();
    let spec = InputSpec {
        closure: vec![],
        inputs: vec![
            (InputId::Input(0), Kind::Cell(info()), Ownership::Borrow),
            (InputId::Input(1), Kind::Value(info()), Ownership::Borrow),
        ],
    };
    let lowered = captured.expr().lower(&spec).unwrap();
    assert!(lowered.expr.outputs().is_empty());
    assert_eq!(
        lowered.outputs[lowered.updates[0].1],
        lower::Output::InputValue {
            input_id: InputId::Input(1)
        }
    );
    assert_eq!(
        lowered.results(),
        vec![lower::Output::InputValue {
            input_id: InputId::Input(1)
        }]
    );
}

#[test]
fn captured_value_ownership_is_supplied_by_lowering_too() {
    let _outer = TraceContext::enter();
    let x = value();
    let inner = TraceContext::enter();
    let y = compute(vec![x]);
    let captured = Capture::from_context(inner.context(), &[], &[y.into()]).unwrap();
    let lowered = captured
        .expr()
        .lower(&all_values(captured.expr(), Ownership::Move))
        .unwrap();
    assert_eq!(lowered.closure, vec![InputId::Closure(0)]);
    assert!(lowered.inputs.is_empty());
    assert_eq!(lowered.expr.closure()[0].1, Ownership::Move);
    assert_eq!(lowered.expr.eqns()[0].inputs()[0].1, Ownership::Move);
}

#[test]
fn input_spec_validates_kinds_metadata_and_arity() {
    let guard = TraceContext::enter();
    let x = value();
    let captured = Capture::from_context(guard.context(), &[x.into()], &[]).unwrap();
    for kind in [Kind::Cell(info()), Kind::Value(ValueInfo::new(1usize))] {
        let spec = InputSpec {
            closure: vec![],
            inputs: vec![(InputId::Input(0), kind, Ownership::Borrow)],
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
        closure: vec![(InputId::Input(0), Kind::Cell(info()), Ownership::Borrow)],
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
        &[
            (x.clone().into(), Ownership::Move),
            (cell.clone().into(), Ownership::Borrow),
        ],
        &[
            (cell.into(), Ownership::Borrow),
            (x.into(), Ownership::Move),
        ],
    );
    assert_eq!(
        spec.closure,
        vec![
            (InputId::Input(1), Kind::Value(info()), Ownership::Move),
            (InputId::Input(0), Kind::Cell(info()), Ownership::Borrow),
        ]
    );
    assert_eq!(
        spec.inputs,
        vec![
            (InputId::Input(0), Kind::Cell(info()), Ownership::Borrow),
            (InputId::Input(1), Kind::Value(info()), Ownership::Move),
        ]
    );
}

#[test]
fn high_level_call_infers_ownership_from_the_outer_boundary() {
    let body = gnx_expr::hexpr!(gnx_expr::ValueInfo::new(0usize); { lambda formal: in (formal) });
    let guard = TraceContext::enter();
    let x = value();
    let result = gnx_expr::builtins::call(
        std::sync::Arc::new(body),
        Vec::<Tracer<Generic>>::new(),
        [x.clone()],
        vec![info()],
    );
    let captured = Capture::from_context(guard.context(), &[x.into()], &result).unwrap();
    let lowered = captured
        .expr()
        .lower(&all_values(captured.expr(), Ownership::Borrow))
        .unwrap();
    assert_eq!(
        lowered.results(),
        &[lower::Output::InputValue {
            input_id: InputId::Input(0)
        }]
    );
}

#[test]
fn repeated_borrowed_results_are_all_forwarded_without_low_level_returns() {
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
        .lower(&all_values(captured.expr(), Ownership::Borrow))
        .unwrap();
    assert!(lowered.expr.outputs().is_empty());
    assert_eq!(
        lowered.results(),
        vec![
            lower::Output::InputValue {
                input_id: InputId::Input(0)
            };
            2
        ]
    );
}

use gnx_expr::{
    Capture, CaptureError, CellAccess, CellTrace, Dialect, Generic, HInstr, InputId, InputSpec,
    Invocation, Kind, LoweringError, Op, Operation, TraceContext, Tracer, TracerKey, VarInfo,
    lower,
};

// Test shorthand for specifying only aliases while borrowing every argument.
#[derive(Default)]
struct CellAliases {
    closure: Vec<InputId>,
    inputs: Vec<InputId>,
}

trait AliasLowering {
    fn lower_with_aliases(
        &self,
        bindings: &CellAliases,
    ) -> Result<gnx_expr::LoweredExpr, LoweringError>;
}
impl AliasLowering for gnx_expr::HExpr {
    fn lower_with_aliases(
        &self,
        bindings: &CellAliases,
    ) -> Result<gnx_expr::LoweredExpr, LoweringError> {
        let mut next = bindings
            .closure
            .iter()
            .chain(&bindings.inputs)
            .map(|id| match id {
                InputId::Closure(index) | InputId::Input(index) => *index,
            })
            .max()
            .map_or(0, |id| id + 1);
        let mut bind = |params: &[(gnx_expr::Var, Kind, VarInfo)], ids: &[InputId]| {
            let mut ids = ids.iter();
            params
                .iter()
                .map(|param| {
                    let id = match &param.1 {
                        Kind::Value => {
                            let id = InputId::Input(next);
                            next += 1;
                            id
                        }
                        Kind::Cell => *ids.next().expect("test cell alias missing"),
                    };
                    (id, param.1, param.2.clone())
                })
                .collect()
        };
        self.lower(&InputSpec {
            closure: bind(self.closure(), &bindings.closure),
            inputs: bind(self.inputs(), &bindings.inputs),
        })
    }
}

fn info() -> VarInfo {
    VarInfo::new(0usize)
}
fn value() -> Tracer<Generic> {
    Tracer::placeholder(info())
}
fn cell() -> CellTrace {
    CellTrace::placeholder(info())
}

struct TestDialect;
#[derive(Clone, PartialEq, Eq, Hash)]
struct Neg;
static DIALECT: TestDialect = TestDialect;
impl Dialect for TestDialect {
    fn name(&self) -> &'static str {
        "test"
    }
}
impl std::fmt::Display for Neg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}::{}",
            Operation::dialect(self).name(),
            Operation::name(self)
        )
    }
}

impl Operation for Neg {
    fn name(&self) -> &'static str {
        "neg"
    }
    fn dialect(&self) -> &'static dyn Dialect {
        &DIALECT
    }
}
mod ops {
    pub mod test {
        pub fn neg() -> gnx_expr::Op {
            gnx_expr::Op::new(super::super::Neg)
        }
    }
}

fn neg(x: Tracer<Generic>) -> Tracer<Generic> {
    Invocation::invoke(Op::new(Neg), vec![], vec![x], vec![info()])
        .pop()
        .unwrap()
}
#[test]
fn captured_equation_result_keeps_its_kind_at_the_definition() {
    let guard = TraceContext::enter();
    let input = value();
    let result_info = VarInfo::new(17usize);
    let result = Invocation::invoke(
        Op::new(Neg),
        vec![],
        vec![input.clone()],
        vec![result_info.clone()],
    )
    .pop()
    .unwrap();
    let captured =
        Capture::from_context(guard.context(), &[input.into()], &[result.into()]).unwrap();
    let body = captured.expr();
    assert_eq!(body.inputs()[0].1, Kind::Value);
    let (var, kind, result) = &body.instructions()[0].as_apply().unwrap().outputs()[0];
    assert_eq!(kind, &Kind::Value);
    assert_eq!(result, &result_info);
    assert_eq!(body.kind(var), Some(*kind));
    let lowered = body.lower_with_aliases(&CellAliases::default()).unwrap();
    assert_eq!(lowered.expr.eqns()[0].outputs()[0].1, result_info.clone());
    assert_eq!(lowered.output_infos, vec![result_info]);
}

fn value_var(lowered: &gnx_expr::LoweredExpr, output: &lower::Output) -> gnx_expr::Var {
    let lower::Output::Value { result_idx } = *output else {
        panic!("expected immutable result")
    };
    lowered.expr.outputs()[result_idx]
}
fn result_var(lowered: &gnx_expr::LoweredExpr, index: usize) -> gnx_expr::Var {
    value_var(lowered, &lowered.results()[index])
}
fn state_var(lowered: &gnx_expr::LoweredExpr, index: usize) -> gnx_expr::Var {
    lowered.expr.outputs()[lowered.updates[index].1]
}

#[test]
fn local_reads_forward_exact_references_and_clones_share_the_slot() {
    let _guard = TraceContext::enter();
    let x = value();
    let y = value();
    let cell = CellTrace::new(x.trace_ref().clone());
    let alias = cell.clone();
    let old = cell.get();
    alias.set(y.trace_ref().clone());
    assert_eq!(TracerKey::from(&old), TracerKey::from(x.trace_ref()));
    assert_eq!(TracerKey::from(&cell.get()), TracerKey::from(y.trace_ref()));
    assert!(
        cell.get().producer().is_none(),
        "local get must not create a DAG node"
    );
}

#[test]
fn foreign_reads_are_structural_and_nested_writes_do_not_rebind_outer_slot() {
    let _outer = TraceContext::enter();
    let x = value();
    let replacement = value();
    let cell = CellTrace::new(x.trace_ref().clone());
    {
        let inner = TraceContext::enter();
        let read = cell.get();
        assert!(matches!(
            read.producer().unwrap().as_ref(),
            Invocation::ReadCell { .. }
        ));
        cell.set(replacement.trace_ref().clone());
        let captured = Capture::from_context(inner.context(), &[], &[read.into()]).unwrap();
        assert!(matches!(
            captured.expr().instructions()[0],
            HInstr::ReadCell { .. }
        ));
        assert!(matches!(
            captured.expr().instructions()[1],
            HInstr::WriteCell { .. }
        ));
        assert_eq!(
            captured.expr().cell_access(&captured.expr().closure()[0].0),
            Some(CellAccess::ReadWrite)
        );
    }
    assert_eq!(TracerKey::from(&cell.get()), TracerKey::from(x.trace_ref()));
}

#[test]
fn concrete_cell_bindings_detect_cloned_handles_across_captures_and_inputs() {
    let _guard = TraceContext::enter();
    let a = cell();
    let b = cell();
    let spec = InputSpec::from_args(&[a.clone().into()], &[b.into(), a.into()]);
    assert_eq!(spec.closure, vec![(InputId::Input(1), Kind::Cell, info())]);
    assert_eq!(
        spec.inputs,
        vec![
            (InputId::Input(0), Kind::Cell, info()),
            (InputId::Input(1), Kind::Cell, info()),
        ]
    );
}

#[test]
fn read_only_input_is_not_marked_writable() {
    let guard = TraceContext::enter();
    let cell = cell();
    let read = cell.get();
    let captured = Capture::from_context(guard.context(), &[cell.into()], &[read.into()]).unwrap();
    assert_eq!(
        captured.expr().cell_access(&captured.expr().inputs()[0].0),
        Some(CellAccess::Read)
    );
    let lowered = captured
        .expr()
        .lower_with_aliases(&CellAliases {
            inputs: vec![InputId::Input(0)],
            ..Default::default()
        })
        .unwrap();
    assert!(lowered.expr.eqns().is_empty());
    assert_eq!(result_var(&lowered, 0), lowered.expr.inputs()[0].0);
    assert!(lowered.updates.is_empty());
}

#[test]
fn the_same_hexpr_lowers_differently_for_aliased_and_distinct_inputs() {
    let guard = TraceContext::enter();
    let a = cell();
    let b = cell();
    let x = value();
    a.set(x.trace_ref().clone());
    let y = b.get();
    let captured = Capture::from_context(
        guard.context(),
        &[a.into(), b.into(), x.into()],
        &[y.into()],
    )
    .unwrap();
    let shared = captured
        .expr()
        .lower_with_aliases(&CellAliases {
            inputs: vec![InputId::Input(0), InputId::Input(0)],
            ..Default::default()
        })
        .unwrap();
    assert_eq!(shared.inputs, vec![InputId::Input(0), InputId::Input(1)]);
    assert_eq!(result_var(&shared, 0), shared.expr.inputs()[1].0);
    assert_eq!(shared.updates.len(), 1);
    assert_eq!(shared.updates[0].0, InputId::Input(0));

    let distinct = captured
        .expr()
        .lower_with_aliases(&CellAliases {
            inputs: vec![InputId::Input(0), InputId::Input(1)],
            ..Default::default()
        })
        .unwrap();
    assert_eq!(result_var(&distinct, 0), distinct.expr.inputs()[1].0);
    assert_eq!(distinct.updates.len(), 1);
    assert_eq!(distinct.updates[0].0, InputId::Input(0));
    assert!(shared.expr.eqns().is_empty());
    assert!(distinct.expr.eqns().is_empty());
}

#[test]
fn read_before_write_keeps_old_value_and_read_after_write_sees_new_value() {
    let guard = TraceContext::enter();
    let a = cell();
    let b = cell();
    let x = value();
    let before = b.get();
    a.set(x.trace_ref().clone());
    let after = b.get();
    let captured = Capture::from_context(
        guard.context(),
        &[a.into(), b.into(), x.into()],
        &[before.into(), after.into()],
    )
    .unwrap();
    let lowered = captured
        .expr()
        .lower_with_aliases(&CellAliases {
            inputs: vec![InputId::Input(7), InputId::Input(7)],
            ..Default::default()
        })
        .unwrap();
    assert_eq!(result_var(&lowered, 0), lowered.expr.inputs()[0].0);
    assert_eq!(result_var(&lowered, 1), lowered.expr.inputs()[1].0);
}

#[test]
fn captured_and_explicit_cells_can_alias() {
    let _outer = TraceContext::enter();
    let initial = value();
    let captured_cell = CellTrace::new(initial.trace_ref().clone());
    let inner = TraceContext::enter();
    let input_cell = cell();
    let x = value();
    captured_cell.set(x.trace_ref().clone());
    let read = input_cell.get();
    let captured = Capture::from_context(
        inner.context(),
        &[input_cell.into(), x.into()],
        &[read.into()],
    )
    .unwrap();
    let lowered = captured
        .expr()
        .lower_with_aliases(&CellAliases {
            closure: vec![InputId::Input(3)],
            inputs: vec![InputId::Input(3)],
        })
        .unwrap();
    assert_eq!(lowered.inputs.len(), 2);
    assert_eq!(result_var(&lowered, 0), lowered.expr.inputs()[1].0);
}

#[test]
fn escaping_local_cell_keeps_identity_and_final_contents() {
    let guard = TraceContext::enter();
    let x = value();
    let cell = CellTrace::new(x.trace_ref().clone());
    let final_value = neg(x.clone());
    cell.set(final_value.trace_ref().clone());
    let captured = Capture::from_context(
        guard.context(),
        &[x.into()],
        &[cell.clone().into(), cell.into()],
    )
    .unwrap();
    assert_eq!(captured.expr().outputs()[0], captured.expr().outputs()[1]);
    assert!(matches!(
        captured.expr().instructions()[1],
        HInstr::CreateCell { .. }
    ));
    let lowered = captured
        .expr()
        .lower_with_aliases(&CellAliases::default())
        .unwrap();
    assert_eq!(lowered.expr.eqns().len(), 1);
    assert!(matches!(
        lowered.results()[1],
        lower::Output::NewCell { id: 0, .. }
    ));
    assert!(matches!(
        lowered.results()[0],
        lower::Output::NewCell {
            id: 0,
            result_idx: 0
        }
    ));
    assert!(lowered.updates.is_empty());
    assert_eq!(
        lowered.expr.outputs()[0],
        lowered.expr.eqns()[0].outputs()[0].0
    );
}

#[test]
fn equal_contents_do_not_merge_distinct_local_cells() {
    let guard = TraceContext::enter();
    let x = value();
    let a = CellTrace::new(x.trace_ref().clone());
    let b = CellTrace::new(x.trace_ref().clone());
    let captured =
        Capture::from_context(guard.context(), &[x.into()], &[a.into(), b.into()]).unwrap();
    let lowered = captured
        .expr()
        .lower_with_aliases(&CellAliases::default())
        .unwrap();
    assert!(lowered.updates.is_empty());
    assert!(matches!(
        lowered.results()[0],
        lower::Output::NewCell { id: 0, .. }
    ));
    assert!(matches!(
        lowered.results()[1],
        lower::Output::NewCell { id: 1, .. }
    ));
    assert_ne!(lowered.results()[0], lowered.results()[1]);
}

#[test]
fn escaping_write_retains_its_computation_but_no_cell_equations_reach_expr() {
    let guard = TraceContext::enter();
    let cell = cell();
    let x = value();
    cell.set(neg(x.clone()).into_trace_ref());
    let captured = Capture::from_context(guard.context(), &[cell.into(), x.into()], &[]).unwrap();
    let lowered = captured
        .expr()
        .lower_with_aliases(&CellAliases {
            inputs: vec![InputId::Input(0)],
            ..Default::default()
        })
        .unwrap();
    assert_eq!(lowered.expr.eqns().len(), 1);
    assert_eq!(lowered.expr.eqns()[0].op().to_string(), "test::neg");
    assert_eq!(
        state_var(&lowered, 0),
        lowered.expr.eqns()[0].outputs()[0].0
    );
}

#[test]
fn explicit_value_input_cuts_its_producer_out_of_the_dag() {
    let guard = TraceContext::enter();
    let x = value();
    let middle = neg(x);
    let result = neg(middle.clone());
    let captured =
        Capture::from_context(guard.context(), &[middle.into()], &[result.into()]).unwrap();
    assert_eq!(captured.expr().instructions().len(), 1);
    assert!(captured.closure().is_empty());
}

#[test]
fn explicit_multioutput_input_is_not_overwritten_when_its_sibling_is_captured() {
    let guard = TraceContext::enter();
    let x = value();
    let mut pair = Invocation::invoke(Op::new(Neg), vec![], vec![x.clone()], vec![info(), info()]);
    let second = pair.pop().unwrap();
    let first = pair.pop().unwrap();
    let use_first = neg(first.clone());
    let captured = Capture::from_context(
        guard.context(),
        &[x.into(), first.into()],
        &[second.into(), use_first.into()],
    )
    .unwrap();
    let use_first = captured
        .expr()
        .instructions()
        .iter()
        .filter_map(HInstr::as_apply)
        .find(|eqn| &eqn.outputs()[0].0 == &captured.expr().outputs()[1])
        .unwrap();
    assert_eq!(&use_first.inputs()[0], &captured.expr().inputs()[1].0);
}

#[test]
fn local_computation_and_rebinding_do_not_form_an_ownership_cycle() {
    let weak = {
        let _guard = TraceContext::enter();
        let x = value();
        let local = CellTrace::new(x.trace_ref().clone());
        let arc: std::sync::Arc<gnx_expr::TraceCell> = local.clone().into();
        let weak = std::sync::Arc::downgrade(&arc);
        let result = Invocation::invoke(Op::new(Neg), vec![], vec![local.get()], vec![info()])
            .pop()
            .unwrap();
        local.set(result.into_trace_ref());
        weak
    };
    assert!(weak.upgrade().is_none());
}

#[test]
fn alias_bindings_must_have_compatible_cell_types() {
    let guard = TraceContext::enter();
    let a = cell();
    let b = CellTrace::placeholder(VarInfo::new(1usize));
    let captured = Capture::from_context(guard.context(), &[a.into(), b.into()], &[]).unwrap();
    assert!(matches!(
        captured.expr().lower_with_aliases(&CellAliases {
            inputs: vec![InputId::Input(0), InputId::Input(0)],
            ..Default::default()
        }),
        Err(LoweringError::AliasedInputKindMismatch(InputId::Input(0)))
    ));
}

#[test]
fn immutable_calls_preserve_the_boundary_without_duplicating_captures_as_arguments() {
    let body = gnx_expr::hexpr!(gnx_expr::VarInfo::new(0usize); {
        lambda capture; argument:
        output = test::neg argument;
        in (output)
    });
    let guard = TraceContext::enter();
    let x = value();
    let y = value();
    let result = gnx_transforms::cf::call_immutable(
        std::sync::Arc::new(body),
        [x.clone()],
        [y.clone()],
        vec![info()],
    );
    let captured = Capture::from_context(
        guard.context(),
        &[x.into(), y.into()],
        &result.into_iter().map(Into::into).collect::<Vec<_>>(),
    )
    .unwrap();
    let lowered = captured
        .expr()
        .lower_with_aliases(&CellAliases::default())
        .unwrap();
    let call = &lowered.expr.eqns()[0];
    assert_eq!(
        call.inputs().len(),
        1,
        "captures must not be duplicated as ordinary call arguments"
    );
    assert_eq!(call.op().to_string(), "cf::call");
    assert_eq!(call.subexprs().len(), 1);
    assert_eq!(call.subexprs()[0].closure().len(), 1);
    assert_eq!(
        call.subexprs()[0].expr().eqns()[0].op().to_string(),
        "test::neg"
    );
    assert_eq!(call.inputs()[0], lowered.expr.inputs()[1].0);
}

#[test]
fn missing_parameters_and_wrong_alias_binding_arity_are_errors() {
    let guard = TraceContext::enter();
    let unbound = value();
    assert!(matches!(
        Capture::from_context(guard.context(), &[], &[unbound.clone().into()]),
        Err(CaptureError::UndefinedValue)
    ));
    let local = CellTrace::new(unbound.into_trace_ref());
    assert!(matches!(
        Capture::from_context(guard.context(), &[local.into()], &[]),
        Err(CaptureError::LocalCellInput)
    ));
    let input = cell();
    let captured = Capture::from_context(guard.context(), &[input.into()], &[]).unwrap();
    assert!(matches!(
        captured.expr().lower(&InputSpec::default()),
        Err(LoweringError::InputSpecArity { .. })
    ));
}

#[test]
fn applying_a_mutable_body_updates_caller_local_cells_without_read_nodes() {
    let outer = TraceContext::enter();
    let initial = value();
    let replacement = value();
    let target = CellTrace::new(initial.trace_ref().clone());
    let captured = {
        let inner = TraceContext::enter();
        let input = cell();
        let new_value = value();
        input.set(new_value.trace_ref().clone());
        let result = input.get();
        Capture::from_context(
            inner.context(),
            &[input.into(), new_value.into()],
            &[result.into()],
        )
        .unwrap()
    };
    let results = captured
        .expr()
        .call_lowered(&[], &[target.clone().into(), replacement.clone().into()])
        .unwrap();
    assert_eq!(
        TracerKey::from(&target.get()),
        TracerKey::from(replacement.trace_ref())
    );
    assert_eq!(
        TracerKey::from(&results[0].resolve()),
        TracerKey::from(replacement.trace_ref())
    );
    let caller = Capture::from_context(outer.context(), &[replacement.into()], &results).unwrap();
    assert!(caller.expr().instructions().is_empty());
}

#[test]
fn lowered_calls_specialize_aliases_before_snapshotting_cells() {
    let outer = TraceContext::enter();
    let captured = {
        let inner = TraceContext::enter();
        let a = cell();
        let b = cell();
        let x = value();
        a.set(x.trace_ref().clone());
        let result = b.get();
        Capture::from_context(
            inner.context(),
            &[a.into(), b.into(), x.into()],
            &[result.into()],
        )
        .unwrap()
    };
    let a = cell();
    let x = value();
    let results = captured
        .expr()
        .call_lowered(&[], &[a.clone().into(), a.clone().into(), x.clone().into()])
        .unwrap();
    assert_eq!(
        TracerKey::from(&results[0].resolve()),
        TracerKey::from(x.trace_ref())
    );
    let caller = Capture::from_context(outer.context(), &[a.into(), x.into()], &results).unwrap();
    assert_eq!(caller.expr().instructions().len(), 1);
    assert!(matches!(
        caller.expr().instructions()[0],
        HInstr::WriteCell { .. }
    ));
}

#[test]
fn repeated_application_returns_fresh_cells_but_preserves_returned_aliases() {
    let outer = TraceContext::enter();
    let captured = {
        let inner = TraceContext::enter();
        let x = value();
        let local = CellTrace::new(x.trace_ref().clone());
        Capture::from_context(
            inner.context(),
            &[x.into()],
            &[local.clone().into(), local.into()],
        )
        .unwrap()
    };
    let x = value();
    let first = captured
        .expr()
        .call_lowered(&[], &[x.clone().into()])
        .unwrap();
    let second = captured
        .expr()
        .call_lowered(&[], &[x.clone().into()])
        .unwrap();
    let results = [first[0].clone(), first[1].clone(), second[0].clone()];
    let caller = Capture::from_context(outer.context(), &[x.into()], &results).unwrap();
    let lowered = caller
        .expr()
        .lower_with_aliases(&CellAliases::default())
        .unwrap();
    assert_eq!(lowered.results()[0], lowered.results()[1]);
    assert_ne!(lowered.results()[0], lowered.results()[2]);
}

#[test]
fn applying_captured_outer_bindings_is_explicit() {
    let _outer = TraceContext::enter();
    let x = value();
    let target = CellTrace::new(x.trace_ref().clone());
    let captured = {
        let inner = TraceContext::enter();
        let replacement = value();
        target.set(replacement.trace_ref().clone());
        Capture::from_context(inner.context(), &[replacement.into()], &[]).unwrap()
    };
    let replacement = value();
    captured
        .expr()
        .call_lowered(captured.closure(), &[replacement.clone().into()])
        .unwrap();
    assert_eq!(
        TracerKey::from(&target.get()),
        TracerKey::from(replacement.trace_ref())
    );
}

#[test]
fn returning_an_input_cell_does_not_read_its_contents() {
    let guard = TraceContext::enter();
    let input = cell();
    let captured =
        Capture::from_context(guard.context(), &[input.clone().into()], &[input.into()]).unwrap();
    assert_eq!(
        captured.expr().cell_access(&captured.expr().inputs()[0].0),
        Some(CellAccess::Unused)
    );
    assert!(captured.expr().instructions().is_empty());
    let lowered = captured
        .expr()
        .lower_with_aliases(&CellAliases {
            inputs: vec![InputId::Input(5)],
            ..Default::default()
        })
        .unwrap();
    assert_eq!(
        lowered.results()[0],
        lower::Output::InputCell {
            input_id: InputId::Input(5)
        }
    );
}

trait CallLowered {
    fn call_lowered(
        &self,
        closure: &[gnx_expr::ObjectTracer],
        inputs: &[gnx_expr::ObjectTracer],
    ) -> Result<Vec<gnx_expr::ObjectTracer>, gnx_expr::lower::CallError>;
}
impl CallLowered for gnx_expr::HExpr {
    fn call_lowered(
        &self,
        closure: &[gnx_expr::ObjectTracer],
        inputs: &[gnx_expr::ObjectTracer],
    ) -> Result<Vec<gnx_expr::ObjectTracer>, gnx_expr::lower::CallError> {
        let spec = InputSpec::from_args(closure, inputs);
        gnx_transforms::cf::call_lowered(&self.lower(&spec).unwrap(), closure, inputs)
    }
}

use std::fmt;

use gnx_expr::lower::{LoweringContext, Output};
use gnx_expr::{
    Capture, CellAccess, CellTrace, ConcreteValue, Eqn, EvalPlanError, HEqn, HExprBuilder,
    HExprRef, HRootExpr, InputId, InputSpec, Invocation, Kind, LoweringError, ObjectTracer, Op,
    Operation, Trace, TraceContext, TracerKey, Value, Var, VarInfo, dialect,
};
use rkyv::{Archive, Portable, Serialize};

#[derive(Archive, Serialize, Portable)]
#[rkyv(as = Self)]
#[repr(C)]
struct Compute;
impl Operation for Compute {}

// Runs its one subexpression on its inputs, like `cf::call`.
#[derive(Archive, Serialize, Portable)]
#[rkyv(as = Self)]
#[repr(C)]
struct Call;
impl Operation for Call {
    fn cell_access(&self, eqn: HEqn<'_>, cell: Var) -> CellAccess {
        let explicit = (eqn.subexprs().next()).map_or(CellAccess::Unused, |body| {
            body.input_cell_access(eqn.inputs(), cell)
        });
        eqn.captured_cell_access(cell).combine(explicit)
    }

    fn lower(&self, eqn: HEqn<'_>, ctx: &mut LoweringContext<'_>) -> Result<(), LoweringError> {
        let mut subexprs = eqn.subexprs();
        let (Some(sub), None) = (subexprs.next(), subexprs.next()) else {
            return Err(LoweringError::CallArity);
        };
        let body = sub.expr();
        if eqn.inputs().len() != body.inputs().len() || eqn.outputs().len() != body.outputs().len()
        {
            return Err(LoweringError::CallArity);
        }
        let region = ctx.lower_region(sub, eqn.inputs())?;
        let physical = ctx.allocate_results(region.output_infos());
        ctx.bind_region_results(eqn.outputs(), &region, &physical)?;
        ctx.apply_region_updates(&region, &physical)?;
        if region.program().expr.view().eqns().len() != 0 || !physical.is_empty() {
            ctx.emit(
                eqn.op().to_owned(),
                region.inputs(),
                physical,
                vec![region.subexpr()],
            );
        }
        Ok(())
    }

    fn check_expr(&self, eqn: Eqn<'_>) -> Result<(), LoweringError> {
        let Some(sub) = eqn.subexprs().next() else {
            return Err(LoweringError::CallArity);
        };
        if eqn.inputs().len() != sub.expr().inputs().len()
            || eqn.outputs().len() != sub.expr().outputs().len()
        {
            return Err(LoweringError::CallArity);
        }
        Ok(())
    }
}

dialect! {
    dyn Operation, "lowtest" => {
        "compute" => Compute,
        "call" => Call,
    }
}

fn info() -> VarInfo {
    VarInfo::new(String::from("number"))
}
fn compute() -> Op {
    Op::new(Compute)
}
fn spec(closure: &[(InputId, Kind)], inputs: &[(InputId, Kind)]) -> InputSpec {
    let entries = |entries: &[(InputId, Kind)]| {
        (entries.iter())
            .map(|(id, kind)| (*id, *kind, info()))
            .collect()
    };
    InputSpec {
        closure: entries(closure),
        inputs: entries(inputs),
    }
}

#[test]
fn local_cells_become_value_flow() {
    let mut b = HRootExpr::builder();
    let x = b.input(Kind::Value, info());
    let cell = b.create_cell(x, info());
    let before = b.read_cell(cell, info());
    let [y] = (b.eqn(compute()).input(before))
        .output(Kind::Value, info())
        .done();
    b.write_cell(cell, y);
    let z = b.read_cell(cell, info());
    b.outputs([z, cell, x]);
    let root = b.build().unwrap();

    let lowered = (root.view())
        .lower(&spec(&[], &[(InputId::Input(0), Kind::Value)]))
        .unwrap();
    let expr = lowered.expr.view();
    assert_eq!(expr.closure_inputs().len(), 0);
    assert_eq!(expr.inputs().len(), 1);
    // The cell actions are gone: only the computation remains.
    assert_eq!(expr.eqns().len(), 1);
    let eqn = expr.eqns().next().unwrap();
    let input = expr.inputs().next().unwrap().var();
    assert_eq!(eqn.inputs(), [input]);
    let computed = eqn.outputs().next().unwrap().var();
    // z and the contents of the new cell are the computed value, and the
    // input is forwarded.
    assert_eq!(expr.outputs(), [computed, computed, input]);
    assert_eq!(
        lowered.outputs,
        [
            Output::Value { result_idx: 0 },
            Output::NewCell {
                id: 0,
                result_idx: 1
            },
            Output::Value { result_idx: 2 },
        ]
    );
    assert!(lowered.updates.is_empty());
    assert_eq!(lowered.inputs, [InputId::Input(0)]);

    // The trace body and the computation return each computed result once.
    assert_eq!(lowered.output_infos.len(), 1);
    assert_eq!(lowered.trace_body().view().outputs(), [computed]);
    assert_eq!(lowered.trace_body().view().instrs().len(), 1);
    assert_eq!(lowered.computation_expr().view().outputs(), [computed]);
}

#[test]
fn writes_to_input_cells_become_updates() {
    let mut b = HRootExpr::builder();
    let cell = b.input(Kind::Cell, info());
    let other = b.input(Kind::Cell, info());
    let before = b.read_cell(cell, info());
    let [y] = (b.eqn(compute()).input(before))
        .output(Kind::Value, info())
        .done();
    b.write_cell(cell, y);
    // Reads the other cell after the write.
    let after = b.read_cell(other, info());
    b.outputs([after, cell]);
    let root = b.build().unwrap();
    let view = root.view();
    assert_eq!(view.cell_access(cell), Some(CellAccess::ReadWrite));
    assert_eq!(view.cell_access(other), Some(CellAccess::Read));

    // Distinct cells
    let distinct = spec(
        &[],
        &[
            (InputId::Input(0), Kind::Cell),
            (InputId::Input(1), Kind::Cell),
        ],
    );
    let lowered = view.lower(&distinct).unwrap();
    let expr = lowered.expr.view();
    let params: Vec<_> = expr.inputs().map(|param| param.var()).collect();
    let computed = expr.eqns().next().unwrap().outputs().next().unwrap().var();
    assert_eq!(params.len(), 2);
    // `after` is the initial contents of the other cell.
    assert_eq!(expr.outputs(), [params[1], computed]);
    assert_eq!(
        lowered.outputs,
        [
            Output::Value { result_idx: 0 },
            Output::InputCell {
                input_id: InputId::Input(0)
            }
        ]
    );
    assert_eq!(lowered.updates, [(InputId::Input(0), 1)]);

    // The same cell passed twice: the read sees the write.
    let aliased = spec(
        &[],
        &[
            (InputId::Input(0), Kind::Cell),
            (InputId::Input(0), Kind::Cell),
        ],
    );
    let lowered = view.lower(&aliased).unwrap();
    let expr = lowered.expr.view();
    assert_eq!(expr.inputs().len(), 1);
    let computed = expr.eqns().next().unwrap().outputs().next().unwrap().var();
    assert_eq!(expr.outputs(), [computed, computed]);
    assert_eq!(lowered.inputs, [InputId::Input(0)]);
    assert_eq!(lowered.updates, [(InputId::Input(0), 1)]);
}

#[test]
fn specs_are_checked() {
    let mut b = HRootExpr::builder();
    let x = b.input(Kind::Value, info());
    b.output(x);
    let root = b.build().unwrap();
    assert_eq!(
        root.view().lower(&spec(&[], &[])).unwrap_err(),
        LoweringError::InputSpecArity {
            closure: false,
            expected: 1,
            actual: 0
        }
    );
    assert_eq!(
        (root.view())
            .lower(&spec(&[], &[(InputId::Input(0), Kind::Cell)]))
            .unwrap_err(),
        LoweringError::KindMismatch {
            closure: false,
            index: 0
        }
    );

    // An operation without a rule for subexpressions
    let mut b = HRootExpr::builder();
    b.eqn(compute()).subexpr(HExprBuilder::new());
    let root = b.build().unwrap();
    assert_eq!(
        root.view().lower(&spec(&[], &[])).unwrap_err(),
        LoweringError::UnsupportedNestedOperation
    );
}

// closure: (state: cell), inputs: (v), returns compute(read state, v) and
// writes it back to the state.
fn accumulate() -> HExprRef {
    let mut b = HExprBuilder::new();
    let state = b.closure_input(Kind::Cell, info());
    let v = b.input(Kind::Value, info());
    let current = b.read_cell(state, info());
    let [next] = (b.eqn(compute()).inputs([current, v]))
        .output(Kind::Value, info())
        .done();
    b.write_cell(state, next);
    b.output(next);
    b.finish()
}

#[test]
fn regions_thread_cell_state_through_their_parent() {
    let body = accumulate();
    let mut b = HRootExpr::builder();
    let x = b.input(Kind::Value, info());
    let state = b.create_cell(x, info());
    let mut results = Vec::new();
    for _ in 0..2 {
        let mut e = b.eqn(Op::new(Call));
        e.input(x).output(Kind::Value, info());
        e.subexpr(body.clone()).vars([state]);
        let [r] = e.done();
        results.push(r);
    }
    let last = b.read_cell(state, info());
    b.outputs([results[0], results[1], last]);
    let root = b.build().unwrap();
    assert_eq!(root.view().cell_access(state), Some(CellAccess::ReadWrite));

    let lowered = (root.view())
        .lower(&spec(&[], &[(InputId::Input(0), Kind::Value)]))
        .unwrap();
    let expr = lowered.expr.view();
    expr.check().unwrap();
    let eqns: Vec<_> = expr.eqns().collect();
    assert_eq!(eqns.len(), 2);
    let input = expr.inputs().next().unwrap().var();
    let first = eqns[0].outputs().next().unwrap().var();
    let second = eqns[1].outputs().next().unwrap().var();

    // Each call takes the explicit input, and is provided the state: first
    // the initial contents of the cell, then the result of the first call.
    assert_eq!(eqns[0].inputs(), [input]);
    assert_eq!(eqns[0].subexprs().next().unwrap().provided_vars(), [input]);
    assert_eq!(eqns[1].subexprs().next().unwrap().provided_vars(), [first]);
    assert_eq!(expr.outputs(), [first, second, second]);

    // The lowered body is pure, and stored in a root of its own.
    let region = eqns[0].subexprs().next().unwrap();
    assert!(region.external().is_some());
    let lowered_body = region.expr();
    assert_eq!(lowered_body.closure_inputs().len(), 1);
    assert_eq!(lowered_body.inputs().len(), 1);
    assert_eq!(lowered_body.eqns().len(), 1);
    // Both calls use the same lowered body.
    assert!(region.external() == eqns[1].subexprs().next().unwrap().external());

    // The trace body mirrors the lowered expression, with value-only bodies.
    let trace = lowered.trace_body().view();
    assert_eq!(trace.instrs().len(), 2);
    let call = trace.instrs().next().unwrap().as_apply().unwrap();
    let body = call.subexprs().next().unwrap().expr();
    assert!(
        body.closure_inputs()
            .all(|param| param.kind() == Kind::Value)
    );
    assert_eq!(body.instrs().len(), 1);
}

// ---- tracing, replay and evaluation ----

#[derive(Debug)]
struct Number(i64);
impl fmt::Display for Number {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}
impl ConcreteValue for Number {
    fn to_info(&self) -> VarInfo {
        info()
    }
}

#[derive(Debug)]
struct WrongType;
impl fmt::Display for WrongType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("wrong type")
    }
}
impl ConcreteValue for WrongType {
    fn to_info(&self) -> VarInfo {
        VarInfo::new(String::from("wrong"))
    }
}

fn value(n: i64) -> Value {
    Value::new(Number(n))
}
fn concrete(n: i64) -> Trace {
    Trace::concrete(value(n), info())
}
fn number(trace: &Trace) -> i64 {
    (trace.try_concrete().unwrap())
        .downcast::<Number>()
        .unwrap()
        .0
}
fn apply(inputs: Vec<Trace>, outputs: usize) -> Vec<Trace> {
    Invocation::invoke(compute(), vec![], inputs, vec![info(); outputs])
        .into_iter()
        .map(|value| value.into_trace_ref())
        .collect()
}

#[test]
fn captured_traces_lower_and_replay() {
    let captured = {
        let guard = TraceContext::enter();
        let cell = CellTrace::placeholder(info());
        let x = Trace::placeholder(info());
        let y = apply(vec![cell.get(), x.clone()], 1).pop().unwrap();
        cell.set(y.clone());
        Capture::from_context(
            guard.context(),
            &[ObjectTracer::from(cell), ObjectTracer::from(x)],
            &[ObjectTracer::from(y)],
        )
        .unwrap()
    };
    assert_eq!(captured.expr().instrs().len(), 3);

    // Replay on new arguments: a cell and a value.
    let home = Trace::placeholder(info());
    let cell = CellTrace::new(home.clone());
    let arg = Trace::placeholder(info());
    let arguments = [ObjectTracer::from(cell.clone()), ObjectTracer::from(arg)];
    let spec = InputSpec::from_args(captured.closure(), &arguments);
    let lowered = captured.expr().lower(&spec).unwrap();
    assert_eq!(lowered.expr.view().eqns().len(), 1);
    assert_eq!(lowered.updates.len(), 1);

    let mut emitted = None;
    let results = lowered
        .replay(
            captured.closure(),
            &arguments,
            |body, captures, inputs, infos| {
                assert_eq!(body.view().instrs().len(), 1);
                assert!(captures.is_empty());
                assert_eq!(inputs.len(), 2);
                // The cell is passed by its contents at the call.
                assert_eq!(TracerKey::from(&inputs[0]), TracerKey::from(&home));
                let outputs: Vec<_> = (infos.into_iter()).map(Trace::placeholder).collect();
                emitted = Some(outputs[0].clone());
                outputs
            },
        )
        .unwrap();
    let emitted = emitted.unwrap();
    let [ObjectTracer::Ref(result)] = results.as_slice() else {
        panic!("expected one value result");
    };
    assert_eq!(TracerKey::from(result), TracerKey::from(&emitted));
    // The update of the cell was applied.
    assert_eq!(
        TracerKey::from(&cell.read_home()),
        TracerKey::from(&emitted)
    );
}

#[test]
fn live_requested_and_sibling_results_are_published_in_place() {
    let a = concrete(2);
    let [z, sibling]: [Trace; 2] = apply(vec![a.clone()], 2).try_into().unwrap();
    let x = apply(vec![z.clone(), z.clone()], 1).pop().unwrap();
    let alias = z.clone();
    let capture = Capture::from_roots([&x, &x]).unwrap();
    assert_eq!(capture.expr().instrs().len(), 2);
    let selected = capture.outputs();
    assert_eq!(
        (selected.iter())
            .map(|out| TracerKey::from(&out.resolve()))
            .collect::<Vec<_>>(),
        [&x, &z, &sibling].map(TracerKey::from)
    );

    let (expr, args, commit) = capture.prepare().unwrap().into_parts();
    assert_eq!(expr.view().outputs().len(), 3);
    assert!(std::ptr::eq(
        args[0].downcast::<Number>().unwrap(),
        a.try_concrete().unwrap().downcast::<Number>().unwrap()
    ));
    commit.publish([value(12), value(6), value(5)]).unwrap();
    assert_eq!(
        [number(&x), number(&z), number(&alias), number(&sibling)],
        [12, 6, 6, 5]
    );
    // Published values cut the trace.
    assert!(x.producer().is_none());
}

#[test]
fn eval_checks_its_boundary() {
    // An unbound placeholder cannot be evaluated.
    let x = apply(vec![Trace::placeholder(info())], 1).pop().unwrap();
    let capture = Capture::from_roots([&x]).unwrap();
    assert!(matches!(
        capture.prepare(),
        Err(EvalPlanError::UnboundPlaceholder)
    ));

    let x = apply(vec![concrete(1)], 1).pop().unwrap();
    let commit = |x: &Trace| {
        let capture = Capture::from_roots([x]).unwrap();
        capture.prepare().unwrap().into_parts().2
    };
    assert!(matches!(
        commit(&x).publish([]),
        Err(EvalPlanError::OutputArity {
            expected: 1,
            actual: 0
        })
    ));
    assert!(matches!(
        commit(&x).publish([Value::new(WrongType)]),
        Err(EvalPlanError::OutputInfo { index: 0, .. })
    ));
    assert!(x.try_concrete().is_none());
    commit(&x).publish([value(7)]).unwrap();
    assert_eq!(number(&x), 7);
}

#[test]
fn discarded_results_and_names_survive_lowering() {
    let mut b = HRootExpr::builder().with_name_hint("body");
    let x = b.input(Kind::Value, info());
    let cell = b.create_cell(x, info());
    let mut e = b.eqn(compute());
    e.input(x).output(Kind::Value, info());
    let [kept] = e.done();
    b.push(gnx_expr::HInstrSource::Apply(gnx_expr::HEqnSource::new(
        compute(),
        vec![],
        vec![kept],
        vec![(Var::hole(), Kind::Value, info())],
    )));
    // Reading into a hole discards the value.
    b.push(gnx_expr::HInstrSource::ReadCell {
        cell,
        output: Var::hole(),
        info: info(),
    });
    b.output(kept);
    let root = b.build().unwrap();

    let lowered = (root.view())
        .lower(&spec(&[], &[(InputId::Input(0), Kind::Value)]))
        .unwrap();
    let expr = lowered.expr.view();
    assert_eq!(expr.name_hint(), Some("body"));
    assert_eq!(lowered.trace_body().view().name_hint(), Some("body"));
    let eqns: Vec<_> = expr.eqns().collect();
    assert_eq!(eqns.len(), 2);
    assert!(eqns[1].outputs().next().unwrap().var().is_hole());
    assert_eq!(
        expr.to_string(),
        concat!(
            "|a: number| => {\n",
            "  b: number = lowtest::compute a;\n",
            "  _: number = lowtest::compute b;\n",
            "  in (b)\n}",
        )
    );
}

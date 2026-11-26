use gnx_base::{ArrayInfo, DType, Shape};
use gnx_expr::{
    Capture, CellTrace, HExpr, HParam, InputSpec, Kind, LoweringError, ObjectTracer, Trace,
    TraceContext, VarInfo, hexpr,
};
use gnx_transforms::cf::{ScanParameters, cond, scan, scan_op, switch, while_loop};
use gnx_transforms::ops;

fn info(shape: &[usize], dtype: DType) -> VarInfo {
    VarInfo::new(ArrayInfo::new(Shape::fixed(shape.iter().copied()), dtype))
}

fn region(
    inputs: &[ObjectTracer],
    body: impl FnOnce(&[ObjectTracer]) -> Vec<ObjectTracer>,
) -> Capture {
    let guard = TraceContext::enter();
    let placeholders: Vec<_> = inputs
        .iter()
        .map(|input| match input {
            ObjectTracer::Ref(value) => Trace::placeholder(value.info().clone()).into(),
            ObjectTracer::Cell(cell) => CellTrace::placeholder(cell.type_info().clone()).into(),
        })
        .collect();
    let outputs = body(&placeholders);
    Capture::from_context(guard.context(), &placeholders, &outputs).unwrap()
}

fn value_only_spec(expr: HExpr<'_>) -> InputSpec {
    use gnx_expr::InputId;
    fn spec<'a>(
        params: impl Iterator<Item = HParam<'a>>,
        id: fn(usize) -> InputId,
    ) -> Vec<(InputId, Kind, VarInfo)> {
        params
            .enumerate()
            .map(|(index, param)| {
                assert!(matches!(param.kind(), Kind::Value));
                (id(index), param.kind(), param.info_ref().to_owned())
            })
            .collect()
    }
    InputSpec {
        closure: spec(expr.closure_inputs(), InputId::Closure),
        inputs: spec(expr.inputs(), InputId::Input),
    }
}

#[test]
fn lowering_preserves_hints_on_programs_and_structured_regions() {
    let flag = info(&[], DType::Bool);
    let value = info(&[], DType::F32);
    let yes = hexpr!({ lambda x: &value => in (x) }).with_name_hint("yes");
    let no = hexpr!({ lambda x: &value => in (x) }).with_name_hint(String::from("no"));
    let program = hexpr!({
        lambda pred: flag, x: &value =>
        result: &value = cf::cond[yes => ()][no => ()] pred x;
        in (result)
    })
    .with_name_hint("choose")
    .build()
    .unwrap();
    let program = program.view();
    assert_eq!(program.name_hint(), Some("choose"));
    let lowered = program.lower(&value_only_spec(program)).unwrap();
    let trace_body = lowered.trace_body().view();
    assert_eq!(trace_body.name_hint(), Some("choose"));
    let replay = trace_body.lower(&value_only_spec(trace_body)).unwrap();
    for expr in [&lowered.expr, &replay.expr] {
        let expr = expr.view();
        assert_eq!(expr.name_hint(), Some("choose"));
        let regions: Vec<_> = expr.eqns().next().unwrap().subexprs().collect();
        assert_eq!(regions[0].expr().name_hint(), Some("yes"));
        assert_eq!(regions[1].expr().name_hint(), Some("no"));
    }
}

#[test]
fn branch_specializes_an_alias_between_an_operand_and_a_capture() {
    let guard = TraceContext::enter();
    let scalar = info(&[], DType::F32);
    let target = CellTrace::placeholder(scalar.clone());
    let other = CellTrace::placeholder(scalar.clone());
    let value = Trace::placeholder(scalar);
    let flag = Trace::placeholder(info(&[], DType::Bool));
    let args = [other.clone().into(), value.clone().into()];
    let yes = region(&args, |inputs| {
        let ObjectTracer::Cell(read) = &inputs[0] else {
            panic!("expected cell")
        };
        target.set(inputs[1].resolve());
        vec![read.get().into()]
    });
    let no = region(&args, |inputs| {
        let ObjectTracer::Cell(read) = &inputs[0] else {
            panic!("expected cell")
        };
        vec![read.get().into()]
    });
    let result = cond(flag.clone(), &yes, &no, &args).unwrap().remove(0);
    let params = [
        flag.clone().into(),
        target.clone().into(),
        other.into(),
        value.clone().into(),
    ];
    let capture = Capture::from_context(guard.context(), &params, &[result.into()]).unwrap();
    let distinct = capture
        .expr()
        .lower(&InputSpec::from_args(&[], &params))
        .unwrap();
    let aliased = [
        flag.into(),
        target.clone().into(),
        target.into(),
        value.into(),
    ];
    let aliased = capture
        .expr()
        .lower(&InputSpec::from_args(&[], &aliased))
        .unwrap();
    assert_eq!(aliased.results().len(), 1);
    assert_eq!(aliased.updates.len(), 1);
    for (program, free_cells) in [(&distinct, 1), (&aliased, 0)] {
        let eqn = program.expr.view().eqns().next().unwrap();
        assert_eq!(eqn.outputs().len(), 2);
        for branch in eqn.subexprs() {
            assert_eq!(branch.expr().inputs().len(), 2);
            assert_eq!(branch.provided_vars().len(), free_cells);
        }
    }
    let eqn = aliased.expr.view().eqns().next().unwrap();
    let true_branch = eqn.subexprs().next().unwrap().expr();
    let new = true_branch.inputs().next().unwrap().var();
    assert_eq!(true_branch.eqns().len(), 0);
    assert_eq!(true_branch.outputs(), &[new, new]);
}

#[test]
fn read_only_branch_cells_stay_captures() {
    let guard = TraceContext::enter();
    let cell = CellTrace::placeholder(info(&[], DType::F32));
    let flag = Trace::placeholder(info(&[], DType::Bool));
    let branch = region(&[], |_| vec![cell.get().into()]);
    let result = cond(flag.clone(), &branch, &branch, &[]).unwrap().remove(0);
    let inputs = [flag.into(), cell.into()];
    let capture = Capture::from_context(guard.context(), &inputs, &[result.into()]).unwrap();
    let lowered = capture
        .expr()
        .lower(&InputSpec::from_args(&[], &inputs))
        .unwrap();
    let eqn = lowered.expr.view().eqns().next().unwrap();
    assert_eq!(eqn.inputs().len(), 1); // selector only
    assert_eq!(eqn.outputs().len(), 1);
    assert!(lowered.updates.is_empty());
    for branch in eqn.subexprs() {
        assert_eq!(branch.provided_vars().len(), 1);
        assert_eq!(branch.expr().inputs().len(), 0);
    }
}

#[test]
fn scan_tracks_hidden_state_separately_and_can_be_lowered_again() {
    let guard = TraceContext::enter();
    let scalar = info(&[], DType::F32);
    let cell = CellTrace::placeholder(scalar.clone());
    let carry = Trace::placeholder(scalar.clone());
    let xs = Trace::placeholder(info(&[0], DType::F32));
    let step = region(
        &[carry.clone().into(), Trace::placeholder(scalar).into()],
        |args| {
            let old = cell.get();
            cell.set(args[1].resolve());
            vec![args[0].clone(), old.into()]
        },
    );
    let (final_carry, ys) = scan(&step, &[carry.clone()], &[xs.clone()], None).unwrap();
    assert_eq!(final_carry.len(), 1);
    assert_eq!(ys.len(), 1);
    assert_eq!(ys[0].info(), xs.info());
    let inputs = [cell.into(), carry.into(), xs.into()];
    let capture = Capture::from_context(
        guard.context(),
        &inputs,
        &[final_carry[0].clone().into(), ys[0].clone().into()],
    )
    .unwrap();
    assert!(capture.expr().to_string().contains("<carry: 1, length: 0>"));
    let lowered = capture
        .expr()
        .lower(&InputSpec::from_args(&[], &inputs))
        .unwrap();
    let trace_body = lowered.trace_body().view();
    let replay = trace_body.lower(&value_only_spec(trace_body)).unwrap();
    for program in [&lowered, &replay] {
        let eqn = program.expr.view().eqns().next().unwrap();
        let config = ScanParameters::from_op_ref(eqn.op()).unwrap();
        assert_eq!(
            (config.carry_len, config.state_len, config.length),
            (1, 1, 0)
        );
        assert_eq!(eqn.inputs().len(), 3);
        assert_eq!(eqn.outputs().len(), 3);
    }
    assert_eq!(lowered.updates.len(), 1);
    assert!(replay.updates.is_empty());
}

#[test]
fn scan_counts_use_usize_and_check_carry_overflow() {
    let op = scan_op(usize::MAX, usize::MAX);
    let parameters = ScanParameters::from_op(&op).unwrap();
    assert_eq!(parameters.physical_carry_len().unwrap(), usize::MAX);
    assert_eq!(parameters.length, usize::MAX);
    assert!(
        ScanParameters {
            state_len: 1,
            ..parameters
        }
        .physical_carry_len()
        .is_err()
    );
}

#[test]
fn control_flow_rejects_invalid_region_interfaces() {
    let guard = TraceContext::enter();
    let scalar = info(&[], DType::F32);
    let carry = Trace::placeholder(scalar.clone());
    let input = [ObjectTracer::from(carry.clone())];
    let one = region(&input, |args| args.to_vec());
    let none = region(&input, |_| vec![]);
    let selector = Trace::placeholder(info(&[], DType::Bool));
    assert!(matches!(
        cond(selector, &one, &none, &input),
        Err(LoweringError::RegionSignature(_))
    ));
    let index = Trace::placeholder(info(&[], DType::I32));
    assert!(matches!(
        switch(index, &[], &input),
        Err(LoweringError::RegionSignature(_))
    ));
    let cell = CellTrace::placeholder(scalar);
    let predicate = region(&input, |args| {
        cell.set(args[0].resolve());
        vec![Trace::concrete(gnx_expr::Value::new(TestBool), info(&[], DType::Bool)).into()]
    });
    assert!(matches!(
        while_loop(&predicate, &one, &[carry.clone()]),
        Err(LoweringError::RegionSignature(
            "while predicate cannot write a captured cell"
        ))
    ));
    assert!(matches!(
        scan(&one, &[carry], &[], None),
        Err(LoweringError::RegionSignature(_))
    ));
    drop(guard);
}

#[derive(Debug)]
struct TestBool;
impl std::fmt::Display for TestBool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("bool")
    }
}
impl gnx_expr::ConcreteValue for TestBool {
    fn to_info(&self) -> VarInfo {
        info(&[], DType::Bool)
    }
}

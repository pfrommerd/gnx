use gnx_expr::{
    Dialect, Eqn, Expr, Input, Op, Operation, Output, SubExpr, ValidationIssue, VarScope,
};
use std::sync::Arc;

struct TestDialect;
struct TestOperation;

static TEST_DIALECT: TestDialect = TestDialect;
static TEST_OP: TestOperation = TestOperation;

impl Dialect for TestDialect {
    fn name(&self) -> &'static str {
        "validation"
    }

    fn create(&self, _name: &'static str) -> Option<&'static dyn Operation> {
        Some(&TEST_OP)
    }
}

impl Operation for TestOperation {
    fn name(&self) -> &'static str {
        "op"
    }

    fn dialect(&self) -> &'static dyn Dialect {
        &TEST_DIALECT
    }
}

fn op() -> Op {
    Op::new(&TEST_OP)
}

#[test]
fn check_collects_multiple_errors() {
    let mut vars = VarScope::new();
    let x = vars.create_var();
    let missing = vars.create_var();
    let out = vars.create_var();

    let expr = Expr::new(
        vec![],
        vec![Input::new(x)],
        vec![
            Eqn::new(op(), vec![Input::new(missing)], vec![Output::New(out)]),
            Eqn::new(op(), vec![Input::new(missing)], vec![Output::New(out)]),
        ],
        vec![missing],
        vars,
    );

    let err = expr.check().expect_err("expression should be invalid");
    assert!(err.errors().len() >= 3);
    assert!(
        err.errors()
            .iter()
            .any(|issue| matches!(issue, ValidationIssue::UndefinedInput { eqn: 0, .. }))
    );
    assert!(
        err.errors()
            .iter()
            .any(|issue| matches!(issue, ValidationIssue::UndefinedInput { eqn: 1, .. }))
    );
    assert!(
        err.errors()
            .iter()
            .any(|issue| matches!(issue, ValidationIssue::DuplicateOutput { eqn: 1, .. }))
    );
}

#[test]
fn check_still_validates_capture_arity_and_variable_bindings() {
    let mut inner_vars = VarScope::new();
    let captured = inner_vars.create_var();
    let inner = Expr::new(
        vec![Input::new(captured)],
        vec![],
        vec![],
        vec![captured],
        inner_vars,
    );

    let mut outer_vars = VarScope::new();
    let missing = outer_vars.create_var();
    let out = outer_vars.create_var();
    let expr = Expr::new(
        vec![],
        vec![],
        vec![Eqn::new_with_subexprs(
            op(),
            vec![SubExpr::new(
                Arc::new(inner),
                vec![Input::new(missing), Input::new(missing)],
            )],
            vec![],
            vec![out],
        )],
        vec![out],
        outer_vars,
    );
    let error = expr.check().unwrap_err();
    assert!(error.errors().iter().any(|issue| matches!(
        issue,
        ValidationIssue::SubExprClosureArity {
            expected: 1,
            actual: 2,
            ..
        }
    )));
    assert!(
        error
            .errors()
            .iter()
            .any(|issue| matches!(issue, ValidationIssue::UndefinedSubExprClosure { .. }))
    );
}

#[test]
fn check_binds_subexpression_capture_to_an_outer_value() {
    let mut inner_vars = VarScope::new();
    let inner_capture = inner_vars.create_var();
    let inner = Expr::new(
        vec![Input::new(inner_capture)],
        vec![],
        vec![],
        vec![inner_capture],
        inner_vars,
    );

    let mut outer_vars = VarScope::new();
    let outer_value = outer_vars.create_var();
    let out = outer_vars.create_var();
    let expr = Expr::new(
        vec![Input::new(outer_value)],
        vec![],
        vec![Eqn::new_with_subexprs(
            op(),
            vec![SubExpr::new(Arc::new(inner), vec![Input::new(outer_value)])],
            vec![],
            vec![Output::New(out)],
        )],
        vec![out],
        outer_vars,
    );

    expr.check()
        .expect("capture should bind to the outer value");
}

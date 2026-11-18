use gnx_expr::{Dialect, Expr, LoweringError, Operation, Ownership, Var, expr, register_dialect};
use std::sync::Arc;

struct TestDialect;
struct TestOperation;
register_dialect!(&TestDialect);

impl Dialect for TestDialect {
    fn name(&self) -> &'static str {
        "validation"
    }
    fn create(&self, name: &'static str) -> Option<&'static dyn Operation> {
        (name == "op").then_some(&TestOperation)
    }
}
impl Operation for TestOperation {
    fn name(&self) -> &'static str {
        "op"
    }
    fn dialect(&self) -> &'static dyn Dialect {
        &TestDialect
    }
}

#[test]
fn check_rejects_undefined_operands() {
    let expr =
        expr!(gnx_expr::ValueInfo::new(0usize); { lambda: out = validation::op missing; in (out) });
    let missing = expr.eqns()[0].inputs()[0].0;
    assert_eq!(expr.check(), Err(LoweringError::UndefinedVariable(missing)));
}

#[test]
fn check_rejects_duplicate_and_hole_definitions() {
    let expr = expr!(gnx_expr::ValueInfo::new(0usize); {
        lambda:
        out = validation::op;
        out = validation::op;
        in (out)
    });
    let out = expr.outputs()[0];
    assert_eq!(expr.check(), Err(LoweringError::DuplicateDefinition(out)));
    let expr = Expr::new(
        vec![],
        vec![(
            Var::hole(),
            Ownership::Move,
            gnx_expr::ValueInfo::new(0usize),
        )],
        vec![],
        vec![],
    );
    assert!(expr.check().is_err());
}

#[test]
fn check_validates_subexpression_capture_arity_and_bindings() {
    let inner =
        Arc::new(expr!(gnx_expr::ValueInfo::new(0usize); { lambda captured;: in (captured) }));
    let expr = expr!(gnx_expr::ValueInfo::new(0usize); { lambda outer;: out = validation::op[inner.clone() => ()]; in (out) });
    assert_eq!(
        expr.check(),
        Err(LoweringError::CaptureArity {
            expected: 1,
            actual: 0
        })
    );
    let expr = expr!(gnx_expr::ValueInfo::new(0usize); { lambda outer;: out = validation::op[inner.clone() => (missing)]; in (out) });
    let missing = expr.eqns()[0].subexprs()[0].captures[0].0;
    assert_eq!(expr.check(), Err(LoweringError::UndefinedVariable(missing)));
    let expr = expr!(gnx_expr::ValueInfo::new(0usize); { lambda outer;: out = validation::op[inner => (outer)]; in (out) });
    expr.check().unwrap();
}

#[test]
fn check_enforces_owned_returns_and_move_lifetimes() {
    let borrowed = expr!(gnx_expr::ValueInfo::new(0usize); { lambda &x: in (x) });
    let x = borrowed.inputs()[0].0;
    assert_eq!(borrowed.check(), Err(LoweringError::BorrowedOutput(x)));
    let moved_twice = expr!(gnx_expr::ValueInfo::new(0usize); { lambda x: in (x, x) });
    let x = moved_twice.inputs()[0].0;
    assert_eq!(
        moved_twice.check(),
        Err(LoweringError::UndefinedVariable(x))
    );
}

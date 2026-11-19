use gnx_expr::{Dialect, Expr, LoweringError, Operation, Var, expr};
use std::sync::Arc;

struct TestDialect;
#[derive(Clone, PartialEq, Eq, Hash)]
struct TestOperation;

impl Dialect for TestDialect {
    fn name(&self) -> &'static str {
        "validation"
    }
}
impl std::fmt::Display for TestOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}::{}", Operation::dialect(self).name(), Operation::name(self))
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

mod ops {
    pub mod validation {
        pub fn op() -> gnx_expr::Op {
            gnx_expr::Op::new(super::super::TestOperation)
        }
    }
}

#[test]
fn check_rejects_undefined_operands() {
    let expr =
        expr!(gnx_expr::ValueInfo::new(0usize); { lambda: out = validation::op missing; in (out) });
    let missing = expr.eqns()[0].inputs()[0];
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
        vec![(Var::hole(), gnx_expr::ValueInfo::new(0usize))],
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
    let missing = expr.eqns()[0].subexprs()[0].captures[0];
    assert_eq!(expr.check(), Err(LoweringError::UndefinedVariable(missing)));
    let expr = expr!(gnx_expr::ValueInfo::new(0usize); { lambda outer;: out = validation::op[inner => (outer)]; in (out) });
    expr.check().unwrap();
}

#[test]
fn check_accepts_returning_parameters_and_reusing_values() {
    let expr = expr!(gnx_expr::ValueInfo::new(0usize); {
        lambda captured; x:
        out = validation::op x x;
        in (x, captured, out, out)
    });
    expr.check().unwrap();
    assert_eq!(
        expr.outputs(),
        &[
            expr.inputs()[0].0,
            expr.closure()[0].0,
            expr.eqns()[0].outputs()[0].0,
            expr.eqns()[0].outputs()[0].0
        ]
    );
}

#[test]
fn check_rejects_undefined_results() {
    let expr = expr!(gnx_expr::ValueInfo::new(0usize); { lambda: in (missing) });
    assert_eq!(
        expr.check(),
        Err(LoweringError::UndefinedVariable(expr.outputs()[0]))
    );
}

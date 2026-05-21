use gnx_expr::{
    Attr, AttrMap, Dialect, Effect, Eqn, Expr, Input, Op, Operation, VarScope, attr, expr,
    register_dialect,
};
use std::fmt::Display;

struct TestDialect;
struct TestOperation {
    name: &'static str,
}

static TEST_DIALECT: TestDialect = TestDialect;
static SIN: TestOperation = TestOperation { name: "sin" };
static ADD: TestOperation = TestOperation { name: "add" };
static CALL: TestOperation = TestOperation { name: "call" };

register_dialect!(&TEST_DIALECT);

impl Dialect for TestDialect {
    fn name(&self) -> &'static str {
        "test"
    }

    fn create(&self, name: &'static str) -> Option<&'static dyn Operation> {
        match name {
            "sin" => Some(&SIN),
            "add" => Some(&ADD),
            "call" => Some(&CALL),
            _ => None,
        }
    }
}

impl Operation for TestOperation {
    fn name(&self) -> &'static str {
        self.name
    }

    fn dialect(&self) -> &'static dyn Dialect {
        &TEST_DIALECT
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct InfoType;

impl Display for InfoType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "InfoType")
    }
}

#[test]
fn pretty_expr_uses_alphabetic_names() {
    let mut vars = VarScope::new();
    let x = vars.create_var();
    let y = vars.create_var();
    let z = vars.create_var();
    let expr = Expr::new(
        vec![],
        vec![Input::new(x, Effect::Read), Input::new(y, Effect::Read)],
        vec![Eqn::new(
            Op::new(&ADD, AttrMap::default(), None),
            vec![],
            vec![Input::new(x, Effect::Read), Input::new(y, Effect::Read)],
            vec![z],
        )],
        vec![z],
        vars,
    );

    assert_eq!(
        expr.to_string(),
        "{ lambda  ; a, b .\n  c = test.add a b\n  in (c) }"
    );
}

#[test]
fn nested_expr_closure_names_match_outer_names() {
    let mut inner_vars = VarScope::new();
    let inner_closure = inner_vars.create_var();
    let inner_input = inner_vars.create_var();
    let inner_output = inner_vars.create_var();
    let inner = Expr::new(
        vec![Input::new(inner_closure, Effect::Read)],
        vec![Input::new(inner_input, Effect::Read)],
        vec![Eqn::new(
            Op::new(&ADD, AttrMap::default(), None),
            vec![],
            vec![
                Input::new(inner_closure, Effect::Read),
                Input::new(inner_input, Effect::Read),
            ],
            vec![inner_output],
        )],
        vec![inner_output],
        inner_vars,
    );

    let mut outer_vars = VarScope::new();
    let outer_closure = outer_vars.create_var();
    let outer_input = outer_vars.create_var();
    let outer_output = outer_vars.create_var();
    let outer = Expr::new(
        vec![Input::new(outer_closure, Effect::Read)],
        vec![Input::new(outer_input, Effect::Read)],
        vec![Eqn::new(
            Op::new(&CALL, AttrMap::default(), Some(vec![inner])),
            vec![Input::new(outer_closure, Effect::Read)],
            vec![Input::new(outer_input, Effect::Read)],
            vec![outer_output],
        )],
        vec![outer_output],
        outer_vars,
    );

    let printed = outer.to_string();
    assert!(printed.contains("{ lambda a ; b ."));
    assert!(printed.contains("test.call { lambda a ; b ."));
}

#[test]
fn attr_macro_builds_pretty_attr_shapes() {
    let attr = attr!({ name: "relu", tags: ["pointwise", "fast"] });
    assert_eq!(
        attr.to_string(),
        r#"{name: "relu", tags: ["pointwise", "fast"]}"#
    );
    assert!(matches!(attr, Attr::Map(_)));
    assert_eq!(attr!(info(InfoType)).to_string(), "info(InfoType)");
}

#[test]
fn expr_macro_builds_printed_structure() {
    let expr = expr!({
        lambda ; x, y .
        z = test::add(x, y);
        in (z)
    });
    assert_eq!(
        expr.to_string(),
        "{ lambda  ; a, b .\n  c = test::add a b\n  in (c) }"
    );
}

use gnx_expr::{
    Attr, AttrMap, Dialect, Effect, Eqn, Expr, Input, Op, Operation, SubExpr, VarScope, attr, expr,
    register_dialect,
};
use std::fmt::Display;
use std::sync::Arc;

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
            Op::new(&ADD, AttrMap::default()),
            vec![Input::new(x, Effect::Read), Input::new(y, Effect::Read)],
            vec![z],
        )],
        vec![z],
        vars,
    );

    assert_eq!(
        expr.to_string(),
        "{ lambda  ; a, b .\n  c = test::add a b\n  in (c) }"
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
            Op::new(&ADD, AttrMap::default()),
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
        vec![Eqn::new_with_subexprs(
            Op::new(&CALL, AttrMap::default()),
            vec![SubExpr::new(Arc::new(inner), vec![outer_closure])],
            vec![Input::new(outer_input, Effect::Read)],
            vec![outer_output],
        )],
        vec![outer_output],
        outer_vars,
    );

    let printed = outer.to_string();
    assert!(printed.contains("{ lambda a ; b ."));
    assert!(printed.contains("test::call { lambda a ; b ."));
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
        "{ lambda  ; x, y .\n  z = test::add x y\n  in (z) }"
    );
}

#[test]
fn hinted_var_names_are_disambiguated() {
    let mut vars = VarScope::new();
    let x1 = vars.create_var_with_hint("x");
    let x2 = vars.create_var_with_hint("x");
    let x3 = vars.create_var_with_hint("x");
    let expr = Expr::new(
        vec![],
        vec![Input::new(x1, Effect::Read), Input::new(x2, Effect::Read)],
        vec![Eqn::new(
            Op::new(&ADD, AttrMap::default()),
            vec![Input::new(x1, Effect::Read), Input::new(x2, Effect::Read)],
            vec![x3],
        )],
        vec![x3],
        vars,
    );

    assert_eq!(
        expr.to_string(),
        "{ lambda  ; x, x_2 .\n  x_3 = test::add x x_2\n  in (x_3) }"
    );
}

#[test]
fn large_subexprs_can_be_deferred_with_hinted_names() {
    let mut inner_vars = VarScope::new();
    let inner_input = inner_vars.create_var_with_hint("b");
    let inner_output = inner_vars.create_var_with_hint("c");
    let inner = Expr::new(
        vec![],
        vec![Input::new(inner_input, Effect::Read)],
        vec![Eqn::new(
            Op::new(&SIN, AttrMap::default()),
            vec![Input::new(inner_input, Effect::Read)],
            vec![inner_output],
        )],
        vec![inner_output],
        inner_vars,
    )
    .with_name_hint("foo");

    let mut outer_vars = VarScope::new();
    let outer_input = outer_vars.create_var_with_hint("b");
    let outer_output = outer_vars.create_var_with_hint("c");
    let outer = Expr::new(
        vec![],
        vec![Input::new(outer_input, Effect::Read)],
        vec![Eqn::new_with_subexprs(
            Op::new(&CALL, AttrMap::default()),
            vec![SubExpr::new(Arc::new(inner), vec![])],
            vec![Input::new(outer_input, Effect::Read)],
            vec![outer_output],
        )],
        vec![outer_output],
        outer_vars,
    );

    assert_eq!(
        outer.pretty().max_inline_eqns(0).to_string(),
        "{ lambda  ; b .\n  c = test::call @foo b\n  in (c) }\nwhere\n@foo = { lambda  ; b .\n  c = test::sin b\n  in (c) }"
    );
}

#[test]
fn inline_subexpr_indentation_is_configurable() {
    let mut inner_vars = VarScope::new();
    let inner_input = inner_vars.create_var_with_hint("b");
    let inner_output = inner_vars.create_var_with_hint("c");
    let inner = Expr::new(
        vec![],
        vec![Input::new(inner_input, Effect::Read)],
        vec![Eqn::new(
            Op::new(&SIN, AttrMap::default()),
            vec![Input::new(inner_input, Effect::Read)],
            vec![inner_output],
        )],
        vec![inner_output],
        inner_vars,
    );

    let mut outer_vars = VarScope::new();
    let outer_input = outer_vars.create_var_with_hint("b");
    let outer_output = outer_vars.create_var_with_hint("c");
    let outer = Expr::new(
        vec![],
        vec![Input::new(outer_input, Effect::Read)],
        vec![Eqn::new_with_subexprs(
            Op::new(&CALL, AttrMap::default()),
            vec![SubExpr::new(Arc::new(inner), vec![])],
            vec![Input::new(outer_input, Effect::Read)],
            vec![outer_output],
        )],
        vec![outer_output],
        outer_vars,
    );

    assert_eq!(
        outer.pretty().indent_width(4).to_string(),
        "{ lambda  ; b .\n    c = test::call { lambda  ; b .\n        c = test::sin b\n        in (c) } b\n    in (c) }"
    );
}

#[test]
fn deferred_subexprs_are_deduplicated_by_arc_pointer() {
    let mut inner_vars = VarScope::new();
    let inner_input = inner_vars.create_var_with_hint("b");
    let inner_output = inner_vars.create_var_with_hint("c");
    let inner = Arc::new(
        Expr::new(
            vec![],
            vec![Input::new(inner_input, Effect::Read)],
            vec![Eqn::new(
                Op::new(&SIN, AttrMap::default()),
                vec![Input::new(inner_input, Effect::Read)],
                vec![inner_output],
            )],
            vec![inner_output],
            inner_vars,
        )
        .with_name_hint("foo"),
    );

    let mut outer_vars = VarScope::new();
    let input = outer_vars.create_var_with_hint("b");
    let first = outer_vars.create_var_with_hint("c");
    let second = outer_vars.create_var_with_hint("d");
    let outer = Expr::new(
        vec![],
        vec![Input::new(input, Effect::Read)],
        vec![
            Eqn::new_with_subexprs(
                Op::new(&CALL, AttrMap::default()),
                vec![SubExpr::new(Arc::clone(&inner), vec![])],
                vec![Input::new(input, Effect::Read)],
                vec![first],
            ),
            Eqn::new_with_subexprs(
                Op::new(&CALL, AttrMap::default()),
                vec![SubExpr::new(Arc::clone(&inner), vec![])],
                vec![Input::new(input, Effect::Read)],
                vec![second],
            ),
        ],
        vec![first, second],
        outer_vars,
    );

    assert_eq!(
        outer.pretty().max_inline_eqns(0).to_string(),
        "{ lambda  ; b .\n  c = test::call @foo b\n  d = test::call @foo b\n  in (c, d) }\nwhere\n@foo = { lambda  ; b .\n  c = test::sin b\n  in (c) }"
    );
}

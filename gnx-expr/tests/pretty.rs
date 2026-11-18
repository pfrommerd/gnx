use gnx_expr::{
    Attr, Dialect, Eqn, Expr, Op, Operation, Ownership, VarScope, attr, expr, register_dialect,
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
        vec![
            (x, Ownership::Move, gnx_expr::ValueInfo::new(0usize)),
            (y, Ownership::Move, gnx_expr::ValueInfo::new(0usize)),
        ],
        vec![Eqn::new(
            Op::new(&ADD),
            vec![(x, Ownership::Move), (y, Ownership::Move)],
            vec![(z, gnx_expr::Kind::Value(gnx_expr::ValueInfo::new(0usize)))],
        )],
        vec![z],
    );

    assert_eq!(
        expr.to_string(),
        "{ lambda a: 0, b: 0 =>\n  c: 0 = test::add a b;\n  in (c) }"
    );
}

#[test]
fn nested_expr_closure_names_match_outer_names() {
    let inner = expr!(gnx_expr::ValueInfo::new(0usize); { lambda captured; b: c = test::add captured b; in (c) });
    let outer = expr!(gnx_expr::ValueInfo::new(0usize); { lambda a; b: c = test::call[inner => (a)] b; in (c) });
    outer.check().unwrap();
    let printed = outer.to_string();
    assert!(printed.contains("{ lambda a: 0; b: 0 =>"));
    assert!(printed.contains("test::call { lambda a: 0; b: 0 =>"));
}

#[test]
fn nested_captures_use_the_same_ownership_syntax_as_operands() {
    let inner = expr!(gnx_expr::ValueInfo::new(0usize); { lambda &borrowed, owned;: in (owned) });
    let outer = expr!(gnx_expr::ValueInfo::new(0usize); {
        lambda &x, y:
        result = test::call[inner => (&x, y)];
        in (result)
    });
    let captures = &outer.eqns()[0].subexprs()[0].captures;
    assert_eq!(captures[0].1, Ownership::Borrow);
    assert_eq!(captures[1].1, Ownership::Move);
    outer.check().unwrap();
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
fn expr_macro_and_printer_preserve_explicit_ownership() {
    let body = expr!(gnx_expr::ValueInfo::new(0usize); {
        lambda &x, y:
        z = test::add &x y;
        in (z)
    });
    body.check().unwrap();
    assert_eq!(body.inputs()[0].1, Ownership::Borrow);
    assert_eq!(body.inputs()[1].1, Ownership::Move);
    assert_eq!(body.eqns()[0].inputs()[0].1, Ownership::Borrow);
    assert_eq!(body.eqns()[0].inputs()[1].1, Ownership::Move);
    assert!(body.to_string().contains("lambda &x: 0, y: 0 =>"));
    assert!(body.to_string().contains("test::add &x y"));
}

#[test]
fn expr_macro_parses_operation_attributes() {
    let body = expr!(gnx_expr::ValueInfo::new(0usize); {
        lambda x:
        y = test::sin<label: "test", tags: ["one", "two"]> x;
        in (y)
    });
    assert_eq!(body.eqns()[0].op().attrs()["label"], attr!("test"));
    assert_eq!(body.eqns()[0].op().attrs()["tags"], attr!(["one", "two"]));
    body.check().unwrap();
}

#[test]
fn expr_macro_builds_printed_structure() {
    let info = gnx_expr::ValueInfo::new(0usize);
    let expr = expr!({
        lambda x: &info, y: &info =>
        z: info = test::add x y;
        in (z)
    });
    assert_eq!(
        expr.to_string(),
        "{ lambda x: 0, y: 0 =>\n  z: 0 = test::add x y;\n  in (z) }"
    );
}

#[test]
fn typed_expr_annotations_evaluate_rust_expressions_and_clone_only_on_borrow() {
    struct Config {
        info: gnx_expr::ValueInfo,
    }
    let config = Config {
        info: gnx_expr::ValueInfo::new(5usize),
    };
    let owned = gnx_expr::ValueInfo::new(6usize);
    let owned_ptr = owned.downcast::<usize>().unwrap() as *const usize;
    let mut evaluated = 0;
    let expr = expr!({
        lambda &bias: &config.info; x: owned =>
        z: { let n = 7; evaluated += 1; gnx_expr::ValueInfo::new(n) } = test::add &bias x;
        in (z)
    });
    assert_eq!(evaluated, 1);
    assert!(std::ptr::eq(
        expr.closure()[0].2.downcast::<usize>().unwrap(),
        config.info.downcast::<usize>().unwrap()
    ));
    assert!(std::ptr::eq(
        expr.inputs()[0].2.downcast::<usize>().unwrap(),
        owned_ptr
    ));
    assert_eq!(
        expr.value_info(&expr.outputs()[0]),
        Some(&gnx_expr::ValueInfo::new(7i32))
    );
    assert_eq!(
        expr.to_string(),
        "{ lambda &bias: 5; x: 6 =>\n  z: 7 = test::add &bias x;\n  in (z) }"
    );
    expr.check().unwrap();
}

#[test]
fn hinted_var_names_are_disambiguated() {
    let mut vars = VarScope::new();
    let x1 = vars.create_var_with_hint("x");
    let x2 = vars.create_var_with_hint("x");
    let x3 = vars.create_var_with_hint("x");
    let expr = Expr::new(
        vec![],
        vec![
            (x1, Ownership::Move, gnx_expr::ValueInfo::new(0usize)),
            (x2, Ownership::Move, gnx_expr::ValueInfo::new(0usize)),
        ],
        vec![Eqn::new(
            Op::new(&ADD),
            vec![(x1, Ownership::Move), (x2, Ownership::Move)],
            vec![(x3, gnx_expr::Kind::Value(gnx_expr::ValueInfo::new(0usize)))],
        )],
        vec![x3],
    );

    assert_eq!(
        expr.to_string(),
        "{ lambda x: 0, x_2: 0 =>\n  x_3: 0 = test::add x x_2;\n  in (x_3) }"
    );
}

#[test]
fn large_subexprs_can_be_deferred_with_hinted_names() {
    let inner = expr!(gnx_expr::ValueInfo::new(0usize); { lambda b: c = test::sin b; in (c) })
        .with_name_hint("foo");
    let outer = expr!(gnx_expr::ValueInfo::new(0usize); { lambda b: c = test::call[inner => ()] b; in (c) });

    assert_eq!(
        outer.pretty().max_inline_eqns(0).to_string(),
        "{ lambda b: 0 =>\n  c: 0 = test::call @foo b;\n  in (c) }\nwhere\n@foo = { lambda b: 0 =>\n  c: 0 = test::sin b;\n  in (c) }"
    );
}

#[test]
fn inline_subexpr_indentation_is_configurable() {
    let inner = expr!(gnx_expr::ValueInfo::new(0usize); { lambda b: c = test::sin b; in (c) });
    let outer = expr!(gnx_expr::ValueInfo::new(0usize); { lambda b: c = test::call[inner => ()] b; in (c) });

    assert_eq!(
        outer.pretty().indent_width(4).to_string(),
        "{ lambda b: 0 =>\n    c: 0 = test::call { lambda b: 0 =>\n        c: 0 = test::sin b;\n        in (c) } b;\n    in (c) }"
    );
}

#[test]
fn deferred_subexprs_are_deduplicated_by_arc_pointer() {
    let inner = Arc::new(
        expr!(gnx_expr::ValueInfo::new(0usize); { lambda b: c = test::sin b; in (c) })
            .with_name_hint("foo"),
    );
    let outer = expr!(gnx_expr::ValueInfo::new(0usize); {
        lambda b:
        c = test::call[Arc::clone(&inner) => ()] &b;
        d = test::call[inner => ()] b;
        in (c, d)
    });

    assert_eq!(
        outer.pretty().max_inline_eqns(0).to_string(),
        "{ lambda b: 0 =>\n  c: 0 = test::call @foo &b;\n  d: 0 = test::call @foo b;\n  in (c, d) }\nwhere\n@foo = { lambda b: 0 =>\n  c: 0 = test::sin b;\n  in (c) }"
    );
}

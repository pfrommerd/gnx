use gnx_expr::{Dialect, Eqn, Expr, Op, Operation, VarScope, expr};
use std::sync::Arc;

struct TestDialect;
#[derive(Clone, PartialEq, Eq, Hash)]
struct TestOperation {
    name: &'static str,
}

mod ops {
    pub mod test {
        use super::super::{ADD, CALL, Parameterized, SIN};
        use gnx_expr::Op;

        pub fn add() -> Op {
            Op::new(ADD.clone())
        }
        pub fn call() -> Op {
            Op::new(CALL.clone())
        }
        pub fn sin() -> Op {
            Op::new(SIN.clone())
        }
        pub fn parameterized(axis: usize, enabled: bool, tags: &[&str]) -> Op {
            Op::new(Parameterized {
                axis,
                enabled,
                tags: tags.iter().map(|tag| (*tag).to_owned()).collect(),
            })
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct Parameterized {
    axis: usize,
    enabled: bool,
    tags: Vec<String>,
}

impl std::fmt::Display for Parameterized {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "test::parameterized<axis: {}, enabled: {}, tags: {:?}>",
            self.axis, self.enabled, self.tags
        )
    }
}

impl Operation for Parameterized {
    fn name(&self) -> &'static str {
        "parameterized"
    }
    fn dialect(&self) -> &'static dyn Dialect {
        &TEST_DIALECT
    }

}

static TEST_DIALECT: TestDialect = TestDialect;
static SIN: TestOperation = TestOperation { name: "sin" };
static ADD: TestOperation = TestOperation { name: "add" };
static CALL: TestOperation = TestOperation { name: "call" };

impl Dialect for TestDialect {
    fn name(&self) -> &'static str {
        "test"
    }
}

impl std::fmt::Display for TestOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}::{}", Operation::dialect(self).name(), Operation::name(self))
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

#[test]
fn pretty_expr_uses_alphabetic_names() {
    let mut vars = VarScope::new();
    let x = vars.create_var();
    let y = vars.create_var();
    let z = vars.create_var();
    let expr = Expr::new(
        vec![],
        vec![
            (x, gnx_expr::VarInfo::new(0usize)),
            (y, gnx_expr::VarInfo::new(0usize)),
        ],
        vec![Eqn::new(
            Op::new(ADD.clone()),
            vec![x, y],
            vec![(z, gnx_expr::VarInfo::new(0usize))],
        )],
        vec![z],
    );

    assert_eq!(
        expr.to_string(),
        "|a: 0, b: 0| => {\n  c: 0 = test::add a b;\n  in (c)\n}"
    );
}

#[test]
fn nested_expr_closure_names_match_outer_names() {
    let inner = expr!(gnx_expr::VarInfo::new(0usize); { lambda captured; a: c = test::add captured a; in (c) });
    let outer = expr!(gnx_expr::VarInfo::new(0usize); { lambda a; b: c = test::call[inner => (a)] b; in (c) });
    outer.check().unwrap();
    assert_eq!(
        outer.to_string(),
        "|a: 0; b: 0| => {\n  c: 0 = test::call [|a_2: 0| => {\n    c: 0 = test::add a a_2;\n    in (c)\n  }] b;\n  in (c)\n}"
    );
}

#[test]
fn inline_expr_captures_values_in_first_use_order() {
    let info = gnx_expr::VarInfo::new(0usize);
    let outer = expr!(|x: &info, y: &info| => {
        local: &info = test::sin x;
        result: &info = test::call [|x: &info| => {
            z: &info = test::add local x;
            in (z, y, local)
        }] [|| => (x)] x;
        in (result)
    });
    outer.check().unwrap();
    let [first, second] = outer.eqns()[1].subexprs() else {
        panic!("expected two regions")
    };
    assert_eq!(
        first.closure(),
        &[outer.eqns()[0].outputs()[0].0, outer.inputs()[1].0]
    );
    assert_eq!(second.closure(), &[outer.inputs()[0].0]);
    assert!(outer.to_string().contains("[|| => (x)]"));
    let inner = first.expr();
    assert_eq!(
        inner.eqns()[0].inputs(),
        &[inner.closure()[0].0, inner.inputs()[0].0]
    );
    assert_eq!(
        inner.outputs()[1..],
        [inner.closure()[1].0, inner.closure()[0].0]
    );
}

#[test]
fn expr_without_equations_can_return_repeated_values_directly() {
    let info = gnx_expr::VarInfo::new(0usize);
    let body = expr!(|arg: info| => (arg, arg));
    body.check().unwrap();
    assert!(body.eqns().is_empty());
    assert_eq!(body.outputs(), &[body.inputs()[0].0; 2]);
    assert_eq!(body.to_string(), "|arg: 0| => (arg, arg)");
}

#[test]
fn inline_expr_passes_captures_through_intermediate_regions() {
    let outer = expr!(gnx_expr::VarInfo::new(0usize); {
        lambda bias, input =>
        result = test::call { lambda x =>
            y = test::call { lambda => z = test::add bias x; in (z) };
            in (y)
        } input;
        in (result)
    });
    outer.check().unwrap();
    let middle = &outer.eqns()[0].subexprs()[0];
    assert_eq!(middle.closure(), &[outer.inputs()[0].0]);
    let inner = &middle.expr().eqns()[0].subexprs()[0];
    assert_eq!(
        inner.closure(),
        &[middle.expr().closure()[0].0, middle.expr().inputs()[0].0]
    );
    assert_eq!(
        outer.pretty().max_inline_eqns(0).to_string(),
        concat!(
            "a = |bias: 0, x: 0;| => {\n  z: 0 = test::add bias x;\n  in (z)\n}\n",
            "b = |bias: 0; x: 0| => {\n  y: 0 = test::call [@a{bias, x}];\n  in (y)\n}\n",
            "in\n|bias: 0, input: 0| => {\n  result: 0 = test::call [@b{bias}] input;\n  in (result)\n}",
        )
    );
}

#[test]
fn nested_captures_bind_in_declaration_order() {
    let inner = expr!(gnx_expr::VarInfo::new(0usize); { lambda borrowed, owned;: in (owned) });
    let outer = expr!(gnx_expr::VarInfo::new(0usize); {
        lambda x, y:
        result = test::call[inner => (x, y)];
        in (result)
    });
    let captures = &outer.eqns()[0].subexprs()[0].captures;
    assert_eq!(captures, &[outer.inputs()[0].0, outer.inputs()[1].0]);
    outer.check().unwrap();
}

#[test]
fn expr_macro_and_printer_preserve_operand_order() {
    let body = expr!(gnx_expr::VarInfo::new(0usize); {
        lambda x, y:
        z = test::add x y;
        in (z)
    });
    body.check().unwrap();
    assert_eq!(
        body.eqns()[0].inputs(),
        &[body.inputs()[0].0, body.inputs()[1].0]
    );
    assert!(body.to_string().contains("|x: 0, y: 0| =>"));
    assert!(body.to_string().contains("test::add x y"));
}

#[test]
fn expr_macro_passes_operation_arguments_as_rust_expressions() {
    let mut evaluated = Vec::new();
    let body = expr!(gnx_expr::VarInfo::new(0usize); {
        lambda x:
        y = test::parameterized<{ evaluated.push(1); 2 },
            { evaluated.push(2); 3 > 2 }, &["one", "two"],> x;
        in (y)
    });
    assert_eq!(evaluated, [1, 2]);
    let op = body.eqns()[0].op();
    assert_eq!(op, &ops::test::parameterized(2, true, &["one", "two"]));
    assert_eq!(
        op.to_string(),
        r#"test::parameterized<axis: 2, enabled: true, tags: ["one", "two"]>"#
    );
    body.check().unwrap();
}

#[test]
fn expr_macro_builds_printed_structure() {
    let info = gnx_expr::VarInfo::new(0usize);
    let expr = expr!({
        lambda x: &info, y: &info =>
        z: info = test::add x y;
        in (z)
    });
    assert_eq!(
        expr.to_string(),
        "|x: 0, y: 0| => {\n  z: 0 = test::add x y;\n  in (z)\n}"
    );
}

#[test]
fn typed_expr_annotations_evaluate_rust_expressions_and_clone_only_on_borrow() {
    struct Config {
        info: gnx_expr::VarInfo,
    }
    let config = Config {
        info: gnx_expr::VarInfo::new(5usize),
    };
    let owned = gnx_expr::VarInfo::new(6usize);
    let owned_ptr = owned.downcast::<usize>().unwrap() as *const usize;
    let mut evaluated = 0;
    let expr = expr!({
        lambda bias: &config.info; x: owned =>
        z: { let n = 7; evaluated += 1; gnx_expr::VarInfo::new(n) } = test::add bias x;
        in (z)
    });
    assert_eq!(evaluated, 1);
    assert!(std::ptr::eq(
        expr.closure()[0].1.downcast::<usize>().unwrap(),
        config.info.downcast::<usize>().unwrap()
    ));
    assert!(std::ptr::eq(
        expr.inputs()[0].1.downcast::<usize>().unwrap(),
        owned_ptr
    ));
    assert_eq!(
        expr.value_info(&expr.outputs()[0]),
        Some(&gnx_expr::VarInfo::new(7i32))
    );
    assert_eq!(
        expr.to_string(),
        "|bias: 5; x: 6| => {\n  z: 7 = test::add bias x;\n  in (z)\n}"
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
            (x1, gnx_expr::VarInfo::new(0usize)),
            (x2, gnx_expr::VarInfo::new(0usize)),
        ],
        vec![Eqn::new(
            Op::new(ADD.clone()),
            vec![(x1), (x2)],
            vec![(x3, gnx_expr::VarInfo::new(0usize))],
        )],
        vec![x3],
    );

    assert_eq!(
        expr.to_string(),
        "|x: 0, x_2: 0| => {\n  x_3: 0 = test::add x x_2;\n  in (x_3)\n}"
    );
}

#[test]
fn large_subexprs_can_be_deferred_with_hinted_names() {
    let inner = expr!(gnx_expr::VarInfo::new(0usize); { lambda b: c = test::sin b; in (c) })
        .with_name_hint("foo");
    let outer = expr!(gnx_expr::VarInfo::new(0usize); { lambda b: c = test::call[inner => ()] b; in (c) });

    assert_eq!(
        outer.pretty().max_inline_eqns(0).to_string(),
        "foo = |b: 0| => {\n  c: 0 = test::sin b;\n  in (c)\n}\nin\n|b: 0| => {\n  c: 0 = test::call [@foo] b;\n  in (c)\n}"
    );
}

#[test]
fn inline_subexpr_indentation_is_configurable() {
    let inner = expr!(gnx_expr::VarInfo::new(0usize); { lambda b: c = test::sin b; in (c) });
    let outer = expr!(gnx_expr::VarInfo::new(0usize); { lambda b: c = test::call[inner => ()] b; in (c) });

    assert_eq!(
        outer.pretty().indent_width(4).to_string(),
        "|b: 0| => {\n    c: 0 = test::call [|b: 0| => {\n        c: 0 = test::sin b;\n        in (c)\n    }] b;\n    in (c)\n}"
    );
}

#[test]
fn deferred_subexprs_are_deduplicated_by_arc_pointer() {
    let inner = Arc::new(
        expr!(gnx_expr::VarInfo::new(0usize); { lambda b: c = test::sin b; in (c) })
            .with_name_hint("foo"),
    );
    let outer = expr!(gnx_expr::VarInfo::new(0usize); {
        lambda b:
        c = test::call[Arc::clone(&inner) => ()] b;
        d = test::call[inner => ()] b;
        in (c, d)
    });

    assert_eq!(
        outer.pretty().max_inline_eqns(0).to_string(),
        "foo = |b: 0| => {\n  c: 0 = test::sin b;\n  in (c)\n}\nin\n|b: 0| => {\n  c: 0 = test::call [@foo] b;\n  d: 0 = test::call [@foo] b;\n  in (c, d)\n}"
    );
}

#[test]
fn deferred_subexprs_share_a_definition_across_capture_bindings() {
    let info = gnx_expr::VarInfo::new(0usize);
    let outer = expr! {
        foo = |c: &info;| => { out: &info = test::sin c; in (out) }
        in |x: &info, y: &info| => {
            a: &info = test::call [@foo{x}];
            b: &info = test::call [@foo{y}];
            c: &info = test::call [@foo{x}];
            in (a, b, c)
        }
    };
    outer.check().unwrap();
    assert!(Arc::ptr_eq(
        outer.eqns()[0].subexprs()[0].expr(),
        outer.eqns()[1].subexprs()[0].expr()
    ));
    assert_eq!(
        outer.pretty().max_inline_eqns(0).to_string(),
        concat!(
            "foo = |c: 0;| => {\n  out: 0 = test::sin c;\n  in (out)\n}\n",
            "in\n|x: 0, y: 0| => {\n",
            "  a: 0 = test::call [@foo{x}];\n",
            "  b: 0 = test::call [@foo{y}];\n",
            "  c: 0 = test::call [@foo{x}];\n",
            "  in (a, b, c)\n}",
        ),
    );
}

#[test]
fn named_exprs_can_reference_earlier_definitions() {
    let outer = expr!(gnx_expr::VarInfo::new(0usize);
        first = || => { y = test::sin; in (y) }
        second = || => { y = test::call [@first]; in (y) }
        in || => { y = test::call [@second]; in (y) }
    );
    outer.check().unwrap();
    let printed = outer.pretty().max_inline_eqns(0).to_string();
    assert!(printed.starts_with("first = |"));
    assert!(printed.contains("\nsecond = |"));
    assert!(printed.contains("test::call [@first];"));
}

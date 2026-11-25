use gnx_expr::{
    BuildError, EqnSource, ExprBuilder, HInstr, Kind, Operation, Var, VarInfo, VarScope, dialect,
    expr, hexpr,
};
use rkyv::{Archive, Portable, Serialize};

macro_rules! operation {
    ($ty:ident) => {
        #[derive(Archive, Serialize, Portable)]
        #[rkyv(as = Self)]
        #[repr(C)]
        struct $ty;
        impl Operation for $ty {}
    };
}
operation!(Sin);
operation!(Add);
operation!(Call);

#[derive(Archive, Serialize)]
#[rkyv(archived = ParameterizedOp)]
struct Parameterized {
    axis: u32,
    enabled: bool,
    tags: Vec<String>,
}

impl Operation for ParameterizedOp {
    fn display(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let tags: Vec<&str> = self.tags.iter().map(|tag| tag.as_str()).collect();
        write!(
            f,
            "test::parameterized<axis: {}, enabled: {}, tags: {:?}>",
            self.axis, self.enabled, tags
        )
    }
}

dialect! {
    dyn Operation, "test" => {
        "sin" => Sin,
        "add" => Add,
        "call" => Call,
        "parameterized" => ParameterizedOp,
    }
}

mod ops {
    pub mod test {
        use super::super::{Add, Call, Parameterized, Sin};
        use gnx_expr::Op;

        pub fn add() -> Op {
            Op::new(Add)
        }
        pub fn call() -> Op {
            Op::new(Call)
        }
        pub fn sin() -> Op {
            Op::new(Sin)
        }
        pub fn parameterized(axis: u32, enabled: bool, tags: &[&str]) -> Op {
            Op::new(Parameterized {
                axis,
                enabled,
                tags: tags.iter().map(|tag| (*tag).to_owned()).collect(),
            })
        }
    }
}

fn info() -> VarInfo {
    VarInfo::new(0usize)
}

// Builds and shows an expression.
fn show(expr: ExprBuilder) -> String {
    expr.build().unwrap().view().to_string()
}

#[test]
fn pretty_expr_uses_alphabetic_names() {
    let mut vars = VarScope::new();
    let x = vars.create_var();
    let y = vars.create_var();
    let z = vars.create_var();
    let expr = ExprBuilder::from_parts(
        vec![],
        vec![],
        vec![(x, info()), (y, info())],
        vec![EqnSource::new(
            ops::test::add(),
            vec![],
            vec![x, y],
            vec![(z, info())],
        )],
        vec![z],
    );
    assert_eq!(
        show(expr),
        "|a: 0, b: 0| => {\n  c: 0 = test::add a b;\n  in (c)\n}"
    );
}

#[test]
fn nested_expr_closure_names_match_outer_names() {
    let inner = expr!(info(); { lambda captured; a: c = test::add captured a; in (c) });
    let outer = expr!(info(); { lambda a; b: c = test::call[inner => (a)] b; in (c) });
    assert_eq!(
        show(outer),
        "|a: 0; b: 0| => {\n  c: 0 = test::call [|b: 0| => {\n    c: 0 = test::add a b;\n    in (c)\n  }] b;\n  in (c)\n}"
    );
}

#[test]
fn inline_expr_captures_values_in_first_use_order() {
    let info = info();
    let outer = expr!(|x: &info, y: &info| => {
        local: &info = test::sin x;
        result: &info = test::call [|x: &info| => {
            z: &info = test::add local x;
            in (z, y, local)
        }] [|| => (x)] x;
        in (result)
    })
    .build()
    .unwrap();
    let outer = outer.view();
    let eqns: Vec<_> = outer.eqns().collect();
    let inputs: Vec<_> = outer.inputs().map(|param| param.var()).collect();
    let regions: Vec<_> = eqns[1].subexprs().collect();
    assert_eq!(regions.len(), 2);
    let local = eqns[0].outputs().next().unwrap().var();
    assert_eq!(regions[0].provided_vars(), [local, inputs[1]]);
    assert_eq!(regions[1].provided_vars(), [inputs[0]]);
    assert!(outer.to_string().contains("[|| => (a)]"));
    let inner = regions[0].expr();
    let closure: Vec<_> = inner.closure_inputs().map(|param| param.var()).collect();
    let inner_input = inner.inputs().next().unwrap().var();
    assert_eq!(
        inner.eqns().next().unwrap().inputs(),
        [closure[0], inner_input]
    );
    assert_eq!(inner.outputs()[1..], [closure[1], closure[0]]);
}

#[test]
fn expr_without_equations_can_return_repeated_values_directly() {
    let body = expr!(|arg: info()| => (arg, arg)).build().unwrap();
    let body = body.view();
    assert_eq!(body.eqns().len(), 0);
    assert_eq!(body.outputs(), [body.inputs().next().unwrap().var(); 2]);
    assert_eq!(body.to_string(), "|a: 0| => (a, a)");
}

#[test]
fn inline_expr_passes_captures_through_intermediate_regions() {
    let outer = expr!(info(); {
        lambda bias, input =>
        result = test::call { lambda x =>
            y = test::call { lambda => z = test::add bias x; in (z) };
            in (y)
        } input;
        in (result)
    })
    .build()
    .unwrap();
    let outer = outer.view();
    let inputs: Vec<_> = outer.inputs().map(|param| param.var()).collect();
    let middle = outer.eqns().next().unwrap().subexprs().next().unwrap();
    assert_eq!(middle.provided_vars(), [inputs[0]]);
    let middle_expr = middle.expr();
    let inner = (middle_expr.eqns().next().unwrap().subexprs().next()).unwrap();
    assert_eq!(
        inner.provided_vars(),
        [
            middle_expr.closure_inputs().next().unwrap().var(),
            middle_expr.inputs().next().unwrap().var()
        ]
    );
    assert_eq!(
        outer.pretty().max_inline_eqns(0).to_string(),
        concat!(
            "a = |a: 0, b: 0;| => {\n  c: 0 = test::add a b;\n  in (c)\n}\n",
            "b = |a: 0; b: 0| => {\n  c: 0 = test::call [@a{a, b}];\n  in (c)\n}\n",
            "in\n|a: 0, b: 0| => {\n  c: 0 = test::call [@b{a}] b;\n  in (c)\n}",
        )
    );
}

#[test]
fn nested_captures_bind_in_declaration_order() {
    let inner = expr!(info(); { lambda borrowed, owned;: in (owned) });
    let outer = expr!(info(); {
        lambda x, y:
        result = test::call[inner => (x, y)];
        in (result)
    })
    .build()
    .unwrap();
    let outer = outer.view();
    let inputs: Vec<_> = outer.inputs().map(|param| param.var()).collect();
    let region = outer.eqns().next().unwrap().subexprs().next().unwrap();
    assert_eq!(region.provided_vars(), [inputs[0], inputs[1]]);
}

#[test]
fn expr_macro_and_printer_preserve_operand_order() {
    let body = expr!(info(); {
        lambda x, y:
        z = test::add x y;
        in (z)
    })
    .build()
    .unwrap();
    let body = body.view();
    let inputs: Vec<_> = body.inputs().map(|param| param.var()).collect();
    assert_eq!(body.eqns().next().unwrap().inputs(), [inputs[0], inputs[1]]);
    assert!(body.to_string().contains("|a: 0, b: 0| =>"));
    assert!(body.to_string().contains("test::add a b"));
}

#[test]
fn expr_macro_passes_operation_arguments_as_rust_expressions() {
    let mut evaluated = Vec::new();
    let body = expr!(info(); {
        lambda x:
        y = test::parameterized<{ evaluated.push(1); 2 },
            { evaluated.push(2); 3 > 2 }, &["one", "two"],> x;
        in (y)
    })
    .build()
    .unwrap();
    assert_eq!(evaluated, [1, 2]);
    let op = body.view().eqns().next().unwrap().op();
    assert!(op == ops::test::parameterized(2, true, &["one", "two"]));
    assert!(op != ops::test::parameterized(2, true, &["one"]));
    assert_eq!(
        op.to_string(),
        r#"test::parameterized<axis: 2, enabled: true, tags: ["one", "two"]>"#
    );
}

#[test]
fn expr_macro_builds_printed_structure() {
    let info = info();
    let expr = expr!({
        lambda x: &info, y: &info =>
        z: info = test::add x y;
        in (z)
    });
    assert_eq!(
        show(expr),
        "|a: 0, b: 0| => {\n  c: 0 = test::add a b;\n  in (c)\n}"
    );
}

#[test]
fn typed_expr_annotations_evaluate_rust_expressions() {
    struct Config {
        info: VarInfo,
    }
    let config = Config {
        info: VarInfo::new(5usize),
    };
    let owned = VarInfo::new(6usize);
    let kept = owned.clone();
    let mut evaluated = 0;
    let expr = expr!({
        lambda bias: &config.info; x: owned =>
        z: { let n = 7; evaluated += 1; VarInfo::new(n) } = test::add bias x;
        in (z)
    });
    assert_eq!(evaluated, 1);
    // The infos are shared with the expression, not copied.
    assert!(expr.value_info(Var::new(0)).unwrap().ptr_eq(&config.info));
    assert!(expr.value_info(Var::new(1)).unwrap().ptr_eq(&kept));
    let root = expr.build().unwrap();
    let view = root.view();
    assert!(view.value_info(view.outputs()[0]).unwrap() == VarInfo::new(7i32));
    assert_eq!(
        view.to_string(),
        "|a: 5; b: 6| => {\n  c: 7 = test::add a b;\n  in (c)\n}"
    );
}

#[test]
fn large_subexprs_can_be_deferred_with_hinted_names() {
    let inner = expr!(info(); { lambda b: c = test::sin b; in (c) }).with_name_hint("foo");
    let outer = expr!(info(); { lambda b: c = test::call[inner => ()] b; in (c) })
        .build()
        .unwrap();
    assert_eq!(
        outer.view().pretty().max_inline_eqns(0).to_string(),
        "foo = |a: 0| => {\n  b: 0 = test::sin a;\n  in (b)\n}\nin\n|a: 0| => {\n  b: 0 = test::call [@foo] a;\n  in (b)\n}"
    );
    let sub = outer
        .view()
        .eqns()
        .next()
        .unwrap()
        .subexprs()
        .next()
        .unwrap();
    assert_eq!(sub.expr().name_hint(), Some("foo"));
    assert_eq!(outer.view().name_hint(), None);
}

#[test]
fn inline_subexpr_indentation_is_configurable() {
    let inner = expr!(info(); { lambda b: c = test::sin b; in (c) });
    let outer = expr!(info(); { lambda b: c = test::call[inner => ()] b; in (c) })
        .build()
        .unwrap();
    assert_eq!(
        outer.view().pretty().indent_width(4).to_string(),
        "|a: 0| => {\n    b: 0 = test::call [|a: 0| => {\n        b: 0 = test::sin a;\n        in (b)\n    }] a;\n    in (b)\n}"
    );
}

#[test]
fn deferred_subexprs_are_deduplicated() {
    let inner = expr!(info(); { lambda b: c = test::sin b; in (c) })
        .with_name_hint("foo")
        .finish();
    let outer = expr!(info(); {
        lambda b:
        c = test::call[inner.clone() => ()] b;
        d = test::call[inner => ()] b;
        in (c, d)
    })
    .build()
    .unwrap();
    assert_eq!(
        outer.view().pretty().max_inline_eqns(0).to_string(),
        "foo = |a: 0| => {\n  b: 0 = test::sin a;\n  in (b)\n}\nin\n|a: 0| => {\n  b: 0 = test::call [@foo] a;\n  c: 0 = test::call [@foo] a;\n  in (b, c)\n}"
    );
}

#[test]
fn deferred_subexprs_share_a_definition_across_capture_bindings() {
    let info = info();
    let outer = expr! {
        foo = |c: &info;| => { out: &info = test::sin c; in (out) }
        in |x: &info, y: &info| => {
            a: &info = test::call [@foo{x}];
            b: &info = test::call [@foo{y}];
            c: &info = test::call [@foo{x}];
            in (a, b, c)
        }
    }
    .build()
    .unwrap();
    let outer = outer.view();
    // One stored body, used three times.
    let bodies: Vec<_> = (outer.eqns())
        .map(|eqn| eqn.subexprs().next().unwrap().expr())
        .collect();
    assert!(std::ptr::eq(
        bodies[0].eqns().next().unwrap().op().get(),
        bodies[1].eqns().next().unwrap().op().get()
    ));
    assert_eq!(
        outer.pretty().max_inline_eqns(0).to_string(),
        concat!(
            "foo = |a: 0;| => {\n  b: 0 = test::sin a;\n  in (b)\n}\n",
            "in\n|a: 0, b: 0| => {\n",
            "  c: 0 = test::call [@foo{a}];\n",
            "  d: 0 = test::call [@foo{b}];\n",
            "  e: 0 = test::call [@foo{a}];\n",
            "  in (c, d, e)\n}",
        ),
    );
}

#[test]
fn named_exprs_can_reference_earlier_definitions() {
    let outer = expr!(info();
        first = || => { y = test::sin; in (y) }
        second = || => { y = test::call [@first]; in (y) }
        in || => { y = test::call [@second]; in (y) }
    )
    .build()
    .unwrap();
    let printed = outer.view().pretty().max_inline_eqns(0).to_string();
    assert!(printed.starts_with("first = |"));
    assert!(printed.contains("\nsecond = |"));
    assert!(printed.contains("test::call [@first];"));
}

#[test]
fn expression_names_are_part_of_identity() {
    let named = |name: Option<&str>| {
        let mut b = ExprBuilder::new();
        let x = b.input(info());
        b.output(x);
        b.set_name_hint(name.map(str::to_owned));
        b.build().unwrap()
    };
    assert_eq!(named(Some("f")).hash(), named(Some("f")).hash());
    assert_ne!(named(Some("f")).hash(), named(Some("g")).hash());
    assert_ne!(named(Some("f")).hash(), named(None).hash());
    let root = named(Some("f"));
    assert_eq!(root.view().name_hint(), Some("f"));
    assert_eq!(root.view().to_builder().name_hint(), Some("f"));
    // Variables have no names of their own.
    assert_eq!(root.view().to_string(), "|a: 0| => (a)");
}

#[test]
fn holes_discard_results() {
    let info = info();
    let expr = expr!(|x: &info| => {
        y: &info, _: &info = test::add x x;
        _: &info = test::sin y;
        in (y)
    });
    let root = expr.build().unwrap();
    let view = root.view();
    assert_eq!(
        view.to_string(),
        "|a: 0| => {\n  b: 0, _: 0 = test::add a a;\n  _: 0 = test::sin b;\n  in (b)\n}"
    );
    let eqns: Vec<_> = view.eqns().collect();
    assert!(eqns[0].outputs().nth(1).unwrap().var().is_hole());
    assert_eq!(view.value_info(Var::hole()), None);

    // An equation that only discards is dead.
    let pruned = gnx_expr::pass::dce(view).build().unwrap();
    assert_eq!(pruned.view().eqns().len(), 1);
    // Holes survive a rewrite.
    let copy = gnx_expr::pass::rewrite::<(), _>(view, &mut |_, _| Ok(None))
        .unwrap()
        .build()
        .unwrap();
    assert_eq!(copy.view().to_string(), view.to_string());

    // A hole cannot be used, returned, or stand for a parameter.
    let hole = Var::hole();
    let mut b = ExprBuilder::new();
    b.output(hole);
    assert!(matches!(b.build(), Err(BuildError::Hole)));
    let mut b = ExprBuilder::new();
    b.eqn(ops::test::sin()).input(hole);
    assert!(matches!(b.build(), Err(BuildError::Hole)));
    let b = ExprBuilder::from_parts(vec![], vec![], vec![(hole, info.clone())], vec![], vec![]);
    assert!(matches!(b.build(), Err(BuildError::Hole)));
}

#[test]
fn hexpr_macro_builds_cells() {
    let info = info();
    let root = hexpr!(|captured[:]: &info; value: &info| => {
        local[:] = value;
        captured[:] <- value;
        result <- captured[:];
        doubled: &info, _: &info = test::add result value;
        in (doubled, local[:])
    })
    .build()
    .unwrap();
    let body = root.view();
    let instrs: Vec<_> = body.instrs().collect();
    assert!(matches!(instrs[0], HInstr::CreateCell { .. }));
    assert!(matches!(instrs[1], HInstr::WriteCell { .. }));
    assert!(matches!(instrs[2], HInstr::ReadCell { .. }));
    assert_eq!(body.closure_inputs().next().unwrap().kind(), Kind::Cell);
    assert_eq!(
        body.to_string(),
        concat!(
            "|a[:]: 0; b: 0| => {\n",
            "  c[:] = b;\n",
            "  a[:] <- b;\n",
            "  d <- a[:];\n",
            "  e: 0, _: 0 = test::add d b;\n",
            "  in (e, c[:])\n}",
        )
    );

    // Inline regions refer to enclosing cells directly.
    let root = hexpr!(|state[:]: &info, x: &info| => {
        y: &info = test::call [|v: &info| => {
            current <- state[:];
            next: &info = test::add current v;
            state[:] <- next;
            in (next)
        }] x;
        in (y)
    })
    .build()
    .unwrap();
    let call = root.view().instrs().next().unwrap().as_apply().unwrap();
    let region = call.subexprs().next().unwrap();
    assert_eq!(region.provided_vars().len(), 1);
    assert_eq!(
        region.expr().closure_inputs().next().unwrap().kind(),
        Kind::Cell
    );
    assert_eq!(
        root.view().to_string(),
        concat!(
            "|a[:]: 0, b: 0| => {\n",
            "  c: 0 = test::call [|b: 0| => {\n",
            "    c <- a[:];\n",
            "    d: 0 = test::add c b;\n",
            "    a[:] <- d;\n",
            "    in (d)\n",
            "  }] b;\n",
            "  in (c)\n}",
        )
    );
}

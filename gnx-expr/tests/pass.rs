use std::sync::Arc;

use gnx_expr::pass::{Builder, CheckSupported, Dce, FnPass, Pass, Pipeline, dce, rewrite};
use gnx_expr::{
    EqnSource, ExprBuilder, Op, Operation, RootExpr, SubExprBuilder, VarInfo, VarScope, dialect,
    register_op_interface,
};
use rkyv::{Archive, Portable, Serialize};

#[derive(Archive, Serialize, Portable)]
#[rkyv(as = Self)]
#[repr(C)]
struct Neg;
impl Operation for Neg {}

#[derive(Archive, Serialize, Portable)]
#[rkyv(as = Self)]
#[repr(C)]
struct Call;
impl Operation for Call {}

dialect! {
    dyn Operation, "pass_test" => {
        "neg" => Neg,
        "call" => Call,
    }
}

trait Supported: Sync {}
impl Supported for Neg {}
register_op_interface!(dyn Supported: Neg);

fn info() -> VarInfo {
    VarInfo::new(0usize)
}
fn neg() -> Op {
    Op::new(Neg)
}

/// call[|c| { d = neg c; e = neg d (dead) } -> d] captures (a, b); b is unused.
fn program() -> Arc<RootExpr> {
    let mut body = Builder::new();
    let [c, _unused] = [body.closure(info()), body.closure(info())];
    let d = body.emit1(neg(), &[c], info());
    body.emit1(neg(), &[d], info());
    let body = body.finish(vec![d]);

    let mut b = Builder::new();
    let x = b.input(info());
    let a = b.emit1(neg(), &[x], info());
    let unused_capture = b.emit1(neg(), &[x], info());
    b.emit1(neg(), &[a], info()); // dead
    let called = b.emit(
        Op::new(Call),
        &[],
        [info()],
        vec![SubExprBuilder::new(body).with_vars([a, unused_capture])],
    );
    b.finish(vec![called[0], x]).build().unwrap()
}

#[test]
fn builder_produces_checked_ssa() {
    let expr = program();
    expr.view().check().unwrap();
    assert_eq!(expr.view().eqns().len(), 4);
}

#[test]
fn dce_drops_dead_equations_and_unused_captures() {
    let pruned = dce(program().view()).build().unwrap();
    let pruned = pruned.view();
    pruned.check().unwrap();
    // `unused_capture` became dead once the region stopped capturing it.
    assert_eq!(pruned.eqns().len(), 2);
    let region = pruned.eqns().nth(1).unwrap().subexprs().next().unwrap();
    assert_eq!(region.provided_vars().len(), 1);
    assert_eq!(region.expr().closure_inputs().len(), 1);
    assert_eq!(region.expr().eqns().len(), 1);
}

#[test]
fn rewrite_replaces_equations_bottom_up() {
    // Replace every `neg` with two `neg`s; nested bodies are rewritten too.
    let expr = program();
    let mut visited = Vec::new();
    let doubled = rewrite::<(), _>(expr.view(), &mut |b, eqn| {
        visited.push(eqn.op().name());
        if eqn.op().name() != "neg" {
            return Ok(None);
        }
        let once = b.emit1(neg(), eqn.inputs(), b.info(eqn.inputs()[0]).clone());
        Ok(Some(vec![b.emit1(neg(), &[once], info())]))
    })
    .unwrap()
    .build()
    .unwrap();
    let doubled = doubled.view();
    doubled.check().unwrap();
    // Body equations are visited before the call that owns them.
    assert_eq!(visited, ["neg", "neg", "neg", "neg", "neg", "call"]);
    assert_eq!(doubled.eqns().len(), 7);
    let call = doubled.eqns().nth(6).unwrap();
    assert_eq!(call.subexprs().next().unwrap().expr().eqns().len(), 4);
}

#[test]
fn inline_binds_captures_and_inputs() {
    let expr = program();
    let mut b = Builder::new();
    let x = b.input(info());
    let results = b.inline(expr.view(), &[], &[], &[x]).unwrap();
    assert_eq!(results[1], x);
    let inlined = b.finish(results).build().unwrap();
    inlined.view().check().unwrap();
    assert_eq!(inlined.view().eqns().len(), 4);
}

struct Breaks;
impl Pass for Breaks {
    fn name(&self) -> &'static str {
        "breaks"
    }
    fn run(&self, _: Arc<RootExpr>) -> Result<Arc<RootExpr>, gnx_expr::pass::PassError> {
        // Redefines its input: invalid SSA.
        let mut vars = VarScope::new();
        let x = vars.create_var();
        ExprBuilder::from_parts(
            vec![],
            vec![],
            vec![(x, info())],
            vec![EqnSource::new(neg(), vec![], vec![x], vec![(x, info())])],
            vec![x],
        )
        .build()
        .map_err(|error| gnx_expr::pass::PassError::new("breaks", error))
    }
}

// Changes the signature: returns one result instead of two.
struct Narrows;
impl Pass for Narrows {
    fn name(&self) -> &'static str {
        "narrows"
    }
    fn run(&self, expr: Arc<RootExpr>) -> Result<Arc<RootExpr>, gnx_expr::pass::PassError> {
        let mut b = Builder::new();
        let x = b.input(info());
        let _ = expr;
        Ok(b.finish(vec![x]).build().unwrap())
    }
}

#[test]
fn finalized_pipeline_produces_another_representation() {
    let count: Pipeline<usize> = Pipeline::new()
        .with(Dce)
        .finalize(FnPass::new("count", |expr: Arc<RootExpr>| {
            Ok::<_, std::convert::Infallible>(expr.view().eqns().len())
        }));
    assert_eq!(count.passes().len(), 1);
    assert_eq!(count.run(program()).unwrap(), 2);
    assert_eq!(count.optimize(program()).unwrap().view().eqns().len(), 2);
}

#[test]
fn pipeline_verifies_each_pass_and_checks_support() {
    let ok = Pipeline::new().with(Dce).run(program()).unwrap();
    assert_eq!(ok.view().eqns().len(), 2);

    let error = Pipeline::new().with(Breaks).run(program()).unwrap_err();
    assert_eq!(error.pass, "breaks");

    let error = Pipeline::new().with(Narrows).run(program()).unwrap_err();
    assert_eq!(error.pass, "narrows");
    assert!(error.to_string().contains("signature"), "{error}");

    let error = Pipeline::new()
        .with(Dce)
        .with(CheckSupported::<dyn Supported>::new())
        .run(program())
        .unwrap_err();
    assert_eq!(error.pass, "check_supported");
    assert!(error.to_string().contains("pass_test.call"), "{error}");
}

#[test]
fn expressions_print() {
    let printed = program().view().to_string();
    assert_eq!(
        printed,
        concat!(
            "|a: 0| => {\n",
            "  b: 0 = pass_test::neg a;\n",
            "  c: 0 = pass_test::neg a;\n",
            "  d: 0 = pass_test::neg b;\n",
            "  e: 0 = pass_test::call [|| => {\n",
            "    a: 0 = pass_test::neg b;\n",
            "    d: 0 = pass_test::neg a;\n",
            "    in (a)\n",
            "  }];\n",
            "  in (e, a)\n",
            "}",
        ),
        "{printed}"
    );
}

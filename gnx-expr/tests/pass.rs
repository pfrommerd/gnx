use gnx_expr::pass::{Builder, CheckSupported, Dce, FnPass, Pass, Pipeline, dce, rewrite};
use gnx_expr::{
    Dialect, Eqn, Expr, Op, Operation, SubExpr, ValueInfo, VarScope, register_op_interface,
};

struct TestDialect;
static DIALECT: TestDialect = TestDialect;
impl Dialect for TestDialect {
    fn name(&self) -> &'static str {
        "pass_test"
    }
}
#[derive(Clone, PartialEq, Eq, Hash)]
struct Neg;
#[derive(Clone, PartialEq, Eq, Hash)]
struct Call;
impl std::fmt::Display for Neg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}::{}", Operation::dialect(self).name(), Operation::name(self))
    }
}

impl Operation for Neg {
    fn name(&self) -> &'static str {
        "neg"
    }
    fn dialect(&self) -> &'static dyn Dialect {
        &DIALECT
    }
}
impl std::fmt::Display for Call {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}::{}", Operation::dialect(self).name(), Operation::name(self))
    }
}

impl Operation for Call {
    fn name(&self) -> &'static str {
        "call"
    }
    fn dialect(&self) -> &'static dyn Dialect {
        &DIALECT
    }
}

trait Supported: Sync {}
impl Supported for Neg {}
register_op_interface!(dyn Supported: Neg);

fn info() -> ValueInfo {
    ValueInfo::new(0usize)
}
fn neg() -> Op {
    Op::new(Neg)
}

/// call[|c| { d = neg c; e = neg d (dead) } -> d] captures (a, b); b is unused.
fn program() -> Expr {
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
        vec![SubExpr::new(body.into(), vec![a, unused_capture])],
    );
    b.finish(vec![called[0], x])
}

#[test]
fn builder_produces_checked_ssa() {
    let expr = program();
    expr.check().unwrap();
    assert_eq!(expr.eqns().len(), 4);
}

#[test]
fn dce_drops_dead_equations_and_unused_captures() {
    let pruned = dce(&program());
    pruned.check().unwrap();
    // `unused_capture` became dead once the region stopped capturing it.
    assert_eq!(pruned.eqns().len(), 2);
    let region = &pruned.eqns()[1].subexprs()[0];
    assert_eq!(region.closure().len(), 1);
    assert_eq!(region.expr().closure().len(), 1);
    assert_eq!(region.expr().eqns().len(), 1);
}

#[test]
fn rewrite_replaces_equations_bottom_up() {
    // Replace every `neg` with two `neg`s; nested bodies are rewritten too.
    let expr = program();
    let mut visited = Vec::new();
    let doubled = rewrite::<(), _>(&expr, &mut |b, eqn| {
        visited.push(eqn.op().name());
        if eqn.op().name() != "neg" {
            return Ok(None);
        }
        let once = b.emit1(neg(), eqn.inputs(), b.info(eqn.inputs()[0]).clone());
        Ok(Some(vec![b.emit1(neg(), &[once], info())]))
    })
    .unwrap();
    doubled.check().unwrap();
    // Body equations are visited before the call that owns them.
    assert_eq!(visited, ["neg", "neg", "neg", "neg", "neg", "call"]);
    assert_eq!(doubled.eqns().len(), 7);
    assert_eq!(doubled.eqns()[6].subexprs()[0].expr().eqns().len(), 4);
}

#[test]
fn inline_binds_captures_and_inputs() {
    let expr = program();
    let mut b = Builder::new();
    let x = b.input(info());
    let results = b.inline(&expr, &[], &[x]);
    assert_eq!(results[1], x);
    b.finish(results).check().unwrap();
}

struct Breaks;
impl Pass for Breaks {
    fn name(&self) -> &'static str {
        "breaks"
    }
    fn run(&self, _: Expr) -> Result<Expr, gnx_expr::pass::PassError> {
        // Redefines its input: invalid SSA.
        let mut vars = VarScope::new();
        let x = vars.create_var();
        Ok(Expr::new(
            vec![],
            vec![(x, info())],
            vec![Eqn::new(neg(), vec![x], vec![(x, info())])],
            vec![x],
        ))
    }
}

#[test]
fn finalized_pipeline_produces_another_representation() {
    let count: Pipeline<usize> = Pipeline::new()
        .with(Dce)
        .finalize(FnPass::new("count", |expr: Expr| {
            Ok::<_, std::convert::Infallible>(expr.eqns().len())
        }));
    assert_eq!(count.passes().len(), 1);
    assert_eq!(count.run(program()).unwrap(), 2);
    assert_eq!(count.optimize(program()).unwrap().eqns().len(), 2);
}

#[test]
fn pipeline_verifies_each_pass_and_checks_support() {
    let ok = Pipeline::new().with(Dce).run(program()).unwrap();
    assert_eq!(ok.eqns().len(), 2);

    let error = Pipeline::new().with(Breaks).run(program()).unwrap_err();
    assert_eq!(error.pass, "breaks");

    let error = Pipeline::new()
        .with(Dce)
        .with(CheckSupported::<dyn Supported>::new())
        .run(program())
        .unwrap_err();
    assert_eq!(error.pass, "check_supported");
    assert!(error.to_string().contains("pass_test.call"), "{error}");
}

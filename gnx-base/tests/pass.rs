use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use futures_executor::block_on;
use gnx_base::cache::Cache;
use gnx_base::store::Store;

use gnx_base::pass::{
    Builder, CheckSupported, Dce, Pass, PassError, Pipeline, PipelineError, dce, rewrite,
};
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

macro_rules! unhashed {
    ($($pass:ty),*) => {
        $(impl gnx_base::anyhash::Hash for $pass {
            fn hash<H: gnx_base::anyhash::HasherWrite>(&self, _: &mut H) {}
        })*
    };
}

struct Breaks;
impl Pass for Breaks {
    fn name(&self) -> &'static str {
        "breaks"
    }
    fn run(&self, _: Arc<RootExpr>) -> Result<Arc<RootExpr>, PassError> {
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
        .map_err(|error| PassError::new("breaks", error))
    }
}

// Changes the signature: returns one result instead of two.
struct Narrows;
impl Pass for Narrows {
    fn name(&self) -> &'static str {
        "narrows"
    }
    fn run(&self, expr: Arc<RootExpr>) -> Result<Arc<RootExpr>, PassError> {
        let mut b = Builder::new();
        let x = b.input(info());
        let _ = expr;
        Ok(b.finish(vec![x]).build().unwrap())
    }
}

// The number of equations, as another representation of an expression.
struct Count;
impl Pass<u32> for Count {
    fn name(&self) -> &'static str {
        "count"
    }
    fn run(&self, expr: Arc<RootExpr>) -> Result<u32, PassError> {
        Ok(expr.view().eqns().len() as u32)
    }
}

/// A pass that leaves the expression as it is, and counts its runs.
#[derive(Clone, Default)]
struct Counted(&'static str, Arc<AtomicUsize>);
impl Counted {
    fn runs(&self) -> usize {
        self.1.load(Ordering::Relaxed)
    }
}
impl Pass for Counted {
    fn name(&self) -> &'static str {
        self.0
    }
    fn run(&self, expr: Arc<RootExpr>) -> Result<Arc<RootExpr>, PassError> {
        self.1.fetch_add(1, Ordering::Relaxed);
        Ok(expr)
    }
}

unhashed!(Breaks, Narrows, Count, Counted);

fn pass_error(error: PipelineError) -> PassError {
    match error {
        PipelineError::Pass(error) => error,
        error => panic!("expected a pass error: {error}"),
    }
}

#[test]
fn finalized_pipeline_produces_another_representation() {
    let cache = Cache::in_memory();
    let count = Pipeline::new().with(Dce).finalize(Count);
    assert_eq!(count.passes().len(), 1);
    assert_eq!(*block_on(count.run(program(), &cache)).unwrap(), 2);
    let optimized = block_on(count.optimize(program(), &cache)).unwrap();
    assert_eq!(optimized.view().eqns().len(), 2);
}

#[test]
fn pipeline_verifies_each_pass_and_checks_support() {
    let cache = Cache::in_memory();
    let run = |pipeline: Pipeline| block_on(pipeline.run(program(), &cache));
    let ok = run(Pipeline::new().with(Dce)).unwrap();
    assert_eq!(ok.view().eqns().len(), 2);

    let error = pass_error(run(Pipeline::new().with(Breaks)).unwrap_err());
    assert_eq!(error.pass, "breaks");

    let error = pass_error(run(Pipeline::new().with(Narrows)).unwrap_err());
    assert_eq!(error.pass, "narrows");
    assert!(error.to_string().contains("signature"), "{error}");

    let unsupported = Pipeline::new()
        .with(Dce)
        .with(CheckSupported::<dyn Supported>::new());
    let error = pass_error(run(unsupported).unwrap_err());
    assert_eq!(error.pass, "check_supported");
    assert!(error.to_string().contains("pass_test.call"), "{error}");
}

#[test]
fn cached_stages_do_not_run_again() {
    let cache = Cache::in_memory();
    let (first, second, last) = (
        Counted("first", Arc::default()),
        Counted("second", Arc::default()),
        Counted("last", Arc::default()),
    );
    let pipeline = Pipeline::new()
        .with(first.clone())
        .with(Dce)
        .with(second.clone());
    let cold = block_on(pipeline.run(program(), &cache)).unwrap();
    assert_eq!((first.runs(), second.runs()), (1, 1));

    // The same pipeline on the same expression runs nothing.
    let warm = block_on(pipeline.run(program(), &cache)).unwrap();
    assert_eq!((first.runs(), second.runs()), (1, 1));
    assert_eq!(warm.hash(), cold.hash());

    // A pipeline that starts the same way only runs what comes after.
    let longer = Pipeline::new()
        .with(first.clone())
        .with(Dce)
        .with(second.clone())
        .with(last.clone());
    block_on(longer.run(program(), &cache)).unwrap();
    assert_eq!((first.runs(), second.runs(), last.runs()), (1, 1, 1));

    // One that differs earlier shares only what comes before that.
    let other = Pipeline::new().with(first.clone()).with(second.clone());
    block_on(other.run(program(), &cache)).unwrap();
    assert_eq!((first.runs(), second.runs()), (1, 2));

    // Nothing is shared with another cache.
    block_on(pipeline.run(program(), &Cache::in_memory())).unwrap();
    assert_eq!((first.runs(), second.runs()), (2, 3));

    // A final stage is cached like any other.
    let counted = Pipeline::new().with(first.clone()).finalize(Count);
    assert_eq!(*block_on(counted.run(program(), &cache)).unwrap(), 4);
    let before = first.runs();
    assert_eq!(*block_on(counted.run(program(), &cache)).unwrap(), 4);
    assert_eq!(first.runs(), before);
}

#[test]
fn cache_failures_are_errors() {
    // Blobs cannot be written where a file is in the way of their directory.
    let dir = tempfile::tempdir().unwrap();
    let cache = Cache::new(Store::local(dir.path()).unwrap());
    std::fs::write(dir.path().join("blobs"), b"").unwrap();
    let error = block_on(Pipeline::new().with(Dce).run(program(), &cache)).unwrap_err();
    assert!(matches!(error, PipelineError::Cache(_)), "{error}");
}

#[test]
fn holes_survive_dce_and_rewrite() {
    let hole = gnx_expr::Var::hole();
    let mut b = Builder::new();
    let x = b.input(info());
    let y = b.fresh(info());
    b.push(EqnSource::new(
        neg(),
        vec![],
        vec![x],
        vec![(y, info()), (hole, info())],
    ));
    b.push(EqnSource::new(neg(), vec![], vec![y], vec![(hole, info())]));
    let root = b.finish(vec![y]).build().unwrap();
    let view = root.view();

    // An equation that only discards is dead.
    let pruned = dce(view).build().unwrap();
    assert_eq!(pruned.view().eqns().len(), 1);
    // Holes survive a rewrite.
    let copy = rewrite::<(), _>(view, &mut |_, _| Ok(None))
        .unwrap()
        .build()
        .unwrap();
    assert_eq!(copy.view().to_string(), view.to_string());
}

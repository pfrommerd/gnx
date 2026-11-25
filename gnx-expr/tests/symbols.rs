use gnx_expr::lower::LoweringContext;
use gnx_expr::pass::Builder;
use gnx_expr::{
    Capture, CaptureError, HEqn, HExprBuilder, HRootExpr, Info, InputId, InputSpec, Invocation,
    Kind, LoweringError, ObjectTracer, Op, Operation, Symbol, Trace, TraceContext, VarInfo,
    dialect,
};
use rkyv::{Archive, Portable, Serialize};

#[derive(Archive, Serialize, Portable)]
#[rkyv(as = Self)]
#[repr(C)]
struct Compute;
impl Operation for Compute {}

#[derive(Archive, Serialize, Portable)]
#[rkyv(as = Self)]
#[repr(C)]
struct Call;
impl Operation for Call {
    fn lower(&self, eqn: HEqn<'_>, ctx: &mut LoweringContext<'_>) -> Result<(), LoweringError> {
        let sub = eqn.subexprs().next().ok_or(LoweringError::CallArity)?;
        let region = ctx.lower_region(sub, eqn.inputs())?;
        let physical = ctx.allocate_results(region.output_infos());
        ctx.bind_region_results(eqn.outputs(), &region, &physical)?;
        ctx.apply_region_updates(&region, &physical)?;
        ctx.emit(
            eqn.op().to_owned(),
            region.inputs(),
            physical,
            vec![region.subexpr()],
        );
        Ok(())
    }
}

dialect! {
    dyn Operation, "symtest" => {
        "compute" => Compute,
        "call" => Call,
    }
}

// An info with one symbol slot
#[derive(Archive, Serialize, Portable)]
#[rkyv(as = Self)]
#[repr(transparent)]
struct Dim(Symbol);
impl Info for Dim {
    fn symbols(&self) -> &[Symbol] {
        std::slice::from_ref(&self.0)
    }
    fn remap_symbols(&self, f: &mut dyn FnMut(Symbol) -> Symbol) -> Option<VarInfo> {
        Some(dim(f(self.0)))
    }
    fn display(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "dim({})", self.0)
    }
}

dialect! {
    dyn Info, "symtest" => { "dim" => Dim }
}

fn dim(s: Symbol) -> VarInfo {
    VarInfo::new(Dim(s))
}
fn compute() -> Op {
    Op::new(Compute)
}
fn apply(inputs: Vec<Trace>, output: VarInfo) -> Trace {
    Invocation::invoke(compute(), vec![], inputs, vec![output])
        .pop()
        .unwrap()
        .into_trace_ref()
}

#[test]
fn infos_remap_their_symbols() {
    let (s, t) = (Symbol::new(0), Symbol::new(5));
    let info = dim(s);
    assert_eq!(info.symbols(), [s]);
    assert_eq!(info.to_string(), "dim(s0)");
    let remapped = info.remap_symbols(|_| t);
    assert_eq!(remapped.symbols(), [t]);
    assert!(remapped == dim(t));
    assert!(remapped != info);
    assert!(info.downcast::<Dim>().is_some());
}

#[test]
fn captures_declare_the_symbols_they_take_from_their_context() {
    let outer = TraceContext::enter();
    let inherited = outer.context().create_symbol();
    let from_outside = Trace::placeholder(dim(inherited));

    let guard = TraceContext::enter();
    let ctx = guard.context();
    let local = ctx.create_symbol();
    assert!(ctx.has_symbol(inherited) && ctx.has_symbol(local));
    let x = Trace::placeholder(dim(local));
    let y = apply(vec![x.clone(), from_outside.clone()], dim(local));
    let capture =
        Capture::from_context(ctx, &[ObjectTracer::from(x)], &[ObjectTracer::from(y)]).unwrap();
    let expr = capture.expr();
    // The input binds its symbol; the imported value's symbol is provided
    // by the enclosing expression.
    assert_eq!(expr.input_symbols(), [local]);
    assert_eq!(expr.closure_symbols(), [inherited]);
    assert_eq!(expr.closure_inputs().len(), 1);
    assert!(expr.declares_symbol(local) && expr.declares_symbol(inherited));

    // A symbol of this context that no input binds cannot be captured.
    let unbound = ctx.create_symbol();
    let z = apply(vec![], dim(unbound));
    assert_eq!(
        Capture::from_context(ctx, &[], &[ObjectTracer::from(z)]).err(),
        Some(CaptureError::UnboundSymbolInput(unbound))
    );
    drop(guard);

    // A symbol of no context at all is undefined.
    let stray = Symbol::new(1000);
    let w = apply(vec![], dim(stray));
    assert_eq!(
        Capture::from_context(outer.context(), &[], &[ObjectTracer::from(w)]).err(),
        Some(CaptureError::UndefinedSymbol(stray))
    );
}

#[test]
fn regions_are_lowered_in_the_symbols_of_their_caller() {
    // The body binds a symbol of its own through its input.
    let mut body = HExprBuilder::new();
    let s = body.symbol();
    let v = body.input(Kind::Value, dim(s));
    let [r] = (body.eqn(compute()).input(v))
        .output(Kind::Value, dim(s))
        .done();
    body.output(r);

    let mut b = HRootExpr::builder();
    b.symbol(); // so that the caller's symbol differs from the body's
    let t = b.symbol();
    assert_ne!(s, t);
    let x = b.input(Kind::Value, dim(t));
    let mut e = b.eqn(Op::new(Call));
    e.input(x).output(Kind::Value, dim(t));
    e.subexpr(body);
    let [y] = e.done();
    b.output(y);
    let root = b.build().unwrap();

    let spec = InputSpec {
        closure: vec![],
        inputs: vec![(InputId::Input(0), Kind::Value, dim(t))],
    };
    let lowered = root.view().lower(&spec).unwrap();
    let expr = lowered.expr.view();
    assert_eq!(expr.input_symbols(), [t]);
    let call = expr.eqns().next().unwrap();
    // The result of the call is typed in the symbols of the caller,
    assert_eq!(call.outputs().next().unwrap().symbols(), [t]);
    // while the body keeps its own.
    let lowered_body = call.subexprs().next().unwrap().expr();
    assert_eq!(lowered_body.input_symbols(), [s]);

    // A result whose info does not match what the body returns is rejected.
    let mut body = HExprBuilder::new();
    let s = body.symbol();
    let v = body.input(Kind::Value, dim(s));
    body.output(v);
    let mut b = HRootExpr::builder();
    let t = b.symbol();
    let u = b.symbol();
    let x = b.input(Kind::Value, dim(t));
    b.input(Kind::Value, dim(u));
    let mut e = b.eqn(Op::new(Call));
    e.input(x).output(Kind::Value, dim(u));
    e.subexpr(body);
    let root = b.build().unwrap();
    let spec = InputSpec {
        closure: vec![],
        inputs: vec![
            (InputId::Input(0), Kind::Value, dim(t)),
            (InputId::Input(1), Kind::Value, dim(u)),
        ],
    };
    assert_eq!(
        root.view().lower(&spec).unwrap_err(),
        LoweringError::RegionSignature("call result metadata mismatch")
    );
}

#[test]
fn builder_regions_and_inlining_bind_symbols() {
    let mut b = Builder::new();
    let s = b.symbol_input();
    let x = b.input(dim(s));
    // The region captures a value whose info uses a symbol of the parent.
    let region = b
        .region::<()>(&[x], &[], |child, closure, _| {
            Ok(vec![child.emit1(compute(), closure, dim(s))])
        })
        .unwrap();
    assert_eq!(region.provided_symbols(), [s]);
    assert_eq!(region.provided_vars(), [x]);
    let called = b.emit(Op::new(Call), &[], [dim(s)], vec![region]);
    let outer = b.finish(called).build().unwrap();
    let body = (outer.view().eqns().next().unwrap().subexprs().next())
        .unwrap()
        .expr();
    assert_eq!(body.closure_symbols(), [s]);

    // Inlining an expression renames the symbols its inputs bind.
    let mut callee = Builder::new();
    let cs = callee.symbol_input();
    let a = callee.input(dim(cs));
    let out = callee.emit1(compute(), &[a], dim(cs));
    let callee = callee.finish(vec![out]).build().unwrap();

    let mut b = Builder::new();
    b.symbol_input();
    let t = b.symbol_input();
    let x = b.input(dim(t));
    let results = b.inline(callee.view(), &[], &[], &[x]).unwrap();
    assert_eq!(b.info(results[0]).symbols(), [t]);
    b.finish(results).build().unwrap();
}

use std::sync::Arc;

use gnx_expr::{
    BuildError, ExprBuilder, ExprRef, Info, Op, Operation, RootExpr, Symbol, Var, VarInfo, dialect,
};
use rkyv::rend::u32_le;
use rkyv::{Archive, Portable, Serialize};

// ---- a small dialect ----

#[derive(Archive, Serialize, Portable)]
#[rkyv(as = Self)]
#[repr(transparent)]
struct Scale(u32_le);
impl Operation for Scale {}

#[derive(Archive, Serialize, Portable)]
#[rkyv(as = Self)]
#[repr(transparent)]
struct Call(u32_le);
impl Operation for Call {}

dialect! {
    dyn Operation, "test" => {
        "scale" => Scale,
        "call" => Call,
    }
}

// An info without symbols
#[derive(Archive, Serialize, Portable)]
#[rkyv(as = Self)]
#[repr(transparent)]
struct Scalar(u32_le);
impl Info for Scalar {}

// An info with one symbol
#[derive(Archive, Serialize, Portable)]
#[rkyv(as = Self)]
#[repr(transparent)]
struct Dim(Symbol);
impl Info for Dim {
    fn symbols(&self) -> &[Symbol] {
        std::slice::from_ref(&self.0)
    }
}

dialect! {
    dyn Info, "test" => {
        "scalar" => Scalar,
        "dim" => Dim,
    }
}

fn scalar() -> VarInfo {
    VarInfo::new(Scalar(0.into()))
}
fn scale(by: u32) -> Op {
    Op::new(Scale(by.into()))
}
fn call() -> Op {
    Op::new(Call(0.into()))
}

// closure: (s; c: dim(s)), inputs: (x), returns scale(x), c
fn child() -> ExprBuilder {
    let mut b = ExprBuilder::new();
    let s = b.closure_symbol();
    let c = b.closure_input(VarInfo::new(Dim(s)));
    let x = b.input(scalar());
    let [y] = b.eqn(scale(3)).input(x).output(scalar()).done();
    b.outputs([y, c]);
    b
}

fn names(names: impl Iterator<Item = gnx_expr::TypeName>) -> Vec<String> {
    names.map(|n| n.to_string()).collect()
}

#[test]
fn round_trip() {
    let mut b = RootExpr::builder();
    let s = b.symbol();
    let x = b.input(VarInfo::new(Dim(s)));
    let y = b
        .eqn(scale(2))
        .input(x)
        .outputs([VarInfo::new(Dim(s)), scalar()])
        .done::<2>();
    let child = child().finish();
    // The same child three times, in two equations
    let mut e = b.eqn(call());
    e.inputs(y);
    e.subexpr(child.clone()).symbols([s]).vars([x]);
    e.output(scalar());
    let [z] = e.done();
    let mut e = b.eqn(call());
    e.subexpr(child.clone()).symbols([s]).vars([x]);
    e.subexpr(child).symbols([s]).vars([x]);
    let [] = e.done();
    b.outputs([z, x, z]);
    let root = b.build().unwrap();

    let expr = root.view();
    assert!(expr.closure_symbols().is_empty());
    assert_eq!(expr.closure_inputs().len(), 0);
    let inputs: Vec<_> = expr.inputs().collect();
    assert_eq!(inputs.len(), 1);
    assert_eq!(inputs[0].var(), x);
    assert_eq!(inputs[0].symbols(), [s]);
    assert_eq!(inputs[0].type_name().to_string(), "test::dim");
    assert_eq!(expr.outputs(), [z, x, z]);

    let eqns: Vec<_> = expr.eqns().collect();
    assert_eq!(eqns.len(), 3);
    assert_eq!(
        names(eqns.iter().map(|e| e.op().type_name())),
        ["test::scale", "test::call", "test::call"]
    );
    assert_eq!(eqns[0].inputs(), [x]);
    let outputs: Vec<Var> = eqns[0].outputs().map(|p| p.var()).collect();
    assert_eq!(outputs, y);
    assert_eq!(
        names(eqns[0].outputs().map(|p| p.type_name())),
        ["test::dim", "test::scalar"]
    );
    assert_eq!(eqns[0].outputs().next().unwrap().info().symbols(), [s]);
    assert_eq!(eqns[0].subexprs().len(), 0);
    assert_eq!(eqns[1].inputs(), y);
    assert_eq!(eqns[2].subexprs().len(), 2);

    // Variables and symbols of the child are local to it.
    let sub = eqns[1].subexprs().next().unwrap();
    assert!(sub.external().is_none());
    assert_eq!(sub.provided_symbols(), [s]);
    assert_eq!(sub.provided_vars(), [x]);
    let child = sub.expr();
    assert_eq!(child.closure_symbols(), [Symbol::new(0)]);
    let closure: Vec<_> = child.closure_inputs().collect();
    assert_eq!(closure[0].var(), Var::new(0));
    assert_eq!(closure[0].symbols(), [Symbol::new(0)]);
    assert_eq!(child.inputs().next().unwrap().var(), Var::new(1));
    assert_eq!(child.outputs(), [Var::new(2), Var::new(0)]);
    let eqn = child.eqns().next().unwrap();
    assert_eq!(format!("{:?}", eqn.op()), "test::scale");
    // Deref reaches the operation itself.
    let _: &dyn Operation = &*eqn.op();

    // Debug output does not need the implementations to be Debug.
    let debug = format!("{root:?}");
    assert!(debug.contains("test::call"), "{debug}");
}

#[test]
#[should_panic(expected = "the equation has 2 outputs")]
fn done_checks_the_number_of_outputs() {
    let mut b = RootExpr::builder();
    let [_] = b.eqn(scale(1)).outputs([scalar(), scalar()]).done();
}

fn uses(child: ExprRef) -> Arc<RootExpr> {
    let mut b = RootExpr::builder();
    let s = b.symbol();
    let x = b.input(VarInfo::new(Dim(s)));
    let mut e = b.eqn(call());
    e.subexpr(child.clone()).symbols([s]).vars([x]);
    e.subexpr(child).symbols([s]).vars([x]);
    b.build().unwrap()
}

#[test]
fn hashes_identify_roots() {
    let a = uses(child().finish());
    let b = uses(child().finish());
    assert_eq!(a.hash(), b.hash());
    assert!(a == b);

    let mut other = child();
    other.eqn(scale(1));
    let c = uses(other.finish());
    assert_ne!(a.hash(), c.hash());
}

#[test]
fn external_exprs() {
    let lib = child().build().unwrap();
    let handle = lib.clone().handle();
    let root = uses(handle.clone().into());

    let eqn = root.view().eqns().next().unwrap();
    for sub in eqn.subexprs() {
        assert!(sub.external() == Some(&handle));
        let child = sub.expr();
        assert!(std::ptr::eq(child.root(), &*lib));
        assert_eq!(child.closure_symbols().len(), 1);
        assert_eq!(
            format!("{:?}", child.eqns().next().unwrap().op()),
            "test::scale"
        );
    }
    // An external expression is stored by reference, so the root differs
    // from one that has the same expression internally.
    assert_ne!(root.hash(), uses(child().finish()).hash());
    assert_eq!(root.hash(), uses(handle.into()).hash());
}

#[test]
fn undefined_var() {
    let mut other = ExprBuilder::new();
    other.input(scalar());
    let foreign = other.input(scalar());

    let mut b = RootExpr::builder();
    b.input(scalar());
    b.output(foreign);
    assert!(matches!(b.build(), Err(BuildError::UndefinedVar(v)) if v == foreign));

    let mut b = RootExpr::builder();
    b.eqn(scale(1)).input(foreign);
    assert!(matches!(b.build(), Err(BuildError::UndefinedVar(_))));

    // Subexpressions are checked too.
    let mut inner = ExprBuilder::new();
    inner.output(foreign);
    let mut b = RootExpr::builder();
    b.eqn(call()).subexpr(inner);
    assert!(matches!(b.build(), Err(BuildError::UndefinedVar(_))));
}

#[test]
fn undefined_symbol() {
    // Not bound by an input
    let mut b = RootExpr::builder();
    let s = b.symbol();
    b.eqn(scale(1)).output(VarInfo::new(Dim(s)));
    assert!(matches!(b.build(), Err(BuildError::UndefinedSymbol(u)) if u == s));

    let mut b = RootExpr::builder();
    let s = b.symbol();
    b.closure_input(VarInfo::new(Dim(s)));
    assert!(matches!(b.build(), Err(BuildError::UndefinedSymbol(_))));

    let mut b = RootExpr::builder();
    let s = b.symbol();
    let x = b.input(scalar());
    b.eqn(call()).subexpr(child()).symbols([s]).vars([x]);
    assert!(matches!(b.build(), Err(BuildError::UndefinedSymbol(_))));
}

#[test]
fn closure_arity() {
    let mut b = RootExpr::builder();
    let x = b.input(scalar());
    b.eqn(call()).subexpr(child()).vars([x]);
    assert!(matches!(
        b.build(),
        Err(BuildError::SymbolArity {
            expected: 1,
            actual: 0
        })
    ));

    let mut b = RootExpr::builder();
    let s = b.closure_symbol();
    let x = b.input(scalar());
    b.eqn(call()).subexpr(child()).symbols([s]).vars([x, x]);
    assert!(matches!(
        b.build(),
        Err(BuildError::VarArity {
            expected: 1,
            actual: 2
        })
    ));

    // The closure of an external expression is checked as well.
    let handle = child().build().unwrap().handle();
    let mut b = RootExpr::builder();
    b.eqn(call()).subexpr(handle);
    assert!(matches!(b.build(), Err(BuildError::SymbolArity { .. })));
}

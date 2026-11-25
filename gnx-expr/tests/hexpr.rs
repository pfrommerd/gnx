use gnx_expr::{
    BuildError, CellAccess, HEqn, HExprBuilder, HInstr, HRootExpr, Info, Kind, Op, Operation,
    Symbol, Var, VarInfo, dialect,
};
use rkyv::rend::u32_le;
use rkyv::{Archive, Portable, Serialize};

// ---- a small dialect ----

#[derive(Archive, Serialize, Portable)]
#[rkyv(as = Self)]
#[repr(transparent)]
struct Scale(u32_le);
impl Operation for Scale {}

// Runs its subexpressions. The default cell access follows what is
// provided to them.
#[derive(Archive, Serialize, Portable)]
#[rkyv(as = Self)]
#[repr(transparent)]
struct Call(u32_le);
impl Operation for Call {}

// Writes to every cell among its inputs.
#[derive(Archive, Serialize, Portable)]
#[rkyv(as = Self)]
#[repr(transparent)]
struct Clear(u32_le);
impl Operation for Clear {
    fn cell_access(&self, eqn: HEqn<'_>, cell: Var) -> CellAccess {
        if eqn.inputs().contains(&cell) {
            CellAccess::Write
        } else {
            CellAccess::Unused
        }
    }
}

dialect! {
    dyn Operation, "test" => {
        "scale" => Scale,
        "call" => Call,
        "clear" => Clear,
    }
}

#[derive(Archive, Serialize, Portable)]
#[rkyv(as = Self)]
#[repr(transparent)]
struct Scalar(u32_le);
impl Info for Scalar {}

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
fn scale() -> Op {
    Op::new(Scale(2.into()))
}
fn call() -> Op {
    Op::new(Call(0.into()))
}
fn clear() -> Op {
    Op::new(Clear(0.into()))
}

// closure: (read: cell, written: cell), inputs: (v, passed: cell)
// Reads `read`, writes the scaled result to `written`, and reads `passed`.
fn child() -> HExprBuilder {
    let mut b = HExprBuilder::new();
    let read = b.closure_input(Kind::Cell, scalar());
    let written = b.closure_input(Kind::Cell, scalar());
    let v = b.input(Kind::Value, scalar());
    let passed = b.input(Kind::Cell, scalar());
    let r = b.read_cell(read, scalar());
    let [y] = b
        .eqn(scale())
        .inputs([r, v])
        .output(Kind::Value, scalar())
        .done();
    b.write_cell(written, y);
    let p = b.read_cell(passed, scalar());
    b.output(p);
    b
}

#[test]
fn round_trip() {
    let mut b = HRootExpr::builder();
    let s = b.symbol();
    let x = b.input(Kind::Value, VarInfo::new(Dim(s)));
    let c = b.input(Kind::Cell, scalar());
    let cell = b.create_cell(x, VarInfo::new(Dim(s)));
    let [y, made] = b
        .eqn(scale())
        .inputs([x, cell])
        .outputs([(Kind::Value, scalar()), (Kind::Cell, scalar())])
        .done();
    b.write_cell(c, y);
    let z = b.read_cell(cell, VarInfo::new(Dim(s)));
    b.outputs([z, made, c]);
    let root = b.build().unwrap();

    let expr = root.view();
    let inputs: Vec<_> = expr.inputs().collect();
    assert_eq!(inputs[0].var(), x);
    assert_eq!(inputs[0].kind(), Kind::Value);
    assert_eq!(inputs[0].symbols(), [s]);
    assert_eq!(inputs[1].kind(), Kind::Cell);
    assert_eq!(expr.outputs(), [z, made, c]);

    let instrs: Vec<_> = expr.instrs().collect();
    assert_eq!(instrs.len(), 4);
    match instrs[0] {
        HInstr::CreateCell {
            cell: created,
            initial,
            info,
        } => {
            assert_eq!((created, initial), (cell, x));
            assert_eq!(info.symbols(), [s]);
            assert_eq!(format!("{info:?}"), "test::dim");
        }
        other => panic!("{other:?}"),
    }
    let eqn = instrs[1].as_apply().unwrap();
    assert_eq!(format!("{:?}", eqn.op()), "test::scale");
    assert_eq!(eqn.inputs(), [x, cell]);
    let outputs: Vec<_> = eqn.outputs().map(|p| (p.var(), p.kind())).collect();
    assert_eq!(outputs, [(y, Kind::Value), (made, Kind::Cell)]);
    assert!(matches!(
        instrs[2],
        HInstr::WriteCell { cell, value } if (cell, value) == (c, y)
    ));
    assert!(matches!(
        instrs[3],
        HInstr::ReadCell { cell: read, output, .. } if (read, output) == (cell, z)
    ));

    // Definitions
    assert_eq!(expr.kind(x), Some(Kind::Value));
    assert_eq!(expr.kind(cell), Some(Kind::Cell));
    assert_eq!(expr.kind(made), Some(Kind::Cell));
    assert_eq!(expr.kind(z), Some(Kind::Value));
    assert_eq!(expr.definition(z).unwrap().symbols(), [s]);
    assert_eq!(expr.kind(Var::new(100)), None);

    // Accesses: structural instructions only, in this expression.
    assert_eq!(expr.cell_access(x), None);
    assert_eq!(expr.cell_access(cell), Some(CellAccess::Read));
    assert_eq!(expr.cell_access(c), Some(CellAccess::Write));
    assert_eq!(expr.cell_access(made), Some(CellAccess::Unused));

    let debug = format!("{root:?}");
    assert!(debug.contains("CreateCell"), "{debug}");
}

#[test]
fn cell_access_through_operations() {
    let mut b = HRootExpr::builder();
    let v = b.input(Kind::Value, scalar());
    let a = b.input(Kind::Cell, scalar());
    let w = b.input(Kind::Cell, scalar());
    let both = b.input(Kind::Cell, scalar());
    let cleared = b.input(Kind::Cell, scalar());
    let unused = b.input(Kind::Cell, scalar());

    // `both` is provided for the cell the child reads and, in a second
    // use of the child, for the one it writes.
    let child = child().finish();
    let mut e = b.eqn(call());
    e.inputs([v, unused]);
    e.subexpr(child.clone()).vars([a, w]);
    e.subexpr(child).vars([both, both]);
    b.eqn(clear()).input(cleared);
    b.read_cell(cleared, scalar());
    let root = b.build().unwrap();

    let expr = root.view();
    assert_eq!(expr.cell_access(a), Some(CellAccess::Read));
    assert_eq!(expr.cell_access(w), Some(CellAccess::Write));
    assert_eq!(expr.cell_access(both), Some(CellAccess::ReadWrite));
    // An operation's own summary, combined with a structural read
    assert_eq!(expr.cell_access(cleared), Some(CellAccess::ReadWrite));
    // An input of the equation, but not provided to a subexpression
    assert_eq!(expr.cell_access(unused), Some(CellAccess::Unused));

    let eqn = expr.instrs().next().unwrap().as_apply().unwrap();
    assert_eq!(eqn.captured_cell_access(a), CellAccess::Read);
    let sub = eqn.subexprs().next().unwrap();
    assert_eq!(sub.captured_cell_access(w), CellAccess::Write);
    // Through the inputs of the child: (v, passed)
    assert_eq!(sub.input_cell_access(&[v, a], a), CellAccess::Read);
    assert_eq!(sub.input_cell_access(&[v, a], w), CellAccess::Unused);
}

#[test]
fn external_exprs() {
    let handle = child().build().unwrap().handle();
    let mut b = HRootExpr::builder();
    let a = b.input(Kind::Cell, scalar());
    let w = b.input(Kind::Cell, scalar());
    b.eqn(call()).subexpr(handle.clone()).vars([a, w]);
    let root = b.build().unwrap();

    let expr = root.view();
    let sub = expr.instrs().next().unwrap().as_apply().unwrap();
    let sub = sub.subexprs().next().unwrap();
    assert!(sub.external() == Some(&handle));
    assert_eq!(sub.expr().instrs().len(), 4);
    assert_eq!(expr.cell_access(w), Some(CellAccess::Write));
}

fn kind_error(build: impl FnOnce(&mut HExprBuilder, Var, Var)) -> (Var, Var, BuildError) {
    let mut b = HRootExpr::builder();
    let value = b.input(Kind::Value, scalar());
    let cell = b.input(Kind::Cell, scalar());
    build(&mut b, value, cell);
    (value, cell, b.build().unwrap_err())
}

#[test]
fn kinds_are_checked() {
    let (value, _, err) = kind_error(|b, value, _| {
        b.read_cell(value, scalar());
    });
    assert!(matches!(err, BuildError::Kind { var, expected: Kind::Cell } if var == value));

    let (value, _, err) = kind_error(|b, value, _| {
        b.write_cell(value, value);
    });
    assert!(matches!(err, BuildError::Kind { var, expected: Kind::Cell } if var == value));

    let (_, cell, err) = kind_error(|b, _, cell| {
        b.write_cell(cell, cell);
    });
    assert!(matches!(err, BuildError::Kind { var, expected: Kind::Value } if var == cell));

    let (_, cell, err) = kind_error(|b, _, cell| {
        b.create_cell(cell, scalar());
    });
    assert!(matches!(err, BuildError::Kind { var, expected: Kind::Value } if var == cell));

    // A value provided for a closure input that is a cell
    let (value, _, err) = kind_error(|b, value, cell| {
        b.eqn(call()).subexpr(child()).vars([cell, value]);
    });
    assert!(matches!(err, BuildError::Kind { var, expected: Kind::Cell } if var == value));
}

#[test]
fn scopes_are_checked() {
    let mut b = HRootExpr::builder();
    let cell = b.input(Kind::Cell, scalar());
    b.write_cell(cell, Var::new(7));
    assert!(matches!(b.build(), Err(BuildError::UndefinedVar(_))));

    // The info of a cell's contents must use symbols in scope.
    let mut b = HRootExpr::builder();
    let s = b.symbol();
    let x = b.input(Kind::Value, scalar());
    b.create_cell(x, VarInfo::new(Dim(s)));
    assert!(matches!(b.build(), Err(BuildError::UndefinedSymbol(u)) if u == s));

    let mut b = HRootExpr::builder();
    let cell = b.input(Kind::Cell, scalar());
    b.eqn(call()).subexpr(child()).vars([cell]);
    assert!(matches!(
        b.build(),
        Err(BuildError::VarArity {
            expected: 2,
            actual: 1
        })
    ));
}

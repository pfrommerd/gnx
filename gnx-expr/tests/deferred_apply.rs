use std::sync::Arc;

use gnx_expr::{
    Capture, CellAccess, CellTrace, DeferredApplyPlan, DeferredRegion, Dialect, HEqn, Invocation,
    Op, Operation, Trace, TraceContext, ValueInfo, Var, hexpr,
};

struct TestDialect;
#[derive(Clone, PartialEq, Eq, Hash)]
struct TestOperation;

impl Dialect for TestDialect {
    fn name(&self) -> &'static str {
        "test"
    }
}

impl std::fmt::Display for TestOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}::{}",
            Operation::dialect(self).name(),
            Operation::name(self)
        )
    }
}

impl Operation for TestOperation {
    fn name(&self) -> &'static str {
        "regions"
    }

    fn dialect(&self) -> &'static dyn Dialect {
        &TestDialect
    }

    fn cell_access(&self, eqn: &HEqn, cell: Var) -> CellAccess {
        let operands = eqn.inputs().get(1..).unwrap_or_default();
        let explicit = eqn.subexprs()[0].input_cell_access(operands, cell);
        eqn.captured_cell_access(cell).combine(explicit)
    }
}

#[test]
fn write_only_deferred_apply_preserves_the_operation_and_all_regions() {
    let info = ValueInfo::new(0usize);
    let write = Arc::new(hexpr!(info.clone(); {
        lambda cell[:], value:
        cell[:] <- value;
        in ()
    }));
    let read = Arc::new(hexpr!(info.clone(); {
        lambda cell[:]; :
        old <- cell[:];
        in ()
    }));
    let guard = TraceContext::enter();
    let selector = Trace::placeholder(info.clone());
    let cell = CellTrace::placeholder(info.clone());
    let value = Trace::placeholder(info);
    let inputs = vec![selector.into(), cell.clone().into(), value.into()];
    let op = Op::new(TestOperation);
    let results = Invocation::defer_apply(
        op.clone(),
        DeferredApplyPlan {
            regions: vec![
                DeferredRegion {
                    symbol_captures: vec![],
                    body: write.clone(),
                    captures: vec![],
                },
                DeferredRegion {
                    symbol_captures: vec![],
                    body: read.clone(),
                    captures: vec![cell.into()],
                },
            ],
            inputs: inputs.clone(),
            outputs: vec![],
            projections: vec![],
            locals: vec![],
            external_access: CellAccess::ReadWrite,
        },
    );
    assert!(results.is_empty());

    let capture = Capture::from_context(guard.context(), &inputs, &[]).unwrap();
    let expr = capture.expr();
    let eqn = expr.instructions()[0].as_apply().unwrap();
    assert_eq!(expr.instructions().len(), 1);
    assert_eq!(eqn.op(), &op);
    assert_eq!(eqn.subexprs().len(), 2);
    assert!(Arc::ptr_eq(eqn.subexprs()[0].expr(), &write));
    assert!(Arc::ptr_eq(eqn.subexprs()[1].expr(), &read));
    assert_eq!(
        expr.cell_access(&expr.inputs()[1].0),
        Some(CellAccess::ReadWrite)
    );
}

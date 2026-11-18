use std::sync::Arc;

use gnx_expr::{
    Capture, CellAccess, CellTrace, DeferredApplyPlan, DeferredRegion, Dialect, HEqn, Invocation,
    Op, Operation, Trace, TraceContext, ValueInfo, Var, hexpr,
};

struct TestDialect;
struct TestOperation;

impl Dialect for TestDialect {
    fn name(&self) -> &'static str {
        "test"
    }

    fn create(&self, name: &'static str) -> Option<&'static dyn Operation> {
        (name == "regions").then_some(&TestOperation)
    }
}

impl Operation for TestOperation {
    fn name(&self) -> &'static str {
        "regions"
    }

    fn dialect(&self) -> &'static dyn Dialect {
        &TestDialect
    }

    fn region_operand(&self, eqn: &HEqn, region: usize, param: usize) -> Option<Var> {
        // Only region zero has explicit parameters. Skip the selector.
        (region == 0)
            .then(|| eqn.inputs().get(param + 1).copied())
            .flatten()
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
    let op = Op::new(&TestOperation);
    let results = Invocation::defer_apply(
        op.clone(),
        DeferredApplyPlan {
            regions: vec![
                DeferredRegion {
                    body: write.clone(),
                    captures: vec![],
                },
                DeferredRegion {
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

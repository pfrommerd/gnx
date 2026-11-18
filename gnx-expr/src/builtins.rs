use crate::trace::{Invocation, ObjectTracer, TraceInput, TraceOutput};
use crate::value::ValueInfo;
use crate::{Dialect, Expr, Op, Operation};
use std::sync::Arc;

/// Built-in immutable computation operations. Cell actions belong to HExpr,
/// not this dialect.
pub struct BuiltinsDialect;
pub struct CallOperation;

pub(crate) static BUILTINS_DIALECT: BuiltinsDialect = BuiltinsDialect;
pub(crate) static CALL_OP: CallOperation = CallOperation;
crate::register_dialect!(&BUILTINS_DIALECT);

impl Dialect for BuiltinsDialect {
    fn name(&self) -> &'static str {
        "builtins"
    }
    fn create(&self, name: &'static str) -> Option<&'static dyn Operation> {
        match name {
            "call" => Some(&CALL_OP),
            _ => None,
        }
    }
}
impl Operation for CallOperation {
    fn name(&self) -> &'static str {
        "call"
    }
    fn dialect(&self) -> &'static dyn Dialect {
        &BUILTINS_DIALECT
    }
}
pub fn call_op() -> Op {
    Op::new(&CALL_OP)
}

/// Emit one call to the canonical immutable Expr. Mutable HExpr bodies must be
/// lowered first; LoweredExpr::call handles cell snapshots and boundary updates.
pub fn call<C, I>(
    expr: Arc<Expr>,
    closure: C,
    inputs: I,
    outputs: Vec<ValueInfo>,
) -> Vec<ObjectTracer>
where
    C: IntoIterator,
    C::Item: Into<TraceInput>,
    I: IntoIterator,
    I::Item: Into<TraceInput>,
{
    expr.check().expect("invalid call expression");
    let closure: Vec<TraceInput> = closure.into_iter().map(Into::into).collect();
    assert_eq!(
        closure.len(),
        expr.closure().len(),
        "call capture arity mismatch"
    );
    let inputs: Vec<TraceInput> = inputs.into_iter().map(Into::into).collect();
    assert_eq!(
        inputs.len(),
        expr.inputs().len(),
        "call input arity mismatch"
    );
    assert_eq!(
        outputs.len(),
        expr.outputs().len(),
        "call output arity mismatch"
    );
    Invocation::invoke_with_subexprs(
        call_op(),
        vec![(expr, closure.into_iter().map(Into::into).collect())],
        inputs,
        outputs.into_iter().map(TraceOutput::Ref).collect(),
    )
}

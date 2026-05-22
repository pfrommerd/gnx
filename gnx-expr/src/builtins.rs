use crate::trace::{Generic, Invocation, TraceObject, Tracer};
use crate::value::ValueInfo;
use crate::{AttrMap, Dialect, Expr, Op, Operation};
use std::sync::Arc;

/// Built-in `builtins` dialect operations.
pub struct BuiltinsDialect;

/// `gnx.update` — records a [`TraceCell`](crate::trace::TraceCell) write in a captured expr.
pub struct UpdateOperation;

/// `gnx.call` — invokes a captured [`Expr`] with the given trace objects.
pub struct CallOperation;

static BUILTINS_DIALECT: BuiltinsDialect = BuiltinsDialect;
static UPDATE_OP: UpdateOperation = UpdateOperation;
static CALL_OP: CallOperation = CallOperation;

crate::register_dialect!(&BUILTINS_DIALECT);

impl Dialect for BuiltinsDialect {
    fn name(&self) -> &'static str {
        "builtins"
    }
    fn create(&self, name: &'static str) -> Option<&'static dyn Operation> {
        match name {
            "update" => Some(&UPDATE_OP),
            "call" => Some(&CALL_OP),
            _ => None,
        }
    }
}

impl Operation for UpdateOperation {
    fn name(&self) -> &'static str {
        "update"
    }
    fn dialect(&self) -> &'static dyn Dialect {
        &BUILTINS_DIALECT
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

/// An [`Op`] for `gnx.update`.
pub fn update_op() -> Op {
    Op::new(&UPDATE_OP, AttrMap::default())
}

/// An [`Op`] for `gnx.call`.
pub fn call_op() -> Op {
    Op::new(&CALL_OP, AttrMap::default())
}

/// Invoke `gnx.call` with `expr` as an attribute and the given closure/input trace objects.
pub fn call<C, I>(
    expr: Expr,
    // The closure arguments for the expression.
    closure: C,
    inputs: I,
    outputs: Vec<ValueInfo>,
) -> Vec<TraceObject>
where
    C: IntoIterator,
    C::Item: Into<TraceInput>,
    I: IntoIterator,
    I::Item: Into<TraceInput>,
{
    let closure: Vec<TraceObject> = closure.into_iter().map(Into::into).collect();
    Invocation::invoke_with_subexprs(
        call_op(),
        vec![(Arc::new(expr), closure.clone())],
        closure,
        inputs.into_iter().map(Into::into).collect(),
        outputs,
    )
}

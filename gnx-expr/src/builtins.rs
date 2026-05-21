use crate::trace::{Generic, Invocation, TraceObject, Tracer};
use crate::value::ValueInfo;
use crate::{AttrMap, Dialect, Expr, Op, Operation};

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
    Op::new(&UPDATE_OP, AttrMap::default(), None)
}

/// An [`Op`] for `gnx.call` with the given subroutine [`Expr`].
pub fn call_op(expr: Expr) -> Op {
    Op::new(&CALL_OP, AttrMap::default(), Some(vec![expr]))
}

/// Invoke `gnx.call` with `expr` as an attribute and the given closure/input trace objects.
pub fn call<C, I>(
    expr: Expr,
    closure: C,
    inputs: I,
    outputs: Vec<ValueInfo>,
) -> Vec<Tracer<Generic>>
where
    C: IntoIterator,
    C::Item: Into<TraceObject>,
    I: IntoIterator,
    I::Item: Into<TraceObject>,
{
    Invocation::invoke(
        call_op(expr),
        closure.into_iter().map(Into::into).collect(),
        inputs.into_iter().map(Into::into).collect(),
        outputs,
    )
}

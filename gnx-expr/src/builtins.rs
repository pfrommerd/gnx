use crate::trace::{Generic, Invocation, TraceObject, Tracer};
use crate::value::ValueInfo;
use crate::{Attr, AttrMap, Dialect, Expr, Op, Operation};

/// Built-in `gnx` dialect operations.
pub struct GnxDialect;

/// `gnx.update` — records a [`TraceCell`](crate::trace::TraceCell) write in a captured expr.
pub struct UpdateOperation;

/// `gnx.call` — invokes a captured [`Expr`] with the given trace objects.
pub struct CallOperation;

static GNX_DIALECT: GnxDialect = GnxDialect;
static UPDATE_OP: UpdateOperation = UpdateOperation;
static CALL_OP: CallOperation = CallOperation;

impl Dialect for GnxDialect {
    fn name(&self) -> &'static str {
        "gnx"
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
        &GNX_DIALECT
    }
}

impl Operation for CallOperation {
    fn name(&self) -> &'static str {
        "call"
    }
    fn dialect(&self) -> &'static dyn Dialect {
        &GNX_DIALECT
    }
}

/// An [`Op`] for `gnx.update`.
pub fn update_op() -> Op {
    Op::new(&UPDATE_OP, AttrMap::default())
}

/// An [`Op`] for `gnx.call` with the given subroutine [`Expr`].
pub fn call_op(expr: Expr) -> Op {
    let mut attrs = AttrMap::default();
    attrs.insert("expr".to_string(), Attr::Expr(expr));
    Op::new(&CALL_OP, attrs)
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

use crate::trace::Tracer;
use crate::trace::{Invocation, TraceInput, TraceObject, TraceOutput};
use crate::value::ValueInfo;
use crate::{AttrMap, Dialect, Expr, Op, Operation};
use std::sync::Arc;

/// Built-in `builtins` dialect operations.
pub struct BuiltinsDialect;

/// `gnx.update` — records a [`TraceCell`](crate::trace::TraceCell) write in a captured expr.
pub struct UpdateOperation;

/// `gnx.read` — records a [`TraceCell`](crate::trace::TraceCell) read in a captured expr.
pub struct ReadOperation;

/// `gnx.create_cell` — creates a [`TraceCell`](crate::trace::TraceCell) in a captured expr.
pub struct CreateCellOperation;

/// `gnx.call` — invokes a captured [`Expr`] with the given trace objects.
pub struct CallOperation;

pub(crate) static BUILTINS_DIALECT: BuiltinsDialect = BuiltinsDialect;
pub(crate) static UPDATE_OP: UpdateOperation = UpdateOperation;
pub(crate) static READ_OP: ReadOperation = ReadOperation;
pub(crate) static CREATE_CELL_OP: CreateCellOperation = CreateCellOperation;
pub(crate) static CALL_OP: CallOperation = CallOperation;

crate::register_dialect!(&BUILTINS_DIALECT);

impl Dialect for BuiltinsDialect {
    fn name(&self) -> &'static str {
        "builtins"
    }
    fn create(&self, name: &'static str) -> Option<&'static dyn Operation> {
        match name {
            "update" => Some(&UPDATE_OP),
            "read" => Some(&READ_OP),
            "create_cell" => Some(&CREATE_CELL_OP),
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

impl Operation for ReadOperation {
    fn name(&self) -> &'static str {
        "read"
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

impl Operation for CreateCellOperation {
    fn name(&self) -> &'static str {
        "create_cell"
    }
    fn dialect(&self) -> &'static dyn Dialect {
        &BUILTINS_DIALECT
    }
}

/// An [`Op`] for `gnx.update`.
pub fn update_op() -> Op {
    Op::new(&UPDATE_OP)
}

/// An [`Op`] for `gnx.read`.
pub fn read_op() -> Op {
    Op::new(&READ_OP)
}

/// An [`Op`] for `gnx.create_cell`.
pub fn create_cell_op() -> Op {
    Op::new(&CREATE_CELL_OP)
}

/// An [`Op`] for `gnx.call`.
pub fn call_op() -> Op {
    Op::new(&CALL_OP)
}

pub fn create_cell<T>(base: Tracer<T>) -> TraceObject
where
    T: crate::Traceable,
{
    let info = base.trace_ref().info().clone();
    Invocation::invoke_with_subexprs(
        create_cell_op(),
        Vec::new(),
        vec![TraceInput::Ref(base.into_trace_ref())],
        vec![TraceOutput::NewCell(info)],
    )
    .into_iter()
    .next()
    .expect("create_cell produces one cell")
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
    let closure: Vec<TraceInput> = closure.into_iter().map(Into::into).collect();
    let inputs: Vec<TraceInput> = inputs.into_iter().map(Into::into).collect();
    Invocation::invoke_with_subexprs(
        call_op(),
        vec![(Arc::new(expr), closure.clone())],
        closure.into_iter().chain(inputs).collect(),
        outputs.into_iter().map(TraceOutput::Ref).collect(),
    )
}

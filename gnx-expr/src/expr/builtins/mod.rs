use crate::expr::{AttrMap, Dialect, Op, Operation};

/// Built-in `gnx` dialect operations.
pub struct GnxDialect;

/// `gnx.update` — records a [`TraceCell`](crate::trace::TraceCell) write in a captured expr.
pub struct UpdateOperation;

static GNX_DIALECT: GnxDialect = GnxDialect;
static UPDATE_OP: UpdateOperation = UpdateOperation;

impl Dialect for GnxDialect {
    fn name(&self) -> &'static str {
        "gnx"
    }
    fn create(&self, name: &'static str) -> Option<&'static dyn Operation> {
        match name {
            "update" => Some(&UPDATE_OP),
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

/// An [`Op`] for `gnx.update`.
pub fn update_op() -> Op {
    Op::new(&UPDATE_OP, AttrMap::default())
}

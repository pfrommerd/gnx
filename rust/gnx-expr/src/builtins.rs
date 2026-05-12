use crate::expr::{Dialect, Operation};

struct BuiltinsDialect;

impl Dialect for BuiltinsDialect {
    fn name(&self) -> &'static str {
        "builtins"
    }
    fn create(&self, name: &'static str) -> Option<&'static dyn Operation> {
        Some(match name {
            "invoke" => &InvokeOp,
            _ => return None
        })
    }
}

pub static BUILTINS_DIALECT: BuiltinsDialect = BuiltinsDialect;

struct InvokeOp;

impl Operation for InvokeOp {
    fn name(&self) -> &'static str {
        "invoke"
    }
    fn dialect(&self) -> &'static dyn Dialect { &BuiltinsDialect }
}
use gnx_expr::{Dialect, Op, Operation};

pub struct ArrayDialect;
pub struct AddOperation;

static ARRAY: ArrayDialect = ArrayDialect;
static ADD: AddOperation = AddOperation;

gnx_expr::register_dialect!(&ARRAY);

impl Dialect for ArrayDialect {
    fn name(&self) -> &'static str {
        "array"
    }

    fn create(&self, name: &'static str) -> Option<&'static dyn Operation> {
        match name {
            "add" => Some(&ADD),
            _ => None,
        }
    }
}

impl Operation for AddOperation {
    fn name(&self) -> &'static str {
        "add"
    }

    fn dialect(&self) -> &'static dyn Dialect {
        &ARRAY
    }
}

/// Elementwise addition. The backend checks the operand and result metadata.
pub fn add_op() -> Op {
    Op::new(&ADD)
}

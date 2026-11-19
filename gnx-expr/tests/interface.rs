use gnx_expr::{Dialect, InterfaceError, Op, Operation, register_op_interface};

struct TestDialect;
static DIALECT: TestDialect = TestDialect;
impl Dialect for TestDialect {
    fn name(&self) -> &'static str {
        "interface_test"
    }
    fn create(&self, _: &'static str) -> Option<&'static dyn Operation> {
        None
    }
}

macro_rules! operation {
    ($ty:ident, $name:literal) => {
        struct $ty;
        impl Operation for $ty {
            fn name(&self) -> &'static str {
                $name
            }
            fn dialect(&self) -> &'static dyn Dialect {
                &DIALECT
            }
        }
    };
}
operation!(Double, "double");
operation!(Square, "square");
operation!(Plain, "plain");
operation!(Twice, "twice");

trait Evaluate: Sync {
    fn evaluate(&self, x: i64) -> i64;
}
trait Describe: Sync {
    fn describe(&self) -> String;
}

impl Evaluate for Double {
    fn evaluate(&self, x: i64) -> i64 {
        2 * x
    }
}
impl Evaluate for Square {
    fn evaluate(&self, x: i64) -> i64 {
        x * x
    }
}
impl Describe for Double {
    fn describe(&self) -> String {
        "doubles".into()
    }
}
impl Evaluate for Twice {
    fn evaluate(&self, x: i64) -> i64 {
        x
    }
}
register_op_interface!(dyn Evaluate: Double, Square);
register_op_interface!(dyn Describe: Double);
register_op_interface!(dyn Evaluate: Twice);
register_op_interface!(dyn Evaluate: Twice);

#[test]
fn dispatches_per_interface_and_operation() {
    let double = Op::new(&Double);
    let square = Op::new(&Square);
    assert_eq!(double.interface::<dyn Evaluate>().unwrap().evaluate(3), 6);
    assert_eq!(square.interface::<dyn Evaluate>().unwrap().evaluate(3), 9);
    assert_eq!(
        double.interface::<dyn Describe>().unwrap().describe(),
        "doubles"
    );
    assert!(double.implements::<dyn Describe>());
    assert!(!square.implements::<dyn Describe>());
}

#[test]
fn reports_missing_and_duplicate_registrations() {
    let missing = Op::new(&Plain).interface::<dyn Evaluate>().err().unwrap();
    assert!(matches!(
        missing,
        InterfaceError::Missing {
            dialect: "interface_test",
            op: "plain",
            ..
        }
    ));
    assert!(missing.to_string().contains("Evaluate"));
    let twice = Op::new(&Twice);
    assert!(matches!(
        twice.interface::<dyn Evaluate>(),
        Err(InterfaceError::Duplicate { op: "twice", .. })
    ));
    assert!(!twice.implements::<dyn Evaluate>());
}

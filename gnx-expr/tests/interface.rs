use std::collections::HashSet;
use std::hash::{BuildHasher, RandomState};

use gnx_expr::{InterfaceError, Op, Operation, Typed, dialect, register_op_interface};
use rkyv::{Archive, Portable, Serialize};

macro_rules! operation {
    ($ty:ident) => {
        #[derive(Archive, Serialize, Portable)]
        #[rkyv(as = Self)]
        #[repr(C)]
        struct $ty;
        impl Operation for $ty {}
    };
}
operation!(Double);
operation!(Square);
operation!(Plain);
operation!(Twice);

// An operation with parameters stored out of line: the builder `Scale`
// archives as `ScaleOp`, which is what implements `Operation`.
#[derive(Archive, Serialize)]
#[rkyv(archived = ScaleOp)]
struct Scale(Vec<i64>);
impl Operation for ScaleOp {
    fn display(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}<{:?}>", self.type_name(), self.0.as_slice())
    }
}

dialect! {
    dyn Operation, "interface_test" => {
        "double" => Double,
        "square" => Square,
        "plain" => Plain,
        "twice" => Twice,
        "scale" => ScaleOp,
    }
}

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
impl Evaluate for ScaleOp {
    fn evaluate(&self, x: i64) -> i64 {
        self.0.iter().map(|k| k.to_native()).sum::<i64>() * x
    }
}

// A stateless provider for an operation defined elsewhere
struct PlainDescription;
impl Describe for PlainDescription {
    fn describe(&self) -> String {
        "plain".into()
    }
}

register_op_interface!(dyn Evaluate: Double, Square);
register_op_interface!(dyn Describe: Double);
register_op_interface!(dyn Evaluate: Twice);
register_op_interface!(dyn Evaluate: Twice);
register_op_interface!(dyn Evaluate: ScaleOp);
register_op_interface!(dyn Describe: Plain => PlainDescription);

#[test]
fn dispatches_per_interface_and_operation() {
    let double = Op::new(Double);
    let square = Op::new(Square);
    assert_eq!(double.interface::<dyn Evaluate>().unwrap().evaluate(3), 6);
    assert_eq!(square.interface::<dyn Evaluate>().unwrap().evaluate(3), 9);
    assert_eq!(
        double.interface::<dyn Describe>().unwrap().describe(),
        "doubles"
    );
    assert!(double.implements::<dyn Describe>());
    assert!(!square.implements::<dyn Describe>());
    let plain = Op::new(Plain);
    assert_eq!(
        plain.interface::<dyn Describe>().unwrap().describe(),
        "plain"
    );
}

#[test]
fn reports_missing_and_duplicate_registrations() {
    let plain = Op::new(Plain);
    let missing = plain.interface::<dyn Evaluate>().err().unwrap();
    assert!(matches!(
        missing,
        InterfaceError::Missing {
            dialect: "interface_test",
            op: "plain",
            ..
        }
    ));
    assert!(missing.to_string().contains("Evaluate"));
    let twice = Op::new(Twice);
    assert!(matches!(
        twice.interface::<dyn Evaluate>(),
        Err(InterfaceError::Duplicate { op: "twice", .. })
    ));
    assert!(!twice.implements::<dyn Evaluate>());
}

#[test]
fn owned_operation_parameters_drive_identity_and_interfaces() {
    let op = Op::new(Scale(vec![2, 3]));
    let equal = Op::new(Scale(vec![2, 3]));
    let different = Op::new(Scale(vec![2, 1]));
    let clone = op.clone();
    assert_eq!(op, equal);
    assert_eq!(op, clone);
    assert_ne!(op, different);
    assert_ne!(op, Op::new(Double));
    let hashes = RandomState::new();
    assert_eq!(hashes.hash_one(&op), hashes.hash_one(&equal));
    assert!(HashSet::from([op, different]).contains(&clone));
    assert_eq!(clone.interface::<dyn Evaluate>().unwrap().evaluate(2), 10);
    let borrowed: &dyn Operation = &*clone;
    assert_eq!(borrowed.name(), "scale");
    assert_eq!(borrowed.dialect(), "interface_test");
    assert_eq!(borrowed.downcast_ref::<ScaleOp>().unwrap().0, [2, 3]);
    assert!(borrowed.downcast_ref::<Double>().is_none());
    assert_eq!(clone.to_string(), "interface_test::scale<[2, 3]>");
    assert_eq!(Op::new(Double).to_string(), "interface_test::double");
}

#[test]
fn operations_cross_threads() {
    let op = Op::new(Scale(vec![4]));
    let result = std::thread::spawn(move || op.interface::<dyn Evaluate>().unwrap().evaluate(2));
    assert_eq!(result.join().unwrap(), 8);
}

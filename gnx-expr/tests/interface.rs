use gnx_expr::{Dialect, InterfaceError, Op, Operation, register_op_interface};

struct TestDialect;
static DIALECT: TestDialect = TestDialect;
impl Dialect for TestDialect {
    fn name(&self) -> &'static str {
        "interface_test"
    }
}

macro_rules! operation {
    ($ty:ident, $name:literal) => {
        #[derive(Clone, PartialEq, Eq, Hash)]
        struct $ty;
        impl std::fmt::Display for $ty {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}::{}", Operation::dialect(self).name(), Operation::name(self))
            }
        }

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
}

#[test]
fn reports_missing_and_duplicate_registrations() {
    let missing = Op::new(Plain).interface::<dyn Evaluate>().err().unwrap();
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
    use std::collections::HashSet;
    use std::hash::{BuildHasher, Hash, Hasher, RandomState};

    #[derive(Clone, PartialEq, Eq, Hash)]
    struct Scale(Vec<i64>);

    impl std::fmt::Display for Scale {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("interface_test::double")
        }
    }

    impl Operation for Scale {
        fn name(&self) -> &'static str {
            "double"
        }
        fn dialect(&self) -> &'static dyn Dialect {
            &DIALECT
        }
    }

    impl Evaluate for Scale {
        fn evaluate(&self, x: i64) -> i64 {
            self.0.iter().sum::<i64>() * x
        }
    }

    register_op_interface!(dyn Evaluate: Scale);

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
    let mut expected = hashes.build_hasher();
    "interface_test".hash(&mut expected);
    "double".hash(&mut expected);
    vec![2_i64, 3].hash(&mut expected);
    assert_eq!(hashes.hash_one(&op), expected.finish());
    assert!(HashSet::from([op, different]).contains(&clone));
    assert_eq!(clone.interface::<dyn Evaluate>().unwrap().evaluate(2), 10);
    let borrowed: &dyn Operation = &*clone;
    assert_eq!(borrowed.name(), "double");
    let boxed: Box<dyn Operation> = clone.into();
    assert_eq!(boxed.downcast_ref::<Scale>().unwrap().0, [2, 3]);
}

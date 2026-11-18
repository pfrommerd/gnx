use gnx_expr::{Dialect, Operation};

struct TestDialect;
struct TestOperation;

static TEST_DIALECT: TestDialect = TestDialect;
static TEST_OPERATION: TestOperation = TestOperation;

gnx_expr::register_dialect!(&TEST_DIALECT);

impl Dialect for TestDialect {
    fn name(&self) -> &'static str {
        "test"
    }

    fn create(&self, name: &'static str) -> Option<&'static dyn Operation> {
        match name {
            "op" => Some(&TEST_OPERATION),
            _ => None,
        }
    }
}

impl Operation for TestOperation {
    fn name(&self) -> &'static str {
        "op"
    }

    fn dialect(&self) -> &'static dyn Dialect {
        &TEST_DIALECT
    }
}

#[test]
fn dialects_returns_registered_dialects() {
    let dialects = gnx_expr::dialects();

    let test = dialects.get("test").expect("test dialect is registered");
    assert_eq!(test.name(), "test");
    assert!(test.create("op").is_some());

    let names = dialects.iter().map(|(name, _)| name).collect::<Vec<_>>();
    assert!(names.contains(&"test"));
}

use gnx_expr::{ConcreteValue, Value, ValueInfo};
use std::fmt::Display;

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct SampleInfo;

impl Display for SampleInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SampleInfo")
    }
}


#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct SampleValue;

impl Display for SampleValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SampleValue")
    }
}

impl ConcreteValue for SampleValue {
    fn to_info(&self) -> ValueInfo {
        ValueInfo::new(SampleInfo)
    }
}

#[test]
fn value_info_downcast() {
    let info = ValueInfo::new(SampleInfo);
    assert!(info.downcast::<SampleInfo>().is_ok());
}

#[test]
fn value_downcast() {
    let value = Value::new(SampleValue);
    assert!(value.downcast::<SampleValue>().is_ok());
}
use super::value::ValueInfo;

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fmt;

// The static attributes of an operation.
#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub enum Attr {
    String(Cow<'static, str>),
    Info(ValueInfo),
    List(Box<AttrList>),
    Map(Box<AttrMap>),
}

impl fmt::Display for Attr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Attr::String(string) => write!(f, "{string:?}"),
            Attr::Info(info) => write!(f, "info({info})"),
            Attr::List(list) => {
                write!(f, "[")?;
                for (i, attr) in list.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{attr}")?;
                }
                write!(f, "]")
            }
            Attr::Map(map) => {
                write!(f, "{{")?;
                for (i, (key, value)) in map.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{key}: {value}")?;
                }
                write!(f, "}}")
            }
        }
    }
}

pub type AttrMap = BTreeMap<String, Attr>;
pub type AttrList = Vec<Attr>;

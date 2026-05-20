use crate::Expr;
use super::value::ValueInfo;

use std::borrow::Cow;
use std::collections::BTreeMap;

// The static attributes of an operation.
#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub enum Attr {
    String(Cow<'static, str>),
    Info(ValueInfo),
    Expr(Expr),
    List(Box<AttrList>),
    Map(Box<AttrMap>),
}

impl std::fmt::Display for Attr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Attr::String(string) => write!(f, "{}", string),
            Attr::Info(info) => write!(f, "{}", info),
            Attr::Expr(expr) => write!(f, "{}", expr),
            Attr::List(list) => {
                write!(f, "[")?;
                for (i, attr) in list.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", attr)?;
                }
                write!(f, "]")
            },
            Attr::Map(map) => {
                write!(f, "{{")?;
                for (i, (k, v)) in map.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}: {}", k, v)?;
                }
                write!(f, "}}")
            }
        }
    }
}


pub type AttrMap = BTreeMap<String, Attr>;
pub type AttrList = Vec<Attr>;

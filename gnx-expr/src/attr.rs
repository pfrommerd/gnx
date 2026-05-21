use super::value::ValueInfo;

use std::borrow::Cow;
use std::collections::BTreeMap;

// The static attributes of an operation.
#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub enum Attr {
    String(Cow<'static, str>),
    Info(ValueInfo),
    List(Box<AttrList>),
    Map(Box<AttrMap>),
}

impl std::fmt::Display for Attr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        crate::pretty::fmt_attr(self, f)
    }
}

pub type AttrMap = BTreeMap<String, Attr>;
pub type AttrList = Vec<Attr>;

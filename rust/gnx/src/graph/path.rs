use std::borrow::Cow;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Key {
    Attr(Cow<'static, str>),
    DictKey(Cow<'static, str>),
    DictIndex(i64),
    Index(usize),
}

#[derive(Clone, Copy)]
pub enum KeyRef<'r> {
    Attr(&'r str),
    DictKey(&'r str),
    DictIndex(i64),
    Index(usize),
}

pub type Path = Vec<Key>;

impl std::fmt::Display for Key {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Key::Attr(name) => write!(f, ".{}", name),
            Key::DictKey(key) => write!(f, "[\"{}\"]", key),
            Key::DictIndex(index) => write!(f, "[{}]", index),
            Key::Index(index) => write!(f, "[{}]", index),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(transparent)]
pub struct GraphId(u64);

impl std::fmt::Display for GraphId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "#{}", self.0)
    }
}

impl From<u64> for GraphId {
    fn from(value: u64) -> Self {
        GraphId(value)
    }
}

impl From<GraphId> for u64 {
    fn from(value: GraphId) -> Self {
        value.0
    }
}

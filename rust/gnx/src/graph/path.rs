use std::borrow::Cow;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Key {
    Attr(Cow<'static, str>),
    DictKey(Cow<'static, str>),
    DictIndex(i64),
    Index(usize),
}

impl Key {
    pub fn as_ref<'r>(&'r self) -> KeyRef<'r> {
        match self {
            Key::Attr(name) => KeyRef::Attr(name.as_ref()),
            Key::DictKey(key) => KeyRef::DictKey(key.as_ref()),
            Key::DictIndex(index) => KeyRef::DictIndex(*index),
            Key::Index(index) => KeyRef::Index(*index),
        }
    }
}

#[derive(Clone, Copy)]
pub enum KeyRef<'r> {
    Attr(&'r str),
    DictKey(&'r str),
    DictIndex(i64),
    Index(usize),
}
impl<'r> KeyRef<'r> {
    pub fn to_value(&self) -> Key {
        match self {
            KeyRef::Attr(name) => Key::Attr(Cow::Owned(name.to_string())),
            KeyRef::DictKey(key) => Key::DictKey(Cow::Owned(key.to_string())),
            KeyRef::DictIndex(index) => Key::DictIndex(*index),
            KeyRef::Index(index) => Key::Index(*index),
        }
    }
}

pub enum KeyCow<'r> {
    Attr(Cow<'r, str>),
    DictKey(Cow<'r, str>),
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

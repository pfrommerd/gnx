use std::borrow::Cow;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Key {
    Named(Cow<'static, str>),
    Index(isize),
    // Used if the parent is a "wrapper" around the child (e.g. newtype structs)
    Transparent,
}

impl From<&'static str> for Key {
    fn from(value: &'static str) -> Self {
        Key::Named(Cow::Borrowed(value))
    }
}
impl From<String> for Key {
    fn from(value: String) -> Self {
        Key::Named(Cow::Owned(value))
    }
}
impl From<isize> for Key {
    fn from(value: isize) -> Self {
        Key::Index(value)
    }
}
impl From<i32> for Key {
    fn from(value: i32) -> Self {
        Key::Index(value as isize)
    }
}
impl From<usize> for Key {
    fn from(value: usize) -> Self {
        Key::Index(value as isize)
    }
}
impl From<()> for Key {
    fn from(_: ()) -> Self {
        Key::Transparent
    }
}

impl Key {
    pub fn as_ref<'r>(&'r self) -> KeyRef<'r> {
        match self {
            Key::Named(name) => KeyRef::Named(name.as_ref()),
            Key::Index(index) => KeyRef::Index(*index),
            Key::Transparent => KeyRef::Transparent,
        }
    }
}

#[derive(Clone, Copy)]
pub enum KeyRef<'r> {
    Named(&'r str),
    Index(isize),
    Transparent,
}

impl<'r> From<&'r str> for KeyRef<'r> {
    fn from(value: &'r str) -> Self {
        KeyRef::Named(value)
    }
}
impl<'r> From<isize> for KeyRef<'r> {
    fn from(value: isize) -> Self {
        KeyRef::Index(value)
    }
}
impl<'r> From<i32> for KeyRef<'r> {
    fn from(value: i32) -> Self {
        KeyRef::Index(value as isize)
    }
}
impl<'r> From<usize> for KeyRef<'r> {
    fn from(value: usize) -> Self {
        KeyRef::Index(value as isize)
    }
}
impl<'r> From<()> for KeyRef<'r> {
    fn from(_: ()) -> Self {
        KeyRef::Transparent
    }
}

impl<'r> KeyRef<'r> {
    pub fn to_value(&self) -> Key {
        match self {
            KeyRef::Named(name) => Key::Named(Cow::Owned(name.to_string())),
            KeyRef::Index(index) => Key::Index(*index),
            KeyRef::Transparent => Key::Transparent,
        }
    }
}

pub type Path = Vec<Key>;

impl std::fmt::Display for Key {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Key::Named(name) => write!(f, ".{}", name),
            Key::Index(index) => write!(f, "[{}]", index),
            Key::Transparent => write!(f, ""), // Transparent keys are not displayed
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

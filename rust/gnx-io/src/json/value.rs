use gnx_graph::{
    Error, Serialize, Serializer, Expecting,
    Deserialize, Deserializer, DataVisitor,
    SeqAccess, MapAccess
};

use ordered_float::NotNan;
use super::JsonError;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum N {
    Float(NotNan<f64>),
    Pos(u64),
    Neg(i64),
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Number {
    pub kind: N
}
impl std::fmt::Debug for Number {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self.kind {
            N::Float(value) => write!(f, "{}", value),
            N::Pos(value) => write!(f, "{}", value),
            N::Neg(value) => write!(f, "{}", value),
        }
    }
}

impl From<f64> for Number {
    fn from(value: f64) -> Self {
        Number { kind: N::Float(NotNan::new(value).unwrap()) }
    }
}

macro_rules! impl_pos_numeric {
    ($($T:ty)*) => {
        $(
        impl From<$T> for Number {
            fn from(value: $T) -> Self {
                Number { kind: N::Pos(value as u64) }
            }
        }
        impl TryInto<$T> for Number {
            type Error = JsonError;

            fn try_into(self) -> Result<$T, Self::Error> {
                match self.kind {
                    N::Float(value) => Ok(value.into_inner() as $T),
                    N::Pos(value) => Ok(value as $T),
                    N::Neg(_) => Err(JsonError::InvalidNumber),
                }
            }
        }
        )*
    };
}
impl_pos_numeric!(u8 u16 u32 u64 u128);

macro_rules! impl_signed_numeric {
    ($($T:ty)*) => {
        $(
        impl From<$T> for Number {
            fn from(value: $T) -> Self {
                if value >= 0 {
                    Number { kind: N::Pos(value as u64) }
                } else {
                    Number { kind: N::Neg(value as i64) }
                }
            }
        }
        impl TryInto<$T> for Number {
            type Error = JsonError;

            fn try_into(self) -> Result<$T, Self::Error> {
                match self.kind {
                    N::Float(value) => Ok(value.into_inner() as $T),
                    N::Pos(value) => Ok(value as $T),
                    N::Neg(value) => Ok(value as $T)
                }
            }
        }
        )*
    };
}
impl_signed_numeric!(i8 i16 i32 i64 i128);

pub type Map<K, V> = std::collections::BTreeMap<K, V>;

#[derive(Debug, PartialEq, Eq, Hash)]
pub enum Value {
    Null,
    Bool(bool),
    Number(Number),
    String(String),
    Array(Vec<Value>),
    Object(Map<String, Value>),
}

impl From<bool> for Value {
    fn from(value: bool) -> Self {
        Value::Bool(value)
    }
}
impl From<Number> for Value {
    fn from(value: Number) -> Self {
        Value::Number(value)
    }
}
impl From<i64> for Value {
    fn from(value: i64) -> Self {
        Value::Number(Number::from(value))
    }
}
impl From<u64> for Value {
    fn from(value: u64) -> Self {
        Value::Number(Number::from(value))
    }
}
impl From<f64> for Value {
    fn from(value: f64) -> Self {
        Value::Number(Number::from(value))
    }
}


impl Serialize for Value {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Value::Null => serializer.serialize_none(),
            Value::Bool(value) => serializer.serialize_bool(*value),
            Value::Number(value) => {
                match value.kind {
                    N::Float(value) => serializer.serialize_f64(value.into_inner()),
                    N::Pos(value) => serializer.serialize_u64(value),
                    N::Neg(value) => serializer.serialize_i64(value),
                }
            },
            Value::String(value) => serializer.serialize_str(value),
            Value::Array(value) => {
                serializer.collect_seq(value.iter())
            },
            Value::Object(value) => {
                serializer.collect_map(value.iter())
            },
        }
    }
}

impl<'de> Deserialize<'de> for Value {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Visitor;
        impl<'de> Expecting for Visitor {
            fn expected(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(fmt, "any JSON-compatible value")
            }
        }
        impl<'de> DataVisitor<'de> for Visitor {
            type Value = Value;

            fn visit_unit<E: Error>(self) -> Result<Self::Value, E> {
                Ok(Value::Null)
            }
            fn visit_bool<E: Error>(self, value: bool) -> Result<Self::Value, E> {
                Ok(Value::Bool(value))
            }
            fn visit_i64<E: Error>(self, value: i64) -> Result<Self::Value, E> {
                Ok(Value::Number(Number::from(value)))
            }
            fn visit_u64<E: Error>(self, value: u64) -> Result<Self::Value, E> {
                Ok(Value::Number(Number::from(value)))
            }
            fn visit_f64<E: Error>(self, value: f64) -> Result<Self::Value, E> {
                Ok(Value::Number(Number::from(value)))
            }
            fn visit_str<E: Error>(self, value: &str) -> Result<Self::Value, E> {
                Ok(Value::String(value.to_string()))
            }
            fn visit_none<E: Error>(self) -> Result<Self::Value, E> {
                Ok(Value::Null)
            }
            fn visit_some<D: Deserializer<'de>>(self, some: D) -> Result<Self::Value, D::Error> {
                Value::deserialize(some)
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let mut v = Vec::new();
                while let Some(item) = seq.next_element()? {
                    v.push(item);
                }
                Ok(Value::Array(v))
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                let mut h = Map::new();
                while let Some((k, v)) = map.next_entry()? {
                    h.insert(k, v);
                }
                Ok(Value::Object(h))
            }
        }
        deserializer.deserialize_any(Visitor)
    }
}
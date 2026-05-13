use std::collections::BTreeMap;

use gnx_graph::{
    DataVisitor, Deserialize, Deserializer, Error, Expecting, MapAccess, SeqAccess, Serialize,
    Serializer,
};
use ordered_float::OrderedFloat;

pub type Map<K, V> = BTreeMap<K, V>;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Value {
    Bool(bool),
    Integer(i64),
    Float(OrderedFloat<f64>),
    String(String),
    OffsetDateTime(String),
    LocalDateTime(String),
    LocalDate(String),
    LocalTime(String),
    Array(Vec<Value>),
    Table(Map<String, Value>),
}

impl From<bool> for Value {
    fn from(value: bool) -> Self {
        Self::Bool(value)
    }
}

impl From<i64> for Value {
    fn from(value: i64) -> Self {
        Self::Integer(value)
    }
}

impl From<f64> for Value {
    fn from(value: f64) -> Self {
        Self::Float(OrderedFloat(value))
    }
}

impl From<String> for Value {
    fn from(value: String) -> Self {
        Self::String(value)
    }
}

impl From<&str> for Value {
    fn from(value: &str) -> Self {
        Self::String(value.to_string())
    }
}

impl Value {
    pub fn as_table_mut(&mut self) -> Option<&mut Map<String, Value>> {
        match self {
            Self::Table(table) => Some(table),
            _ => None,
        }
    }

    pub fn as_table(&self) -> Option<&Map<String, Value>> {
        match self {
            Self::Table(table) => Some(table),
            _ => None,
        }
    }
}

impl Serialize for Value {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Value::Bool(value) => serializer.serialize_bool(*value),
            Value::Integer(value) => serializer.serialize_i64(*value),
            Value::Float(value) => serializer.serialize_f64(value.0),
            Value::String(value)
            | Value::OffsetDateTime(value)
            | Value::LocalDateTime(value)
            | Value::LocalDate(value)
            | Value::LocalTime(value) => serializer.serialize_str(value),
            Value::Array(value) => serializer.collect_seq(value.iter()),
            Value::Table(value) => serializer.collect_map(value.iter()),
        }
    }
}

impl<'de> Deserialize<'de> for Value {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Visitor;

        impl Expecting for Visitor {
            fn expected(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(fmt, "any TOML-compatible value")
            }
        }

        impl<'de> DataVisitor<'de> for Visitor {
            type Value = Value;

            fn visit_bool<E: Error>(self, value: bool) -> Result<Self::Value, E> {
                Ok(Value::Bool(value))
            }

            fn visit_i64<E: Error>(self, value: i64) -> Result<Self::Value, E> {
                Ok(Value::Integer(value))
            }

            fn visit_u64<E: Error>(self, value: u64) -> Result<Self::Value, E> {
                let value = i64::try_from(value)
                    .map_err(|_| Error::custom("TOML integer is outside i64 range"))?;
                Ok(Value::Integer(value))
            }

            fn visit_f64<E: Error>(self, value: f64) -> Result<Self::Value, E> {
                Ok(Value::Float(OrderedFloat(value)))
            }

            fn visit_str<E: Error>(self, value: &str) -> Result<Self::Value, E> {
                Ok(Value::String(value.to_string()))
            }

            fn visit_string<E: Error>(self, value: String) -> Result<Self::Value, E> {
                Ok(Value::String(value))
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let mut values = Vec::new();
                while let Some(value) = seq.next_element()? {
                    values.push(value);
                }
                Ok(Value::Array(values))
            }

            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                let mut values = Map::new();
                while let Some((key, value)) = map.next_entry()? {
                    values.insert(key, value);
                }
                Ok(Value::Table(values))
            }
        }

        deserializer.deserialize_any(Visitor)
    }
}

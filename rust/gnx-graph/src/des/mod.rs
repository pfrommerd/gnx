use std::marker::PhantomData;
use std::fmt::Display;

use crate::Error;
use crate::{GraphId, GraphContext};
use gnx_expr::Array;

mod builtins;

pub trait Expecting {
    fn expected(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result;
}

pub struct Expected<'a, E: Expecting>(&'a E);

impl<'a, E: Expecting> Display for Expected<'a, E> {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.0.expected(fmt)
    }
}

pub trait Deserialize<'de>: Sized {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error>;
}

pub trait DeserializeSeed<'de>: Sized {
    type Value;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error>;
}

impl<'de, T: Deserialize<'de>> DeserializeSeed<'de> for PhantomData<T> {
    type Value = T;
    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
        T::deserialize(deserializer)
    }
}

pub trait Deserializer<'de>: Sized {
    type Error: Error;

    fn deserialize_any<V: DataVisitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error>;

    // New Graph-based deserialization methods
    fn deserialize_shared<V: DataVisitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error>;
    fn deserialize_hinted<V: DataVisitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error>;
    fn deserialize_array<V: DataVisitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error>;

    fn deserialize_bool<V: DataVisitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error>;
    fn deserialize_i8<V: DataVisitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error>;
    fn deserialize_i16<V: DataVisitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error>;
    fn deserialize_i32<V: DataVisitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error>;
    fn deserialize_i64<V: DataVisitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error>;
    fn deserialize_i128<V: DataVisitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        let _ = visitor;
        Err(Error::custom("i128 is not supported"))
    }
    fn deserialize_u8<V: DataVisitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error>;
    fn deserialize_u16<V: DataVisitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error>;
    fn deserialize_u32<V: DataVisitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error>;
    fn deserialize_u64<V: DataVisitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error>;
    fn deserialize_u128<V: DataVisitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        let _ = visitor;
        Err(Error::custom("u128 is not supported"))
    }
    fn deserialize_f32<V: DataVisitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error>;
    fn deserialize_f64<V: DataVisitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error>;
    fn deserialize_char<V: DataVisitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error>;
    fn deserialize_str<V: DataVisitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error>;
    fn deserialize_string<V: DataVisitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error>;
    fn deserialize_bytes<V: DataVisitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error>;
    fn deserialize_byte_buf<V: DataVisitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error>;
    fn deserialize_option<V: DataVisitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error>;
    fn deserialize_unit<V: DataVisitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error>;

    fn deserialize_unit_struct<V: DataVisitor<'de>>(
        self,
        name: &str,
        visitor: V,
    ) -> Result<V::Value, Self::Error>;
    fn deserialize_newtype_struct<V: DataVisitor<'de>>(
        self,
        name: &str,
        visitor: V,
    ) -> Result<V::Value, Self::Error>;

    fn deserialize_seq<V: DataVisitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error>;

    fn deserialize_tuple<V: DataVisitor<'de>>(
        self,
        len: usize,
        visitor: V,
    ) -> Result<V::Value, Self::Error>;

    fn deserialize_tuple_struct<V: DataVisitor<'de>>(
        self,
        name: &str,
        len: usize,
        visitor: V,
    ) -> Result<V::Value, Self::Error>;

    /// Hint that the `Deserialize` type is expecting a map of key-value pairs.
    fn deserialize_map<V: DataVisitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error>;
    fn deserialize_struct<V: DataVisitor<'de>>(
        self, name: &str, fields: &[&str], visitor: V,
    ) -> Result<V::Value, Self::Error>;

    fn deserialize_enum<V: DataVisitor<'de>>(
        self, name: &str, variants: &[&str], visitor: V,
    ) -> Result<V::Value, Self::Error>;

    fn deserialize_identifier<V: DataVisitor<'de>>(
        self, visitor: V,
    ) -> Result<V::Value, Self::Error>;

    fn deserialize_ignored_any<V: DataVisitor<'de>>(
        self,
        visitor: V,
    ) -> Result<V::Value, Self::Error>;

    #[inline]
    fn is_human_readable(&self) -> bool {
        true
    }
}

// Deserialization visitor
#[rustfmt::skip]
pub trait DataVisitor<'de> : Sized + Expecting {
    type Value;

    fn visit_shared<D: Deserializer<'de>>(self, id: GraphId, inner: D, ctx: &mut GraphContext) -> Result<Self::Value, D::Error> {
        let _ = (id, ctx);
        inner.deserialize_any(self)
    }
    fn visit_hinted<H: Deserialize<'de>, D: Deserializer<'de>>(self, hint: H, value: D) -> Result<Self::Value, D::Error> {
        let _ = hint;
        value.deserialize_any(self)
    }

    fn visit_array<E: Error>(self, array: Array) -> Result<Self::Value, E> {
        Err(Error::invalid_type(Some(array.shape().to_string()), "array", Expected(&self)))
    }

    fn visit_bool<E: Error>(self, v: bool) -> Result<Self::Value, E> {
        Err(Error::invalid_type(Some(v), "bool", Expected(&self)))
    }
    fn visit_i8<E: Error>(self, v: i8) -> Result<Self::Value, E> { self.visit_i64(v as i64) }
    fn visit_i16<E: Error>(self, v: i16) -> Result<Self::Value, E> { self.visit_i64(v as i64) }
    fn visit_i32<E: Error>(self, v: i32) -> Result<Self::Value, E> { self.visit_i64(v as i64) }
    fn visit_i64<E: Error>(self, v: i64) -> Result<Self::Value, E> {
        Err(Error::invalid_type(Some(v), "int", Expected(&self)))
    }
    fn visit_i128<E: Error>(self, v: i128) -> Result<Self::Value, E> {
        Err(Error::invalid_type(Some(v), "128-bit signed int", Expected(&self)))
    }

    fn visit_u8<E: Error>(self, v: u8) -> Result<Self::Value, E> { self.visit_u64(v as u64) }
    fn visit_u16<E: Error>(self, v: u16) -> Result<Self::Value, E> { self.visit_u64(v as u64) }
    fn visit_u32<E: Error>(self, v: u32) -> Result<Self::Value, E> { self.visit_u64(v as u64) }
    fn visit_u64<E: Error>(self, v: u64) -> Result<Self::Value, E> {
        Err(Error::invalid_type(Some(v), "unsigned int", Expected(&self)))
    }
    fn visit_u128<E: Error>(self, v: u128) -> Result<Self::Value, E> {
        Err(Error::invalid_type(Some(v), "128-bit unsigned int", Expected(&self)))
    }

    fn visit_f32<E: Error>(self, v: f32) -> Result<Self::Value, E> { self.visit_f64(v as f64) }
    fn visit_f64<E: Error>(self, v: f64) -> Result<Self::Value, E> {
        Err(Error::invalid_type(Some(v), "float", Expected(&self)))
    }

    #[inline]
    fn visit_char<E: Error>(self, v: char) -> Result<Self::Value, E> {
        self.visit_str(v.encode_utf8(&mut [0u8; 4]))
    }
    fn visit_str<E: Error>(self, v: &str) -> Result<Self::Value, E> {
        Err(Error::invalid_type(Some(v), "string", Expected(&self)))
    }
    #[inline]
    fn visit_borrowed_str<E: Error>(self, v: &'de str) -> Result<Self::Value, E> { self.visit_str(v) }
    #[inline]
    fn visit_string<E: Error>(self, v: String) -> Result<Self::Value, E> { self.visit_str(&v) }

    fn visit_bytes<E: Error>(self, v: &[u8]) -> Result<Self::Value, E> {
        let _ = v;
        Err(Error::invalid_type(Option::<&str>::None, "bytes", Expected(&self)))
    }

    #[inline]
    fn visit_borrowed_bytes<E: Error>(self, v: &'de [u8]) -> Result<Self::Value, E> { self.visit_bytes(v) }
    fn visit_byte_buf<E: Error>(self, v: Vec<u8>) -> Result<Self::Value, E> { self.visit_bytes(&v) }

    fn visit_none<E: Error>(self) -> Result<Self::Value, E> {
        Err(Error::invalid_type(Option::<&str>::None, "option", Expected(&self)))
    }

    fn visit_some<D: Deserializer<'de>>(self, some: D) -> Result<Self::Value, D::Error> {
        let _ = some;
        Err(Error::invalid_type(Option::<&str>::None, "some", Expected(&self)))
    }
    fn visit_unit<E: Error>(self) -> Result<Self::Value, E> {
        Err(Error::invalid_type(Option::<&str>::None, "()", Expected(&self)))
    }
    fn visit_newtype_struct<D: Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
        let _ = deserializer;
        Err(Error::invalid_type(Option::<&str>::None, "newtype struct", Expected(&self)))
    }

    fn visit_seq<A: SeqAccess<'de>>(self, seq: A) -> Result<Self::Value, A::Error> {
        let _ = seq;
        Err(Error::invalid_type(Option::<&str>::None, "sequence", Expected(&self)))
    }

    fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<Self::Value, A::Error> {
        let _ = map;
        Err(Error::invalid_type(Option::<&str>::None, "map", Expected(&self)))
    }

    fn visit_enum<A: EnumAccess<'de>>(self, data: A) -> Result<Self::Value, A::Error> {
        let _ = data;
        Err(Error::invalid_type(Option::<&str>::None, "enum", Expected(&self)))
    }
}

pub trait SeqAccess<'de> : Sized {
    type Error: Error;
    fn next_element_seed<T>(
        &mut self, 
        seed: T
    ) -> Result<Option<T::Value>, Self::Error>
    where
        T: DeserializeSeed<'de>;

    fn next_element<T>(&mut self) -> Result<Option<T>, Self::Error>
    where
        T: Deserialize<'de>,
    { 
        self.next_element_seed(PhantomData)
    }
    fn size_hint(&self) -> Option<usize> { None }
}

pub trait MapAccess<'de> : Sized {
    type Error: Error;
    fn next_key_seed<K: DeserializeSeed<'de>>(
        &mut self, 
        seed: K
    ) -> Result<Option<K::Value>, Self::Error>;
    fn next_value_seed<V: DeserializeSeed<'de>>(
        &mut self, 
        seed: V
    ) -> Result<V::Value, Self::Error>;

    fn next_entry_seed<K: DeserializeSeed<'de>, V: DeserializeSeed<'de>>(
        &mut self, 
        key_seed: K,
        value_seed: V
    ) -> Result<Option<(K::Value, V::Value)>, Self::Error> {
        let key = self.next_key_seed(key_seed)?;
        match key {
            Some(key) => Ok(Some({
                let value = self.next_value_seed(value_seed)?;
                (key, value)
            })),
            None => Ok(None)
        }
    }

    fn next_key<K: Deserialize<'de>>(&mut self) -> Result<Option<K>, Self::Error> { 
        self.next_key_seed(PhantomData)
    }
    fn next_value<V: Deserialize<'de>>(&mut self) -> Result<V, Self::Error> { 
        self.next_value_seed(PhantomData)
    }
    fn next_entry<K: Deserialize<'de>, V: Deserialize<'de>>(&mut self) -> Result<Option<(K, V)>, Self::Error> {
        self.next_entry_seed(PhantomData, PhantomData)
    }
    fn size_hint(&self) -> Option<usize> { None }
}

pub trait EnumAccess<'de>: Sized {
    type Error: Error;
    type Variant: VariantAccess<'de>;
    fn variant_seed<V: DeserializeSeed<'de>>(self, seed: V) -> Result<(V::Value, Self::Variant), Self::Error>;
    fn variant<V: Deserialize<'de>>(self) -> Result<(V, Self::Variant), Self::Error> {
        self.variant_seed(PhantomData)
    }
}

pub trait VariantAccess<'de>: Sized {
    type Error: Error;
    fn unit_variant(self) -> Result<(), Self::Error>;

    fn newtype_variant_seed<T: DeserializeSeed<'de>>(self, seed: T) -> Result<T::Value, Self::Error>;
    fn newtype_variant<T: Deserialize<'de>>(self) -> Result<T, Self::Error> {
        self.newtype_variant_seed(PhantomData)
    }

    fn tuple_variant<V: DataVisitor<'de>>(
        self, len: usize, visitor: V
    ) -> Result<V::Value, Self::Error>;
    fn struct_variant<V: DataVisitor<'de>>(
        self, fields: &[&str],  visitor: V
    ) -> Result<V::Value, Self::Error>;
}
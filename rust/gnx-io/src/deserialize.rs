use crate::visitor::DesVisitor;
use gnx::util::Error;

use std::marker::PhantomData;

pub trait GraphDeserialize<'de>: Sized {
    fn deserialize<D: GraphDeserializer<'de>>(deserializer: D) -> Result<Self, D::Error>;
}

pub trait GraphDeserializeSeed<'de>: Sized {
    type Value;

    fn deserialize<D: GraphDeserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error>;
}

impl<'de, T: GraphDeserialize<'de>> GraphDeserializeSeed<'de> for PhantomData<T> {
    type Value = T;
    fn deserialize<D: GraphDeserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
        T::deserialize(deserializer)
    }
}

pub trait GraphDeserializer<'de>: Sized {
    type Error: Error;

    fn deserialize_any<V: DesVisitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error>;

    // New Graph-based deserialization methods
    fn deserialize_shared<V: DesVisitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error>;
    fn deserialize_hinted<V: DesVisitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error>;
    fn deserialize_array<V: DesVisitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error>;

    fn deserialize_bool<V: DesVisitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error>;
    fn deserialize_i8<V: DesVisitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error>;
    fn deserialize_i16<V: DesVisitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error>;
    fn deserialize_i32<V: DesVisitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error>;
    fn deserialize_i64<V: DesVisitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error>;
    fn deserialize_i128<V: DesVisitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        let _ = visitor;
        Err(Error::custom("i128 is not supported"))
    }
    fn deserialize_u8<V: DesVisitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error>;
    fn deserialize_u16<V: DesVisitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error>;
    fn deserialize_u32<V: DesVisitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error>;
    fn deserialize_u64<V: DesVisitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error>;
    fn deserialize_u128<V: DesVisitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        let _ = visitor;
        Err(Error::custom("u128 is not supported"))
    }
    fn deserialize_f32<V: DesVisitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error>;
    fn deserialize_f64<V: DesVisitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error>;
    fn deserialize_char<V: DesVisitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error>;
    fn deserialize_str<V: DesVisitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error>;
    fn deserialize_string<V: DesVisitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error>;
    fn deserialize_bytes<V: DesVisitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error>;
    fn deserialize_byte_buf<V: DesVisitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error>;
    fn deserialize_option<V: DesVisitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error>;
    fn deserialize_unit<V: DesVisitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error>;

    fn deserialize_unit_struct<V: DesVisitor<'de>>(
        self,
        name: &'static str,
        visitor: V,
    ) -> Result<V::Value, Self::Error>;
    fn deserialize_newtype_struct<V: DesVisitor<'de>>(
        self,
        name: &'static str,
        visitor: V,
    ) -> Result<V::Value, Self::Error>;

    fn deserialize_seq<V: DesVisitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error>;

    fn deserialize_tuple<V: DesVisitor<'de>>(
        self,
        len: usize,
        visitor: V,
    ) -> Result<V::Value, Self::Error>;

    fn deserialize_tuple_struct<V: DesVisitor<'de>>(
        self,
        name: &'static str,
        len: usize,
        visitor: V,
    ) -> Result<V::Value, Self::Error>;

    /// Hint that the `Deserialize` type is expecting a map of key-value pairs.
    fn deserialize_map<V: DesVisitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error>;
    fn deserialize_struct<V: DesVisitor<'de>>(
        self,
        name: &'static str,
        fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Self::Error>;

    fn deserialize_enum<V: DesVisitor<'de>>(
        self,
        name: &'static str,
        variants: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Self::Error>;

    fn deserialize_identifier<V: DesVisitor<'de>>(
        self,
        visitor: V,
    ) -> Result<V::Value, Self::Error>;
    fn deserialize_ignored_any<V: DesVisitor<'de>>(
        self,
        visitor: V,
    ) -> Result<V::Value, Self::Error>;

    #[inline]
    fn is_human_readable(&self) -> bool {
        true
    }
}

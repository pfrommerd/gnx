use base64::{Engine as _, engine::general_purpose::STANDARD};

use crate::{Deserializer, DataVisitor};
use crate::des::{SeqAccess, MapAccess, EnumAccess, VariantAccess, DeserializeSeed};
use crate::util::{ScratchBuffer, TextSource};
use super::*;

pub struct JsonParser<S> {
    source: S,
    // Configuration flags
    // Refs are enabled by default,
    // but only for "tagged" references of the form ${#tag}
    // which must tagged before the reference can be used.
    enable_tagged_refs: bool,
    // Scratch buffer for non-str sources
    scratch: ScratchBuffer<str>,
}

impl<'src, S: TextSource<'src>> JsonParser<S> {
    pub fn new(source: S) -> Self {
        Self {
            source,
            enable_tagged_refs: true,
            scratch: ScratchBuffer::new(),
        }
    }
}

macro_rules! fn_deserialize_number {
    ($method:ident) => {
        fn_deserialize_number!($method, deserialize_number);
    };
    ($method:ident, $using:ident) => {
        fn $method<V: DataVisitor<'src>>(self, visitor: V) -> Result<V::Value> {
            self.$using(visitor)
        }
    };
}
macro_rules! test_ref {
    ($self:ident, $visitor:ident) => {
        if $self.enable_tagged_refs && $self.source.peek_type()? == ValueType::String &&
                $self.source.peek_chars(3)? == "\"${" { 
            return $self.deserialize_shared($visitor)
        }
    };
}

impl<'src, S: TextSource<'src>> JsonParser<S> {
    pub fn deserialize_number<V: DataVisitor<'src>>(&mut self, visitor: V) -> Result<V::Value> {
        test_ref!(self, visitor);
        let number = self.source.consume_number(&mut self.scratch)?;
        match number.kind {
            N::Float(f) => visitor.visit_f64(f.into_inner()),
            N::Pos(p) => visitor.visit_u64(p),
            N::Neg(n) => visitor.visit_i64(n),
        }
    }
}

// The deserializer type
impl<'src, S: TextSource<'src>> Deserializer<'src> for &mut JsonParser<S> {
    type Error = JsonError;

    fn deserialize_any<V: DataVisitor<'src>>(self, visitor: V) -> Result<V::Value> {
        match self.source.peek_type()? {
            ValueType::Null => self.deserialize_unit(visitor),
            ValueType::Bool => self.deserialize_bool(visitor),
            ValueType::Object => self.deserialize_map(visitor),
            ValueType::Array => self.deserialize_seq(visitor),
            ValueType::String => self.deserialize_string(visitor),
            ValueType::Number => self.deserialize_number(visitor),
        }
    }

    fn deserialize_unit<V: DataVisitor<'src>>(self, visitor: V) -> Result<V::Value> {
        test_ref!(self, visitor);
        self.source.consume_null()?;
        visitor.visit_unit()
    }

    fn deserialize_bool<V: DataVisitor<'src>>(self, visitor: V) -> Result<V::Value> {
        test_ref!(self, visitor);
        let bool = self.source.consume_bool()?;
        visitor.visit_bool(bool)
    }

    fn_deserialize_number!(deserialize_i8);
    fn_deserialize_number!(deserialize_i16);
    fn_deserialize_number!(deserialize_i32);
    fn_deserialize_number!(deserialize_i64);
    fn_deserialize_number!(deserialize_u8);
    fn_deserialize_number!(deserialize_u16);
    fn_deserialize_number!(deserialize_u32);
    fn_deserialize_number!(deserialize_u64);
    fn_deserialize_number!(deserialize_f32);
    fn_deserialize_number!(deserialize_f64);

    fn deserialize_char<V: DataVisitor<'src>>(self, visitor: V) -> Result<V::Value> {
        self.deserialize_str(visitor)
    }
    fn deserialize_str<V: DataVisitor<'src>>(self, visitor: V) -> std::result::Result<V::Value, Self::Error> {
        self.deserialize_string(visitor)
    }
    fn deserialize_string<V: DataVisitor<'src>>(self, visitor: V) -> Result<V::Value> {
        self.source.consume_string(&mut self.scratch)?.visit(visitor)
    }
    fn deserialize_bytes<V: DataVisitor<'src>>(self, visitor: V) -> std::result::Result<V::Value, Self::Error> {
        self.deserialize_byte_buf(visitor)
    }
    fn deserialize_byte_buf<V: DataVisitor<'src>>(self, visitor: V) -> std::result::Result<V::Value, Self::Error> {
        // If we are in human-readable mode, we require that the bytes are encoded as a list
        // of numbers between 0 and 255. Otherwise, we assume that the bytes are encoded as a base64 string.
        let text = self.source.consume_string(&mut self.scratch)?;
        let t = text.borrow();
        // Convert the text to base64
        let bytes = STANDARD.decode(&*t)?;
        visitor.visit_byte_buf(bytes)
    }

    // more complex types

    fn deserialize_option<V: DataVisitor<'src>>(self, visitor: V) -> Result<V::Value> {
        match self.source.peek_type()? {
            ValueType::Null => {
                self.source.consume_null()?;
                visitor.visit_none()
            },
            _ => {
                visitor.visit_some(self)
            }
        }
    }

    fn deserialize_unit_struct<V: DataVisitor<'src>>(
        self, _name: &str, visitor: V,
    ) -> Result<V::Value>  {
        self.deserialize_unit(visitor)
    }

    fn deserialize_newtype_struct<V: DataVisitor<'src>>(
        self, _name: &str, visitor: V,
    ) -> Result<V::Value> {
        visitor.visit_newtype_struct(self)
    }

    fn deserialize_array<V: DataVisitor<'src>>(self, visitor: V) -> Result<V::Value> {
        let _ = visitor;
        todo!()
    }

    fn deserialize_shared<V: DataVisitor<'src>>(self, visitor: V) -> Result<V::Value> {
        let _ = visitor;
        todo!()
    }

    fn deserialize_hinted<V: DataVisitor<'src>>(self, visitor: V) -> Result<V::Value> {
        let _ = visitor;
        todo!()
    }

    // map/struct deserialization
    fn deserialize_identifier<V: DataVisitor<'src>>(self, visitor: V) -> Result<V::Value> {
        // consume an identifier string
        self.source.consume_string(&mut self.scratch)?.visit(visitor)
    }

    // Deserialization of more complex types

    fn deserialize_seq<V: DataVisitor<'src>>(self, visitor: V) -> Result<V::Value> {
        test_ref!(self, visitor);

        struct ListSeqAccess<'a, S> { parser: &'a mut JsonParser<S> }
        impl<'src, S: TextSource<'src>> SeqAccess<'src> for ListSeqAccess<'_, S> {
            type Error = JsonError;

            fn next_element_seed<T: DeserializeSeed<'src>>(
                &mut self,
                seed: T,
            ) -> Result<Option<T::Value>> {
                if self.parser.source.is_array_end()? {
                    return Ok(None);
                }
                Ok(Some(seed.deserialize(&mut *self.parser)?))
            }
        }
        let access = ListSeqAccess { parser: self };
        visitor.visit_seq(access)
    }

    fn deserialize_tuple<V: DataVisitor<'src>>(
        self,
        _len: usize,
        visitor: V,
    ) -> Result<V::Value> {
        self.deserialize_seq(visitor)
    }

    fn deserialize_tuple_struct<V: DataVisitor<'src>>(
        self,
        _name: &str,
        _len: usize,
        visitor: V,
    ) -> Result<V::Value> {
        self.deserialize_seq(visitor)
    }

    fn deserialize_map<V: DataVisitor<'src>>(self, visitor: V) -> Result<V::Value> {
        struct JsonMapAccess<'a, S> { parser: &'a mut JsonParser<S> }
        impl<'a, 'src, S: TextSource<'src>> MapAccess<'src> for JsonMapAccess<'a, S> {
            type Error = JsonError;

            fn next_key_seed<K: DeserializeSeed<'src>>(
                &mut self,
                seed: K,
            ) -> Result<Option<K::Value>> {
                if self.parser.source.is_object_end()? {
                    return Ok(None);
                }
                let key = seed.deserialize(&mut *self.parser)?;
                // consume WHITESPACE : WHITESPACE after the key
                self.parser.source.consume_colon_sep()?;
                Ok(Some(key))
            }

            fn next_value_seed<VS: DeserializeSeed<'src>>(
                &mut self,
                seed: VS,
            ) -> Result<VS::Value> {
                let value = seed.deserialize(&mut *self.parser)?;
                if !self.parser.source.is_object_end()? {
                    self.parser.source.consume_delim_whitespace()?;
                }
                Ok(value)
            }
        }
        // consume the object start
        self.source.consume_object_start()?;
        let value = visitor.visit_map(JsonMapAccess { parser: self })?;
        self.source.consume_object_end()?;
        Ok(value)
    }

    // Structs should be encoded as maps.
    // This method can handle a tuple struct or a map struct
    // for maximum flexibility to support changing
    // a tuple struct into a map struct
    fn deserialize_struct<V: DataVisitor<'src>>(
        self,
        _name: &str,
        _fields: &[&str],
        visitor: V,
    ) -> Result<V::Value> {
        match self.source.peek_type()? {
            ValueType::Object => self.deserialize_map(visitor),
            ValueType::Array => self.deserialize_seq(visitor),
            _ => Err(JsonError::Unexpected(self.source.peek()?.unwrap()))
        }
    }

    fn deserialize_enum<V: DataVisitor<'src>>(
        self,
        _name: &str,
        _variants: &[&str],
        visitor: V,
    ) -> Result<V::Value> {
        struct JsonEnumAccess<'a, S> { parser: &'a mut JsonParser<S> }
        struct JsonVariantAccess<'a, S> { parser: &'a mut JsonParser<S> }
        impl<'a,'src, S: TextSource<'src>> EnumAccess<'src> for JsonEnumAccess<'a, S> {
            type Error = JsonError;
            type Variant = JsonVariantAccess<'a, S>;

            fn variant_seed<VS: DeserializeSeed<'src>>(
                self,
                seed: VS,
            ) -> Result<(VS::Value, Self::Variant)> {
                // Parse the variant type information:
                let enum_info = seed.deserialize(&mut *self.parser)?;
                let variant_access = JsonVariantAccess { parser: &mut *self.parser };
                Ok((enum_info, variant_access))
            }
        }
        impl<'a,'src, S: TextSource<'src>> VariantAccess<'src> for JsonVariantAccess<'a, S> {
            type Error = JsonError;

            fn unit_variant(self) -> Result<()> {
                Ok(())
            }
            fn newtype_variant_seed<T: DeserializeSeed<'src>>(self, seed: T) -> Result<T::Value> {
                seed.deserialize(&mut *self.parser)
            }
            fn tuple_variant<V: DataVisitor<'src>>(
                self, _len: usize, visitor: V
            ) -> Result<V::Value> {
                self.parser.deserialize_seq(visitor)
            }
            fn struct_variant<V: DataVisitor<'src>>(
                self, fields: &[&str],  visitor: V
            ) -> Result<V::Value> {
                self.parser.deserialize_struct("", fields, visitor)
            }
        }
        visitor.visit_enum(JsonEnumAccess { parser: self })
    }

    fn deserialize_ignored_any<V: DataVisitor<'src>>(self, visitor: V) -> Result<V::Value> {
        self.source.skip()?;
        visitor.visit_unit()
    }
}
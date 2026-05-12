use std::fmt::Display;
use std::io::Write;

use base64::{Engine as _, engine::general_purpose::STANDARD};
use gnx_graph::{
    GraphId, Serialize, SerializeMap, SerializeSeq, SerializeStruct, SerializeStructVariant,
    SerializeTuple, SerializeTupleStruct, SerializeTupleVariant, Serializer,
};

use super::{JsonError, Result};

pub struct JsonWriter<W> {
    writer: W,
}

impl<W> JsonWriter<W> {
    pub fn new(writer: W) -> Self {
        Self { writer }
    }

    pub fn into_inner(self) -> W {
        self.writer
    }
}

impl<W: Write> JsonWriter<W> {
    pub fn serialize<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<()> {
        value.serialize(self)
    }

    fn write_raw(&mut self, value: &str) -> Result<()> {
        self.writer.write_all(value.as_bytes())?;
        Ok(())
    }

    fn write_display<T: Display>(&mut self, value: T) -> Result<()> {
        write!(self.writer, "{value}")?;
        Ok(())
    }

    fn write_f32(&mut self, value: f32) -> Result<()> {
        if !value.is_finite() {
            return Err(JsonError::InvalidNumber);
        }
        self.write_display(value)
    }

    fn write_f64(&mut self, value: f64) -> Result<()> {
        if !value.is_finite() {
            return Err(JsonError::InvalidNumber);
        }
        self.write_display(value)
    }

    fn write_string(&mut self, value: &str) -> Result<()> {
        self.writer.write_all(b"\"")?;
        for c in value.chars() {
            match c {
                '"' => self.writer.write_all(br#"\""#)?,
                '\\' => self.writer.write_all(br#"\\"#)?,
                '\u{08}' => self.writer.write_all(br#"\b"#)?,
                '\u{0c}' => self.writer.write_all(br#"\f"#)?,
                '\n' => self.writer.write_all(br#"\n"#)?,
                '\r' => self.writer.write_all(br#"\r"#)?,
                '\t' => self.writer.write_all(br#"\t"#)?,
                '\u{00}'..='\u{1f}' => write!(self.writer, "\\u{:04x}", c as u32)?,
                c => write!(self.writer, "{c}")?,
            }
        }
        self.writer.write_all(b"\"")?;
        Ok(())
    }

    fn write_object_field<T: ?Sized + Serialize>(
        &mut self,
        first: &mut bool,
        key: &str,
        value: &T,
    ) -> Result<()> {
        if !*first {
            self.writer.write_all(b",")?;
        }
        *first = false;
        self.write_string(key)?;
        self.writer.write_all(b":")?;
        value.serialize(&mut *self)
    }
}

pub fn to_vec<T: ?Sized + Serialize>(value: &T) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    let mut writer = JsonWriter::new(&mut bytes);
    value.serialize(&mut writer)?;
    Ok(bytes)
}

pub fn to_string<T: ?Sized + Serialize>(value: &T) -> Result<String> {
    let bytes = to_vec(value)?;
    Ok(String::from_utf8(bytes).expect("JSON writer only emits UTF-8"))
}

impl<'a, W: Write> Serializer for &'a mut JsonWriter<W> {
    type Ok = ();
    type Error = JsonError;
    type SerializeSeq = Compound<'a, W, Array>;
    type SerializeTuple = Compound<'a, W, Array>;
    type SerializeTupleStruct = Compound<'a, W, Array>;
    type SerializeTupleVariant = Compound<'a, W, TupleVariant>;
    type SerializeMap = Compound<'a, W, ObjectMap>;
    type SerializeStruct = Compound<'a, W, ObjectMap>;
    type SerializeStructVariant = Compound<'a, W, StructVariant>;

    fn serialize_shared<T: ?Sized + Serialize>(self, id: GraphId, value: &T) -> Result<()> {
        self.writer.write_all(b"{")?;
        let mut first = true;
        let id: u64 = id.into();
        self.write_object_field(&mut first, "$id", &id)?;
        self.write_object_field(&mut first, "$value", value)?;
        self.writer.write_all(b"}")?;
        Ok(())
    }

    fn serialize_hinted<H: ?Sized + Serialize, T: ?Sized + Serialize>(
        self,
        hint: &H,
        value: &T,
    ) -> Result<()> {
        self.writer.write_all(b"{")?;
        let mut first = true;
        self.write_object_field(&mut first, "$hint", hint)?;
        self.write_object_field(&mut first, "$value", value)?;
        self.writer.write_all(b"}")?;
        Ok(())
    }

    fn serialize_array(self, value: gnx_expr::array::Array) -> Result<()> {
        self.writer.write_all(b"{")?;
        self.write_string("$array")?;
        self.writer.write_all(b":{")?;
        self.write_string("dtype")?;
        self.writer.write_all(b":")?;
        self.write_string(&value.dtype().to_string())?;
        self.writer.write_all(b",")?;
        self.write_string("shape")?;
        self.writer.write_all(b":")?;
        self.write_string(&value.shape().to_string())?;
        self.writer.write_all(b"}}")?;
        Ok(())
    }

    fn serialize_bool(self, value: bool) -> Result<()> {
        self.write_raw(if value { "true" } else { "false" })
    }

    fn serialize_i8(self, value: i8) -> Result<()> {
        self.write_display(value)
    }

    fn serialize_i16(self, value: i16) -> Result<()> {
        self.write_display(value)
    }

    fn serialize_i32(self, value: i32) -> Result<()> {
        self.write_display(value)
    }

    fn serialize_i64(self, value: i64) -> Result<()> {
        self.write_display(value)
    }

    fn serialize_u8(self, value: u8) -> Result<()> {
        self.write_display(value)
    }

    fn serialize_u16(self, value: u16) -> Result<()> {
        self.write_display(value)
    }

    fn serialize_u32(self, value: u32) -> Result<()> {
        self.write_display(value)
    }

    fn serialize_u64(self, value: u64) -> Result<()> {
        self.write_display(value)
    }

    fn serialize_f32(self, value: f32) -> Result<()> {
        self.write_f32(value)
    }

    fn serialize_f64(self, value: f64) -> Result<()> {
        self.write_f64(value)
    }

    fn serialize_char(self, value: char) -> Result<()> {
        self.write_string(value.encode_utf8(&mut [0; 4]))
    }

    fn serialize_str(self, value: &str) -> Result<()> {
        self.write_string(value)
    }

    fn serialize_bytes(self, value: &[u8]) -> Result<()> {
        self.write_string(&STANDARD.encode(value))
    }

    fn serialize_none(self) -> Result<()> {
        self.write_raw("null")
    }

    fn serialize_some<T>(self, value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        value.serialize(self)
    }

    fn serialize_unit(self) -> Result<()> {
        self.write_raw("null")
    }

    fn serialize_unit_struct(self, _name: &'static str) -> Result<()> {
        self.serialize_unit()
    }

    fn serialize_unit_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        variant: &'static str,
    ) -> Result<()> {
        self.write_string(variant)
    }

    fn serialize_newtype_struct<T>(self, _name: &'static str, value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        value.serialize(self)
    }

    fn serialize_newtype_variant<T>(
        self,
        _name: &'static str,
        _variant_index: u32,
        variant: &'static str,
        value: &T,
    ) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        self.writer.write_all(b"{")?;
        self.write_string(variant)?;
        self.writer.write_all(b":")?;
        value.serialize(&mut *self)?;
        self.writer.write_all(b"}")?;
        Ok(())
    }

    fn serialize_seq(self, _len: Option<usize>) -> Result<Self::SerializeSeq> {
        self.writer.write_all(b"[")?;
        Ok(Compound::new(self, Array))
    }

    fn serialize_tuple(self, _len: usize) -> Result<Self::SerializeTuple> {
        self.serialize_seq(None)
    }

    fn serialize_tuple_struct(
        self,
        _name: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeTupleStruct> {
        self.serialize_seq(None)
    }

    fn serialize_tuple_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeTupleVariant> {
        self.writer.write_all(b"{")?;
        self.write_string(variant)?;
        self.writer.write_all(b":[")?;
        Ok(Compound::new(self, TupleVariant))
    }

    fn serialize_map(self, _len: Option<usize>) -> Result<Self::SerializeMap> {
        self.writer.write_all(b"{")?;
        Ok(Compound::new(self, ObjectMap))
    }

    fn serialize_struct(self, _name: &'static str, _len: usize) -> Result<Self::SerializeStruct> {
        self.serialize_map(None)
    }

    fn serialize_struct_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeStructVariant> {
        self.writer.write_all(b"{")?;
        self.write_string(variant)?;
        self.writer.write_all(b":{")?;
        Ok(Compound::new(self, StructVariant))
    }
}

struct Array;
struct ObjectMap;
struct TupleVariant;
struct StructVariant;

struct Compound<'a, W, Kind> {
    writer: &'a mut JsonWriter<W>,
    first: bool,
    kind: Kind,
}

impl<'a, W, Kind> Compound<'a, W, Kind> {
    fn new(writer: &'a mut JsonWriter<W>, kind: Kind) -> Self {
        Self {
            writer,
            first: true,
            kind,
        }
    }
}

impl<W: Write, Kind> Compound<'_, W, Kind> {
    fn write_seq_value<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<()> {
        if !self.first {
            self.writer.writer.write_all(b",")?;
        }
        self.first = false;
        value.serialize(&mut *self.writer)
    }

    fn write_map_key<T: ?Sized + Serialize>(&mut self, key: &T) -> Result<()> {
        if !self.first {
            self.writer.writer.write_all(b",")?;
        }
        self.first = false;
        key.serialize(MapKeySerializer {
            writer: self.writer,
        })
    }

    fn write_map_value<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<()> {
        self.writer.writer.write_all(b":")?;
        value.serialize(&mut *self.writer)
    }
}

impl<W: Write> SerializeSeq for Compound<'_, W, Array> {
    type Ok = ();
    type Error = JsonError;

    fn serialize_element<T>(&mut self, value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        self.write_seq_value(value)
    }

    fn end(self) -> Result<()> {
        let _ = self.kind;
        self.writer.writer.write_all(b"]")?;
        Ok(())
    }
}

impl<W: Write> SerializeTuple for Compound<'_, W, Array> {
    type Ok = ();
    type Error = JsonError;

    fn serialize_element<T>(&mut self, value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        self.write_seq_value(value)
    }

    fn end(self) -> Result<()> {
        let _ = self.kind;
        self.writer.writer.write_all(b"]")?;
        Ok(())
    }
}

impl<W: Write> SerializeTupleStruct for Compound<'_, W, Array> {
    type Ok = ();
    type Error = JsonError;

    fn serialize_field<T>(&mut self, value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        self.write_seq_value(value)
    }

    fn end(self) -> Result<()> {
        let _ = self.kind;
        self.writer.writer.write_all(b"]")?;
        Ok(())
    }
}

impl<W: Write> SerializeTupleVariant for Compound<'_, W, TupleVariant> {
    type Ok = ();
    type Error = JsonError;

    fn serialize_field<T>(&mut self, value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        self.write_seq_value(value)
    }

    fn end(self) -> Result<()> {
        let _ = self.kind;
        self.writer.writer.write_all(b"]}")?;
        Ok(())
    }
}

impl<W: Write> SerializeMap for Compound<'_, W, ObjectMap> {
    type Ok = ();
    type Error = JsonError;

    fn serialize_key<T>(&mut self, key: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        self.write_map_key(key)
    }

    fn serialize_value<T>(&mut self, value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        self.write_map_value(value)
    }

    fn end(self) -> Result<()> {
        let _ = self.kind;
        self.writer.writer.write_all(b"}")?;
        Ok(())
    }
}

impl<W: Write> SerializeStruct for Compound<'_, W, ObjectMap> {
    type Ok = ();
    type Error = JsonError;

    fn serialize_field<T>(&mut self, key: &'static str, value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        if !self.first {
            self.writer.writer.write_all(b",")?;
        }
        self.first = false;
        self.writer.write_string(key)?;
        self.writer.writer.write_all(b":")?;
        value.serialize(&mut *self.writer)
    }

    fn end(self) -> Result<()> {
        let _ = self.kind;
        self.writer.writer.write_all(b"}")?;
        Ok(())
    }
}

impl<W: Write> SerializeStructVariant for Compound<'_, W, StructVariant> {
    type Ok = ();
    type Error = JsonError;

    fn serialize_field<T>(&mut self, key: &'static str, value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        if !self.first {
            self.writer.writer.write_all(b",")?;
        }
        self.first = false;
        self.writer.write_string(key)?;
        self.writer.writer.write_all(b":")?;
        value.serialize(&mut *self.writer)
    }

    fn end(self) -> Result<()> {
        let _ = self.kind;
        self.writer.writer.write_all(b"}}")?;
        Ok(())
    }
}

struct MapKeySerializer<'a, W> {
    writer: &'a mut JsonWriter<W>,
}

impl<W: Write> MapKeySerializer<'_, W> {
    fn write_key<T: Display>(self, value: T) -> Result<()> {
        self.writer.write_string(&value.to_string())
    }

    fn unsupported<T>(self) -> Result<T> {
        Err(JsonError::Other(
            "JSON object keys must serialize as strings or primitive scalars".to_string(),
        ))
    }
}

impl<W: Write> Serializer for MapKeySerializer<'_, W> {
    type Ok = ();
    type Error = JsonError;
    type SerializeSeq = Impossible;
    type SerializeTuple = Impossible;
    type SerializeTupleStruct = Impossible;
    type SerializeTupleVariant = Impossible;
    type SerializeMap = Impossible;
    type SerializeStruct = Impossible;
    type SerializeStructVariant = Impossible;

    fn serialize_shared<T: ?Sized + Serialize>(self, _id: GraphId, _value: &T) -> Result<()> {
        self.unsupported()
    }

    fn serialize_hinted<H: ?Sized + Serialize, T: ?Sized + Serialize>(
        self,
        _hint: &H,
        _value: &T,
    ) -> Result<()> {
        self.unsupported()
    }

    fn serialize_array(self, _value: gnx_expr::array::Array) -> Result<()> {
        self.unsupported()
    }

    fn serialize_bool(self, value: bool) -> Result<()> {
        self.write_key(value)
    }

    fn serialize_i8(self, value: i8) -> Result<()> {
        self.write_key(value)
    }

    fn serialize_i16(self, value: i16) -> Result<()> {
        self.write_key(value)
    }

    fn serialize_i32(self, value: i32) -> Result<()> {
        self.write_key(value)
    }

    fn serialize_i64(self, value: i64) -> Result<()> {
        self.write_key(value)
    }

    fn serialize_u8(self, value: u8) -> Result<()> {
        self.write_key(value)
    }

    fn serialize_u16(self, value: u16) -> Result<()> {
        self.write_key(value)
    }

    fn serialize_u32(self, value: u32) -> Result<()> {
        self.write_key(value)
    }

    fn serialize_u64(self, value: u64) -> Result<()> {
        self.write_key(value)
    }

    fn serialize_f32(self, value: f32) -> Result<()> {
        if !value.is_finite() {
            return Err(JsonError::InvalidNumber);
        }
        self.write_key(value)
    }

    fn serialize_f64(self, value: f64) -> Result<()> {
        if !value.is_finite() {
            return Err(JsonError::InvalidNumber);
        }
        self.write_key(value)
    }

    fn serialize_char(self, value: char) -> Result<()> {
        self.writer.write_string(value.encode_utf8(&mut [0; 4]))
    }

    fn serialize_str(self, value: &str) -> Result<()> {
        self.writer.write_string(value)
    }

    fn serialize_bytes(self, _value: &[u8]) -> Result<()> {
        self.unsupported()
    }

    fn serialize_none(self) -> Result<()> {
        self.write_key("null")
    }

    fn serialize_some<T>(self, value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        value.serialize(self)
    }

    fn serialize_unit(self) -> Result<()> {
        self.write_key("null")
    }

    fn serialize_unit_struct(self, _name: &'static str) -> Result<()> {
        self.serialize_unit()
    }

    fn serialize_unit_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        variant: &'static str,
    ) -> Result<()> {
        self.writer.write_string(variant)
    }

    fn serialize_newtype_struct<T>(self, _name: &'static str, value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        value.serialize(self)
    }

    fn serialize_newtype_variant<T>(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        _value: &T,
    ) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        self.unsupported()
    }

    fn serialize_seq(self, _len: Option<usize>) -> Result<Self::SerializeSeq> {
        self.unsupported()
    }

    fn serialize_tuple(self, _len: usize) -> Result<Self::SerializeTuple> {
        self.unsupported()
    }

    fn serialize_tuple_struct(
        self,
        _name: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeTupleStruct> {
        self.unsupported()
    }

    fn serialize_tuple_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeTupleVariant> {
        self.unsupported()
    }

    fn serialize_map(self, _len: Option<usize>) -> Result<Self::SerializeMap> {
        self.unsupported()
    }

    fn serialize_struct(self, _name: &'static str, _len: usize) -> Result<Self::SerializeStruct> {
        self.unsupported()
    }

    fn serialize_struct_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeStructVariant> {
        self.unsupported()
    }
}

struct Impossible;

impl SerializeSeq for Impossible {
    type Ok = ();
    type Error = JsonError;

    fn serialize_element<T>(&mut self, _value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        unreachable!("impossible JSON serializer state")
    }

    fn end(self) -> Result<()> {
        unreachable!("impossible JSON serializer state")
    }
}

impl SerializeTuple for Impossible {
    type Ok = ();
    type Error = JsonError;

    fn serialize_element<T>(&mut self, _value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        unreachable!("impossible JSON serializer state")
    }

    fn end(self) -> Result<()> {
        unreachable!("impossible JSON serializer state")
    }
}

impl SerializeTupleStruct for Impossible {
    type Ok = ();
    type Error = JsonError;

    fn serialize_field<T>(&mut self, _value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        unreachable!("impossible JSON serializer state")
    }

    fn end(self) -> Result<()> {
        unreachable!("impossible JSON serializer state")
    }
}

impl SerializeTupleVariant for Impossible {
    type Ok = ();
    type Error = JsonError;

    fn serialize_field<T>(&mut self, _value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        unreachable!("impossible JSON serializer state")
    }

    fn end(self) -> Result<()> {
        unreachable!("impossible JSON serializer state")
    }
}

impl SerializeMap for Impossible {
    type Ok = ();
    type Error = JsonError;

    fn serialize_key<T>(&mut self, _key: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        unreachable!("impossible JSON serializer state")
    }

    fn serialize_value<T>(&mut self, _value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        unreachable!("impossible JSON serializer state")
    }

    fn end(self) -> Result<()> {
        unreachable!("impossible JSON serializer state")
    }
}

impl SerializeStruct for Impossible {
    type Ok = ();
    type Error = JsonError;

    fn serialize_field<T>(&mut self, _key: &'static str, _value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        unreachable!("impossible JSON serializer state")
    }

    fn end(self) -> Result<()> {
        unreachable!("impossible JSON serializer state")
    }
}

impl SerializeStructVariant for Impossible {
    type Ok = ();
    type Error = JsonError;

    fn serialize_field<T>(&mut self, _key: &'static str, _value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        unreachable!("impossible JSON serializer state")
    }

    fn end(self) -> Result<()> {
        unreachable!("impossible JSON serializer state")
    }
}

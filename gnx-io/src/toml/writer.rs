use std::fmt::Display;
use std::io::Write;

use gnx_graph::{
    GraphId, Serialize, SerializeMap, SerializeSeq, SerializeStruct, SerializeStructVariant,
    SerializeTuple, SerializeTupleStruct, SerializeTupleVariant, Serializer,
};

use super::{Result, TomlError, Value};

pub struct TomlWriter<W> {
    writer: W,
    depth: usize,
}

impl<W> TomlWriter<W> {
    pub fn new(writer: W) -> Self {
        Self { writer, depth: 0 }
    }

    pub fn into_inner(self) -> W {
        self.writer
    }
}

impl<W: Write> TomlWriter<W> {
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

    fn serialize_nested<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<()> {
        self.depth += 1;
        let result = value.serialize(&mut *self);
        self.depth -= 1;
        result
    }

    fn serialize_root_value<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<()> {
        value.serialize(&mut *self)
    }

    fn write_float(&mut self, value: f64) -> Result<()> {
        if value.is_nan() {
            self.write_raw("nan")
        } else if value == f64::INFINITY {
            self.write_raw("inf")
        } else if value == f64::NEG_INFINITY {
            self.write_raw("-inf")
        } else {
            self.write_display(value)
        }
    }

    fn write_key(&mut self, value: &str) -> Result<()> {
        if is_bare_key(value) {
            self.write_raw(value)
        } else {
            self.write_string(value)
        }
    }

    fn write_string(&mut self, value: &str) -> Result<()> {
        self.writer.write_all(b"\"")?;
        for ch in value.chars() {
            match ch {
                '\u{08}' => self.writer.write_all(br#"\b"#)?,
                '\t' => self.writer.write_all(br#"\t"#)?,
                '\n' => self.writer.write_all(br#"\n"#)?,
                '\u{0c}' => self.writer.write_all(br#"\f"#)?,
                '\r' => self.writer.write_all(br#"\r"#)?,
                '\u{1b}' => self.writer.write_all(br#"\e"#)?,
                '"' => self.writer.write_all(br#"\""#)?,
                '\\' => self.writer.write_all(br#"\\"#)?,
                '\u{00}'..='\u{1f}' | '\u{7f}' => write!(self.writer, "\\u{:04X}", ch as u32)?,
                ch => write!(self.writer, "{ch}")?,
            }
        }
        self.writer.write_all(b"\"")?;
        Ok(())
    }
}

pub fn to_vec<T: ?Sized + Serialize>(value: &T) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    let mut writer = TomlWriter::new(&mut bytes);
    value.serialize(&mut writer)?;
    Ok(bytes)
}

pub fn to_string<T: ?Sized + Serialize>(value: &T) -> Result<String> {
    let bytes = to_vec(value)?;
    Ok(String::from_utf8(bytes).expect("TOML writer only emits UTF-8"))
}

pub fn value_to_string(value: &Value) -> Result<String> {
    let mut bytes = Vec::new();
    write_value_document(&mut bytes, value)?;
    Ok(String::from_utf8(bytes).expect("TOML writer only emits UTF-8"))
}

impl<'a, W: Write> Serializer for &'a mut TomlWriter<W> {
    type Ok = ();
    type Error = TomlError;
    type SerializeSeq = TomlCompound<'a, W, TomlArray>;
    type SerializeTuple = TomlCompound<'a, W, TomlArray>;
    type SerializeTupleStruct = TomlCompound<'a, W, TomlArray>;
    type SerializeTupleVariant = TomlCompound<'a, W, TomlInlineTable>;
    type SerializeMap = TomlCompound<'a, W, TomlTable>;
    type SerializeStruct = TomlCompound<'a, W, TomlTable>;
    type SerializeStructVariant = TomlCompound<'a, W, TomlInlineTable>;

    fn serialize_shared<T: ?Sized + Serialize>(self, id: GraphId, value: &T) -> Result<()> {
        if self.depth == 0 {
            self.write_raw("$id = ")?;
            let id: u64 = id.into();
            self.serialize_nested(&id)?;
            self.write_raw("\n$value = ")?;
            self.serialize_nested(value)
        } else {
            self.write_raw("{ ")?;
            self.write_key("$id")?;
            self.write_raw(" = ")?;
            let id: u64 = id.into();
            self.serialize_nested(&id)?;
            self.write_raw(", ")?;
            self.write_key("$value")?;
            self.write_raw(" = ")?;
            self.serialize_nested(value)?;
            self.write_raw(" }")
        }
    }

    fn serialize_hinted<H: ?Sized + Serialize, T: ?Sized + Serialize>(
        self,
        hint: &H,
        value: &T,
    ) -> Result<()> {
        if self.depth == 0 {
            self.write_raw("$hint = ")?;
            self.serialize_nested(hint)?;
            self.write_raw("\n$value = ")?;
            self.serialize_nested(value)
        } else {
            self.write_raw("{ ")?;
            self.write_key("$hint")?;
            self.write_raw(" = ")?;
            self.serialize_nested(hint)?;
            self.write_raw(", ")?;
            self.write_key("$value")?;
            self.write_raw(" = ")?;
            self.serialize_nested(value)?;
            self.write_raw(" }")
        }
    }

    fn serialize_array(self, value: gnx_expr::array::Array) -> Result<()> {
        if self.depth == 0 {
            return Err(TomlError::RootMustBeTable);
        }
        self.write_raw("{ ")?;
        self.write_key("$array")?;
        self.write_raw(" = { dtype = ")?;
        self.write_string(&value.dtype().to_string())?;
        self.write_raw(", shape = ")?;
        self.write_string(&value.shape().to_string())?;
        self.write_raw(" } }")
    }

    fn serialize_bool(self, value: bool) -> Result<()> {
        if self.depth == 0 {
            return Err(TomlError::RootMustBeTable);
        }
        self.write_raw(if value { "true" } else { "false" })
    }

    fn serialize_i8(self, value: i8) -> Result<()> {
        self.serialize_i64(value as i64)
    }

    fn serialize_i16(self, value: i16) -> Result<()> {
        self.serialize_i64(value as i64)
    }

    fn serialize_i32(self, value: i32) -> Result<()> {
        self.serialize_i64(value as i64)
    }

    fn serialize_i64(self, value: i64) -> Result<()> {
        if self.depth == 0 {
            return Err(TomlError::RootMustBeTable);
        }
        self.write_display(value)
    }

    fn serialize_u8(self, value: u8) -> Result<()> {
        self.serialize_u64(value as u64)
    }

    fn serialize_u16(self, value: u16) -> Result<()> {
        self.serialize_u64(value as u64)
    }

    fn serialize_u32(self, value: u32) -> Result<()> {
        self.serialize_u64(value as u64)
    }

    fn serialize_u64(self, value: u64) -> Result<()> {
        if self.depth == 0 {
            return Err(TomlError::RootMustBeTable);
        }
        let value = i64::try_from(value).map_err(|_| TomlError::InvalidNumber)?;
        self.write_display(value)
    }

    fn serialize_f32(self, value: f32) -> Result<()> {
        self.serialize_f64(value as f64)
    }

    fn serialize_f64(self, value: f64) -> Result<()> {
        if self.depth == 0 {
            return Err(TomlError::RootMustBeTable);
        }
        self.write_float(value)
    }

    fn serialize_char(self, value: char) -> Result<()> {
        self.serialize_str(value.encode_utf8(&mut [0; 4]))
    }

    fn serialize_str(self, value: &str) -> Result<()> {
        if self.depth == 0 {
            return Err(TomlError::RootMustBeTable);
        }
        self.write_string(value)
    }

    fn serialize_bytes(self, value: &[u8]) -> Result<()> {
        if self.depth == 0 {
            return Err(TomlError::RootMustBeTable);
        }
        self.write_raw("[")?;
        for (idx, byte) in value.iter().enumerate() {
            if idx > 0 {
                self.write_raw(", ")?;
            }
            self.write_display(*byte)?;
        }
        self.write_raw("]")
    }

    fn serialize_none(self) -> Result<()> {
        Err(TomlError::Unsupported("none/null values"))
    }

    fn serialize_some<T>(self, value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        if self.depth == 0 {
            self.serialize_root_value(value)
        } else {
            self.serialize_nested(value)
        }
    }

    fn serialize_unit(self) -> Result<()> {
        Err(TomlError::Unsupported("unit/null values"))
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
        self.serialize_str(variant)
    }

    fn serialize_newtype_struct<T>(self, _name: &'static str, value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        if self.depth == 0 {
            self.serialize_root_value(value)
        } else {
            self.serialize_nested(value)
        }
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
        if self.depth == 0 {
            self.write_key(variant)?;
            self.write_raw(" = ")?;
            self.serialize_nested(value)
        } else {
            self.write_raw("{ ")?;
            self.write_key(variant)?;
            self.write_raw(" = ")?;
            self.serialize_nested(value)?;
            self.write_raw(" }")
        }
    }

    fn serialize_seq(self, _len: Option<usize>) -> Result<Self::SerializeSeq> {
        if self.depth == 0 {
            return Err(TomlError::RootMustBeTable);
        }
        self.write_raw("[")?;
        Ok(TomlCompound::new(self, TomlArray, false))
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
        if self.depth == 0 {
            return Err(TomlError::RootMustBeTable);
        }
        self.write_raw("{ ")?;
        self.write_key(variant)?;
        self.write_raw(" = [")?;
        Ok(TomlCompound::new(self, TomlInlineTable, false))
    }

    fn serialize_map(self, _len: Option<usize>) -> Result<Self::SerializeMap> {
        let root = self.depth == 0;
        if !root {
            self.write_raw("{ ")?;
        }
        Ok(TomlCompound::new(self, TomlTable, root))
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
        if self.depth == 0 {
            return Err(TomlError::RootMustBeTable);
        }
        self.write_raw("{ ")?;
        self.write_key(variant)?;
        self.write_raw(" = { ")?;
        Ok(TomlCompound::new(self, TomlInlineTable, false))
    }
}

#[doc(hidden)]
pub struct TomlArray;
#[doc(hidden)]
pub struct TomlTable;
#[doc(hidden)]
pub struct TomlInlineTable;

#[doc(hidden)]
pub struct TomlCompound<'a, W, Kind> {
    writer: &'a mut TomlWriter<W>,
    first: bool,
    root: bool,
    kind: Kind,
}

impl<'a, W, Kind> TomlCompound<'a, W, Kind> {
    fn new(writer: &'a mut TomlWriter<W>, kind: Kind, root: bool) -> Self {
        Self {
            writer,
            first: true,
            root,
            kind,
        }
    }
}

impl<W: Write, Kind> TomlCompound<'_, W, Kind> {
    fn write_seq_value<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<()> {
        if !self.first {
            self.writer.write_raw(", ")?;
        }
        self.first = false;
        self.writer.serialize_nested(value)
    }

    fn write_map_key<T: ?Sized + Serialize>(&mut self, key: &T) -> Result<()> {
        if self.root {
            if !self.first {
                self.writer.write_raw("\n")?;
            }
        } else if !self.first {
            self.writer.write_raw(", ")?;
        }
        self.first = false;
        key.serialize(MapKeySerializer {
            writer: self.writer,
        })?;
        self.writer.write_raw(" = ")
    }

    fn write_map_value<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<()> {
        self.writer.serialize_nested(value)
    }
}

impl<W: Write> SerializeSeq for TomlCompound<'_, W, TomlArray> {
    type Ok = ();
    type Error = TomlError;

    fn serialize_element<T>(&mut self, value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        self.write_seq_value(value)
    }

    fn end(self) -> Result<()> {
        let _ = self.kind;
        self.writer.write_raw("]")
    }
}

impl<W: Write> SerializeTuple for TomlCompound<'_, W, TomlArray> {
    type Ok = ();
    type Error = TomlError;

    fn serialize_element<T>(&mut self, value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        self.write_seq_value(value)
    }

    fn end(self) -> Result<()> {
        let _ = self.kind;
        self.writer.write_raw("]")
    }
}

impl<W: Write> SerializeTupleStruct for TomlCompound<'_, W, TomlArray> {
    type Ok = ();
    type Error = TomlError;

    fn serialize_field<T>(&mut self, value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        self.write_seq_value(value)
    }

    fn end(self) -> Result<()> {
        let _ = self.kind;
        self.writer.write_raw("]")
    }
}

impl<W: Write> SerializeTupleVariant for TomlCompound<'_, W, TomlInlineTable> {
    type Ok = ();
    type Error = TomlError;

    fn serialize_field<T>(&mut self, value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        self.write_seq_value(value)
    }

    fn end(self) -> Result<()> {
        let _ = self.kind;
        self.writer.write_raw("] }")
    }
}

impl<W: Write> SerializeMap for TomlCompound<'_, W, TomlTable> {
    type Ok = ();
    type Error = TomlError;

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
        if self.root {
            Ok(())
        } else {
            self.writer.write_raw(" }")
        }
    }
}

impl<W: Write> SerializeStruct for TomlCompound<'_, W, TomlTable> {
    type Ok = ();
    type Error = TomlError;

    fn serialize_field<T>(&mut self, key: &'static str, value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        if self.root {
            if !self.first {
                self.writer.write_raw("\n")?;
            }
        } else if !self.first {
            self.writer.write_raw(", ")?;
        }
        self.first = false;
        self.writer.write_key(key)?;
        self.writer.write_raw(" = ")?;
        self.writer.serialize_nested(value)
    }

    fn end(self) -> Result<()> {
        let _ = self.kind;
        if self.root {
            Ok(())
        } else {
            self.writer.write_raw(" }")
        }
    }
}

impl<W: Write> SerializeStructVariant for TomlCompound<'_, W, TomlInlineTable> {
    type Ok = ();
    type Error = TomlError;

    fn serialize_field<T>(&mut self, key: &'static str, value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        if !self.first {
            self.writer.write_raw(", ")?;
        }
        self.first = false;
        self.writer.write_key(key)?;
        self.writer.write_raw(" = ")?;
        self.writer.serialize_nested(value)
    }

    fn end(self) -> Result<()> {
        let _ = self.kind;
        self.writer.write_raw(" } }")
    }
}

struct MapKeySerializer<'a, W> {
    writer: &'a mut TomlWriter<W>,
}

impl<W: Write> MapKeySerializer<'_, W> {
    fn write_key<T: Display>(self, value: T) -> Result<()> {
        self.writer.write_key(&value.to_string())
    }

    fn unsupported<T>(self) -> Result<T> {
        Err(TomlError::Unsupported("TOML table keys must be scalar"))
    }
}

impl<W: Write> Serializer for MapKeySerializer<'_, W> {
    type Ok = ();
    type Error = TomlError;
    type SerializeSeq = TomlImpossible;
    type SerializeTuple = TomlImpossible;
    type SerializeTupleStruct = TomlImpossible;
    type SerializeTupleVariant = TomlImpossible;
    type SerializeMap = TomlImpossible;
    type SerializeStruct = TomlImpossible;
    type SerializeStructVariant = TomlImpossible;

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
        self.write_key(value)
    }

    fn serialize_f64(self, value: f64) -> Result<()> {
        self.write_key(value)
    }

    fn serialize_char(self, value: char) -> Result<()> {
        self.writer.write_key(value.encode_utf8(&mut [0; 4]))
    }

    fn serialize_str(self, value: &str) -> Result<()> {
        self.writer.write_key(value)
    }

    fn serialize_bytes(self, _value: &[u8]) -> Result<()> {
        self.unsupported()
    }

    fn serialize_none(self) -> Result<()> {
        self.unsupported()
    }

    fn serialize_some<T>(self, value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        value.serialize(self)
    }

    fn serialize_unit(self) -> Result<()> {
        self.unsupported()
    }

    fn serialize_unit_struct(self, _name: &'static str) -> Result<()> {
        self.unsupported()
    }

    fn serialize_unit_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        variant: &'static str,
    ) -> Result<()> {
        self.writer.write_key(variant)
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

#[doc(hidden)]
pub struct TomlImpossible;

macro_rules! impossible {
    () => {
        unreachable!("impossible TOML serializer state")
    };
}

impl SerializeSeq for TomlImpossible {
    type Ok = ();
    type Error = TomlError;

    fn serialize_element<T>(&mut self, _value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        impossible!()
    }

    fn end(self) -> Result<()> {
        impossible!()
    }
}

impl SerializeTuple for TomlImpossible {
    type Ok = ();
    type Error = TomlError;

    fn serialize_element<T>(&mut self, _value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        impossible!()
    }

    fn end(self) -> Result<()> {
        impossible!()
    }
}

impl SerializeTupleStruct for TomlImpossible {
    type Ok = ();
    type Error = TomlError;

    fn serialize_field<T>(&mut self, _value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        impossible!()
    }

    fn end(self) -> Result<()> {
        impossible!()
    }
}

impl SerializeTupleVariant for TomlImpossible {
    type Ok = ();
    type Error = TomlError;

    fn serialize_field<T>(&mut self, _value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        impossible!()
    }

    fn end(self) -> Result<()> {
        impossible!()
    }
}

impl SerializeMap for TomlImpossible {
    type Ok = ();
    type Error = TomlError;

    fn serialize_key<T>(&mut self, _key: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        impossible!()
    }

    fn serialize_value<T>(&mut self, _value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        impossible!()
    }

    fn end(self) -> Result<()> {
        impossible!()
    }
}

impl SerializeStruct for TomlImpossible {
    type Ok = ();
    type Error = TomlError;

    fn serialize_field<T>(&mut self, _key: &'static str, _value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        impossible!()
    }

    fn end(self) -> Result<()> {
        impossible!()
    }
}

impl SerializeStructVariant for TomlImpossible {
    type Ok = ();
    type Error = TomlError;

    fn serialize_field<T>(&mut self, _key: &'static str, _value: &T) -> Result<()>
    where
        T: ?Sized + Serialize,
    {
        impossible!()
    }

    fn end(self) -> Result<()> {
        impossible!()
    }
}

fn is_bare_key(value: &str) -> bool {
    !value.is_empty()
        && value
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '-')
}

fn write_value_document<W: Write>(writer: &mut W, value: &Value) -> Result<()> {
    let Value::Table(table) = value else {
        return Err(TomlError::RootMustBeTable);
    };
    let mut first = true;
    for (key, value) in table {
        if !first {
            writer.write_all(b"\n")?;
        }
        first = false;
        write_value_key(writer, key)?;
        writer.write_all(b" = ")?;
        write_value_inline(writer, value)?;
    }
    Ok(())
}

fn write_value_inline<W: Write>(writer: &mut W, value: &Value) -> Result<()> {
    match value {
        Value::Bool(value) => writer.write_all(if *value { b"true" } else { b"false" })?,
        Value::Integer(value) => write!(writer, "{value}")?,
        Value::Float(value) if value.0.is_nan() => writer.write_all(b"nan")?,
        Value::Float(value) if value.0 == f64::INFINITY => writer.write_all(b"inf")?,
        Value::Float(value) if value.0 == f64::NEG_INFINITY => writer.write_all(b"-inf")?,
        Value::Float(value) => write!(writer, "{}", value.0)?,
        Value::String(value) => write_value_string(writer, value)?,
        Value::OffsetDateTime(value)
        | Value::LocalDateTime(value)
        | Value::LocalDate(value)
        | Value::LocalTime(value) => writer.write_all(value.as_bytes())?,
        Value::Array(values) => {
            writer.write_all(b"[")?;
            for (idx, item) in values.iter().enumerate() {
                if idx > 0 {
                    writer.write_all(b", ")?;
                }
                write_value_inline(writer, item)?;
            }
            writer.write_all(b"]")?;
        }
        Value::Table(values) => {
            writer.write_all(b"{ ")?;
            for (idx, (key, item)) in values.iter().enumerate() {
                if idx > 0 {
                    writer.write_all(b", ")?;
                }
                write_value_key(writer, key)?;
                writer.write_all(b" = ")?;
                write_value_inline(writer, item)?;
            }
            writer.write_all(b" }")?;
        }
    }
    Ok(())
}

fn write_value_key<W: Write>(writer: &mut W, key: &str) -> Result<()> {
    if is_bare_key(key) {
        writer.write_all(key.as_bytes())?;
    } else {
        write_value_string(writer, key)?;
    }
    Ok(())
}

fn write_value_string<W: Write>(writer: &mut W, value: &str) -> Result<()> {
    writer.write_all(b"\"")?;
    for ch in value.chars() {
        match ch {
            '\u{08}' => writer.write_all(br#"\b"#)?,
            '\t' => writer.write_all(br#"\t"#)?,
            '\n' => writer.write_all(br#"\n"#)?,
            '\u{0c}' => writer.write_all(br#"\f"#)?,
            '\r' => writer.write_all(br#"\r"#)?,
            '\u{1b}' => writer.write_all(br#"\e"#)?,
            '"' => writer.write_all(br#"\""#)?,
            '\\' => writer.write_all(br#"\\"#)?,
            '\u{00}'..='\u{1f}' | '\u{7f}' => write!(writer, "\\u{:04X}", ch as u32)?,
            ch => write!(writer, "{ch}")?,
        }
    }
    writer.write_all(b"\"")?;
    Ok(())
}

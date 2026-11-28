//! Forward entries of a flattened field into an already open map.
use super::*;

#[doc(hidden)]
pub struct Flat<'a, M>(pub &'a mut M);
fn invalid<T, E: Error>() -> Result<T, E> {
    Err(E::custom("expected a map or struct for flattened content"))
}
macro_rules! reject { ($($f:ident($($t:ty),*) ),*) => {$(fn $f(self,$(_:$t),*)->Result<(),M::Error>{invalid()})*}; }
impl<'a, M: SerializeMap> Serializer for Flat<'a, M> {
    type Ok = ();
    type Error = M::Error;
    type SerializeSeq = Self;
    type SerializeTuple = Self;
    type SerializeTupleStruct = Self;
    type SerializeTupleVariant = NestedVariant<'a, M>;
    type SerializeMap = Self;
    type SerializeStruct = Self;
    type SerializeStructVariant = NestedVariant<'a, M>;
    reject!(
        serialize_bool(bool),
        serialize_i8(i8),
        serialize_i16(i16),
        serialize_i32(i32),
        serialize_i64(i64),
        serialize_u8(u8),
        serialize_u16(u16),
        serialize_u32(u32),
        serialize_u64(u64),
        serialize_f32(f32),
        serialize_f64(f64),
        serialize_char(char),
        serialize_str(&str),
        serialize_bytes(&[u8]),
        serialize_array(Array)
    );
    fn serialize_shared<T: ?Sized + Serialize>(self, _: GraphId, _: &T) -> Result<(), M::Error> {
        invalid()
    }
    fn serialize_hinted<H: ?Sized + Serialize, T: ?Sized + Serialize>(
        self,
        _: &H,
        _: &T,
    ) -> Result<(), M::Error> {
        invalid()
    }
    fn serialize_none(self) -> Result<(), M::Error> {
        Ok(())
    }
    fn serialize_unit(self) -> Result<(), M::Error> {
        Ok(())
    }
    fn serialize_unit_struct(self, _: &'static str) -> Result<(), M::Error> {
        Ok(())
    }
    fn serialize_some<T: ?Sized + Serialize>(self, v: &T) -> Result<(), M::Error> {
        v.serialize(self)
    }
    fn serialize_newtype_struct<T: ?Sized + Serialize>(
        self,
        _: &'static str,
        v: &T,
    ) -> Result<(), M::Error> {
        v.serialize(self)
    }
    fn serialize_unit_variant(
        self,
        _: &'static str,
        _: u32,
        v: &'static str,
    ) -> Result<(), M::Error> {
        self.0.serialize_entry(v, &())
    }
    fn serialize_newtype_variant<T: ?Sized + Serialize>(
        self,
        _: &'static str,
        _: u32,
        k: &'static str,
        v: &T,
    ) -> Result<(), M::Error> {
        self.0.serialize_entry(k, v)
    }
    fn serialize_map(self, _: Option<usize>) -> Result<Self, M::Error> {
        Ok(self)
    }
    fn serialize_struct(self, _: &'static str, _: usize) -> Result<Self, M::Error> {
        Ok(self)
    }
    fn serialize_seq(self, _: Option<usize>) -> Result<Self, M::Error> {
        invalid()
    }
    fn serialize_tuple(self, _: usize) -> Result<Self, M::Error> {
        invalid()
    }
    fn serialize_tuple_struct(self, _: &'static str, _: usize) -> Result<Self, M::Error> {
        invalid()
    }
    fn serialize_tuple_variant(
        self,
        _: &'static str,
        _: u32,
        k: &'static str,
        _: usize,
    ) -> Result<Self::SerializeTupleVariant, M::Error> {
        Ok(NestedVariant(self.0, k, buffer::Value::Seq(Vec::new())))
    }
    fn serialize_struct_variant(
        self,
        _: &'static str,
        _: u32,
        k: &'static str,
        _: usize,
    ) -> Result<Self::SerializeStructVariant, M::Error> {
        Ok(NestedVariant(self.0, k, buffer::Value::Map(Vec::new())))
    }
}
#[doc(hidden)]
pub struct NestedVariant<'a, M>(&'a mut M, &'static str, buffer::Value);
impl<M: SerializeMap> SerializeTupleVariant for NestedVariant<'_, M> {
    type Ok = ();
    type Error = M::Error;
    fn serialize_field<T: ?Sized + Serialize>(&mut self, v: &T) -> Result<(), M::Error> {
        let buffer::Value::Seq(items) = &mut self.2 else {
            unreachable!()
        };
        items.push(buffer::capture(v)?);
        Ok(())
    }
    fn end(self) -> Result<(), M::Error> {
        self.0.serialize_entry(self.1, &self.2)
    }
}
impl<M: SerializeMap> SerializeStructVariant for NestedVariant<'_, M> {
    type Ok = ();
    type Error = M::Error;
    fn serialize_field<T: ?Sized + Serialize>(
        &mut self,
        k: &'static str,
        v: &T,
    ) -> Result<(), M::Error> {
        let buffer::Value::Map(items) = &mut self.2 else {
            unreachable!()
        };
        items.push((buffer::Value::String(k.into()), buffer::capture(v)?));
        Ok(())
    }
    fn end(self) -> Result<(), M::Error> {
        self.0.serialize_entry(self.1, &self.2)
    }
}
impl<M: SerializeMap> SerializeMap for Flat<'_, M> {
    type Ok = ();
    type Error = M::Error;
    fn serialize_key<T: ?Sized + Serialize>(&mut self, k: &T) -> Result<(), M::Error> {
        self.0.serialize_key(k)
    }
    fn serialize_value<T: ?Sized + Serialize>(&mut self, v: &T) -> Result<(), M::Error> {
        self.0.serialize_value(v)
    }
    fn end(self) -> Result<(), M::Error> {
        Ok(())
    }
}
impl<M: SerializeMap> SerializeStruct for Flat<'_, M> {
    type Ok = ();
    type Error = M::Error;
    fn serialize_field<T: ?Sized + Serialize>(
        &mut self,
        k: &'static str,
        v: &T,
    ) -> Result<(), M::Error> {
        self.0.serialize_entry(k, v)
    }
    fn end(self) -> Result<(), M::Error> {
        Ok(())
    }
}
macro_rules! impossible{($tr:ident,$f:ident $(,$key:ty)?)=>{impl<M:SerializeMap> $tr for Flat<'_,M>{type Ok=();type Error=M::Error;fn $f<T:?Sized+Serialize>(&mut self,$(_:$key,)?_:&T)->Result<(),M::Error>{invalid()}fn end(self)->Result<(),M::Error>{invalid()}}};}
impossible!(SerializeSeq, serialize_element);
impossible!(SerializeTuple, serialize_element);
impossible!(SerializeTupleStruct, serialize_field);
impossible!(SerializeTupleVariant, serialize_field);
impossible!(SerializeStructVariant, serialize_field, &'static str);

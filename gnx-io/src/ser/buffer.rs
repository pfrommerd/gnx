//! A flattened compound variant must be passed to the parent map as one value,
//! while its fields arrive in separate calls. Buffer just that payload.
use super::*;
use std::marker::PhantomData;

pub(super) enum Value {
    Unit,
    Bool(bool),
    I(i64),
    U(u64),
    I128(i128),
    U128(u128),
    F(f64),
    String(String),
    Bytes(Vec<u8>),
    Seq(Vec<Self>),
    Map(Vec<(Self, Self)>),
    Array(Array),
    Shared(GraphId, Box<Self>),
    Hinted(Box<Self>, Box<Self>),
}
impl Serialize for Value {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Unit => s.serialize_unit(),
            Self::Bool(x) => s.serialize_bool(*x),
            Self::I(x) => s.serialize_i64(*x),
            Self::U(x) => s.serialize_u64(*x),
            Self::I128(x) => s.serialize_i128(*x),
            Self::U128(x) => s.serialize_u128(*x),
            Self::F(x) => s.serialize_f64(*x),
            Self::String(x) => s.serialize_str(x),
            Self::Bytes(x) => s.serialize_bytes(x),
            Self::Seq(x) => s.collect_seq(x),
            Self::Map(x) => s.collect_map(x.iter().map(|(k, v)| (k, v))),
            Self::Array(x) => s.serialize_array(x.clone()),
            Self::Shared(id, v) => s.serialize_shared(*id, &**v),
            Self::Hinted(h, v) => s.serialize_hinted(&**h, &**v),
        }
    }
}
pub(super) fn capture<T: ?Sized + Serialize, E: Error>(v: &T) -> Result<Value, E> {
    v.serialize(Buffer(PhantomData))
}
struct Buffer<E>(PhantomData<E>);
struct Items<E> {
    value: Value,
    key: Option<Value>,
    variant: Option<&'static str>,
    marker: PhantomData<E>,
}
impl<E: Error> Items<E> {
    fn new(value: Value, variant: Option<&'static str>) -> Self {
        Self {
            value,
            key: None,
            variant,
            marker: PhantomData,
        }
    }
    fn finish(self) -> Result<Value, E> {
        Ok(match self.variant {
            Some(k) => Value::Map(vec![(Value::String(k.into()), self.value)]),
            None => self.value,
        })
    }
    fn push<T: ?Sized + Serialize>(&mut self, v: &T) -> Result<(), E> {
        let Value::Seq(seq) = &mut self.value else {
            unreachable!()
        };
        seq.push(capture(v)?);
        Ok(())
    }
    fn entry<T: ?Sized + Serialize>(&mut self, k: Value, v: &T) -> Result<(), E> {
        let Value::Map(map) = &mut self.value else {
            unreachable!()
        };
        map.push((k, capture(v)?));
        Ok(())
    }
}
macro_rules! scalar{($($f:ident($ty:ty)=>$v:ident),*)=>{$(fn $f(self,x:$ty)->Result<Value,E>{Ok(Value::$v(x.into()))})*};}
impl<E: Error> Serializer for Buffer<E> {
    type Ok = Value;
    type Error = E;
    type SerializeSeq = Items<E>;
    type SerializeTuple = Items<E>;
    type SerializeTupleStruct = Items<E>;
    type SerializeTupleVariant = Items<E>;
    type SerializeStruct = Items<E>;
    type SerializeStructVariant = Items<E>;
    type SerializeMap = Items<E>;
    scalar!(serialize_bool(bool)=>Bool,serialize_i8(i8)=>I,serialize_i16(i16)=>I,serialize_i32(i32)=>I,serialize_i64(i64)=>I,serialize_i128(i128)=>I128,serialize_u8(u8)=>U,serialize_u16(u16)=>U,serialize_u32(u32)=>U,serialize_u64(u64)=>U,serialize_u128(u128)=>U128,serialize_f32(f32)=>F,serialize_f64(f64)=>F,serialize_char(char)=>String,serialize_str(&str)=>String,serialize_bytes(&[u8])=>Bytes,serialize_array(Array)=>Array);
    fn serialize_unit(self) -> Result<Value, E> {
        Ok(Value::Unit)
    }
    fn serialize_unit_struct(self, _: &'static str) -> Result<Value, E> {
        self.serialize_unit()
    }
    fn serialize_none(self) -> Result<Value, E> {
        self.serialize_unit()
    }
    fn serialize_some<T: ?Sized + Serialize>(self, v: &T) -> Result<Value, E> {
        capture(v)
    }
    fn serialize_newtype_struct<T: ?Sized + Serialize>(
        self,
        _: &'static str,
        v: &T,
    ) -> Result<Value, E> {
        capture(v)
    }
    fn serialize_shared<T: ?Sized + Serialize>(self, id: GraphId, v: &T) -> Result<Value, E> {
        Ok(Value::Shared(id, Box::new(capture(v)?)))
    }
    fn serialize_hinted<H: ?Sized + Serialize, T: ?Sized + Serialize>(
        self,
        h: &H,
        v: &T,
    ) -> Result<Value, E> {
        Ok(Value::Hinted(Box::new(capture(h)?), Box::new(capture(v)?)))
    }
    fn serialize_unit_variant(self, _: &'static str, _: u32, k: &'static str) -> Result<Value, E> {
        self.serialize_str(k)
    }
    fn serialize_newtype_variant<T: ?Sized + Serialize>(
        self,
        _: &'static str,
        _: u32,
        k: &'static str,
        v: &T,
    ) -> Result<Value, E> {
        Ok(Value::Map(vec![(Value::String(k.into()), capture(v)?)]))
    }
    fn serialize_seq(self, _: Option<usize>) -> Result<Items<E>, E> {
        Ok(Items::new(Value::Seq(vec![]), None))
    }
    fn serialize_tuple(self, n: usize) -> Result<Items<E>, E> {
        self.serialize_seq(Some(n))
    }
    fn serialize_tuple_struct(self, _: &'static str, n: usize) -> Result<Items<E>, E> {
        self.serialize_tuple(n)
    }
    fn serialize_tuple_variant(
        self,
        _: &'static str,
        _: u32,
        k: &'static str,
        _: usize,
    ) -> Result<Items<E>, E> {
        Ok(Items::new(Value::Seq(vec![]), Some(k)))
    }
    fn serialize_map(self, _: Option<usize>) -> Result<Items<E>, E> {
        Ok(Items::new(Value::Map(vec![]), None))
    }
    fn serialize_struct(self, _: &'static str, n: usize) -> Result<Items<E>, E> {
        self.serialize_map(Some(n))
    }
    fn serialize_struct_variant(
        self,
        _: &'static str,
        _: u32,
        k: &'static str,
        _: usize,
    ) -> Result<Items<E>, E> {
        Ok(Items::new(Value::Map(vec![]), Some(k)))
    }
}
macro_rules! seq {
    ($tr:ident,$f:ident) => {
        impl<E: Error> $tr for Items<E> {
            type Ok = Value;
            type Error = E;
            fn $f<T: ?Sized + Serialize>(&mut self, v: &T) -> Result<(), E> {
                self.push(v)
            }
            fn end(self) -> Result<Value, E> {
                self.finish()
            }
        }
    };
}
seq!(SerializeSeq, serialize_element);
seq!(SerializeTuple, serialize_element);
seq!(SerializeTupleStruct, serialize_field);
seq!(SerializeTupleVariant, serialize_field);
macro_rules! fields {
    ($tr:ident) => {
        impl<E: Error> $tr for Items<E> {
            type Ok = Value;
            type Error = E;
            fn serialize_field<T: ?Sized + Serialize>(
                &mut self,
                k: &'static str,
                v: &T,
            ) -> Result<(), E> {
                self.entry(Value::String(k.into()), v)
            }
            fn end(self) -> Result<Value, E> {
                self.finish()
            }
        }
    };
}
fields!(SerializeStruct);
fields!(SerializeStructVariant);
impl<E: Error> SerializeMap for Items<E> {
    type Ok = Value;
    type Error = E;
    fn serialize_key<T: ?Sized + Serialize>(&mut self, k: &T) -> Result<(), E> {
        self.key = Some(capture(k)?);
        Ok(())
    }
    fn serialize_value<T: ?Sized + Serialize>(&mut self, v: &T) -> Result<(), E> {
        let k = self
            .key
            .take()
            .ok_or_else(|| E::custom("map value without key"))?;
        self.entry(k, v)
    }
    fn end(self) -> Result<Value, E> {
        self.finish()
    }
}

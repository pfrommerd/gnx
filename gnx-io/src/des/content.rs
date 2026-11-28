//! Replayable input for representations whose variant is not known before reading its payload.
use super::*;
use std::borrow::Cow;

#[doc(hidden)]
#[derive(Clone)]
pub enum Content<'de> {
    Unit,
    Bool(bool),
    I(i128),
    U(u128),
    F(f64),
    Str(Cow<'de, str>),
    Bytes(Cow<'de, [u8]>),
    Seq(Vec<Self>),
    Map(Vec<(Self, Self)>),
    Some(Box<Self>),
    Array(Array),
}

impl<'de> Deserialize<'de> for Content<'de> {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        d.deserialize_any(Buffer)
    }
}
struct Buffer;
impl Expecting for Buffer {
    fn expected(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str("a value")
    }
}
impl<'de> DataVisitor<'de> for Buffer {
    type Value = Content<'de>;
    fn visit_unit<E: Error>(self) -> Result<Self::Value, E> {
        Ok(Content::Unit)
    }
    fn visit_none<E: Error>(self) -> Result<Self::Value, E> {
        Ok(Content::Unit)
    }
    fn visit_some<D: Deserializer<'de>>(self, d: D) -> Result<Self::Value, D::Error> {
        Ok(Content::Some(Box::new(Content::deserialize(d)?)))
    }
    fn visit_newtype_struct<D: Deserializer<'de>>(self, d: D) -> Result<Self::Value, D::Error> {
        Content::deserialize(d)
    }
    fn visit_bool<E: Error>(self, v: bool) -> Result<Self::Value, E> {
        Ok(Content::Bool(v))
    }
    fn visit_i64<E: Error>(self, v: i64) -> Result<Self::Value, E> {
        self.visit_i128(v.into())
    }
    fn visit_i128<E: Error>(self, v: i128) -> Result<Self::Value, E> {
        Ok(Content::I(v))
    }
    fn visit_u64<E: Error>(self, v: u64) -> Result<Self::Value, E> {
        self.visit_u128(v.into())
    }
    fn visit_u128<E: Error>(self, v: u128) -> Result<Self::Value, E> {
        Ok(Content::U(v))
    }
    fn visit_f64<E: Error>(self, v: f64) -> Result<Self::Value, E> {
        Ok(Content::F(v))
    }
    fn visit_borrowed_str<E: Error>(self, v: &'de str) -> Result<Self::Value, E> {
        Ok(Content::Str(Cow::Borrowed(v)))
    }
    fn visit_str<E: Error>(self, v: &str) -> Result<Self::Value, E> {
        self.visit_string(v.into())
    }
    fn visit_string<E: Error>(self, v: String) -> Result<Self::Value, E> {
        Ok(Content::Str(Cow::Owned(v)))
    }
    fn visit_borrowed_bytes<E: Error>(self, v: &'de [u8]) -> Result<Self::Value, E> {
        Ok(Content::Bytes(Cow::Borrowed(v)))
    }
    fn visit_bytes<E: Error>(self, v: &[u8]) -> Result<Self::Value, E> {
        Ok(Content::Bytes(Cow::Owned(v.into())))
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut a: A) -> Result<Self::Value, A::Error> {
        let mut items = Vec::new();
        while let Some(x) = a.next_element()? {
            items.push(x);
        }
        Ok(Content::Seq(items))
    }
    fn visit_map<A: MapAccess<'de>>(self, mut a: A) -> Result<Self::Value, A::Error> {
        let mut items = Vec::new();
        while let Some(x) = a.next_entry()? {
            items.push(x);
        }
        Ok(Content::Map(items))
    }
    fn visit_array<E: Error>(self, v: Array) -> Result<Self::Value, E> {
        Ok(Content::Array(v))
    }
}

impl<'de> Content<'de> {
    pub fn replay<E: Error>(&self) -> Replay<'_, 'de, E> {
        Replay(self, PhantomData)
    }
    pub fn named(&self, key: &str) -> bool {
        matches!(self,Self::Str(s) if s==key)
    }
    pub fn take_field<E: Error>(&mut self, key: &str) -> Result<Option<Self>, E> {
        let Self::Map(items) = self else {
            return Err(E::custom("expected a tagged map"));
        };
        let mut it = items.iter().enumerate().filter(|(_, (k, _))| k.named(key));
        let first = it.next().map(|(i, _)| i);
        if it.next().is_some() {
            return Err(E::custom(format!("duplicate field: {key}")));
        }
        Ok(first.map(|i| items.remove(i).1))
    }
}
#[doc(hidden)]
pub struct Replay<'a, 'de, E>(&'a Content<'de>, PhantomData<E>);
macro_rules! forward { ($($method:ident $(($($p:ident:$t:ty),*))?),* $(,)?)=>{$(fn $method<V:DataVisitor<'de>>(self,$($($p:$t,)*)?v:V)->Result<V::Value,E>{self.deserialize_any(v)})*}; }
impl<'de, E: Error> Deserializer<'de> for Replay<'_, 'de, E> {
    type Error = E;
    fn deserialize_any<V: DataVisitor<'de>>(self, v: V) -> Result<V::Value, E> {
        use Content::*;
        match self.0 {
            Unit => v.visit_unit(),
            Bool(x) => v.visit_bool(*x),
            I(x) => match (*x).try_into() {
                Ok(i) => v.visit_i64(i),
                Err(_) => v.visit_i128(*x),
            },
            U(x) => match (*x).try_into() {
                Ok(i) => v.visit_u64(i),
                Err(_) => v.visit_u128(*x),
            },
            F(x) => v.visit_f64(*x),
            Str(Cow::Owned(x)) => v.visit_str(x),
            Str(Cow::Borrowed(x)) => v.visit_borrowed_str(x),
            Bytes(Cow::Owned(x)) => v.visit_bytes(x),
            Bytes(Cow::Borrowed(x)) => v.visit_borrowed_bytes(x),
            Some(x) => v.visit_some(x.replay()),
            Array(x) => v.visit_array(x.clone()),
            Seq(x) => v.visit_seq(Items(x.iter(), PhantomData)),
            Map(x) => v.visit_map(Entries(x.iter(), None, PhantomData)),
        }
    }
    forward!(
        deserialize_bool,
        deserialize_i8,
        deserialize_i16,
        deserialize_i32,
        deserialize_i64,
        deserialize_i128,
        deserialize_u8,
        deserialize_u16,
        deserialize_u32,
        deserialize_u64,
        deserialize_u128,
        deserialize_f32,
        deserialize_f64,
        deserialize_char,
        deserialize_str,
        deserialize_string,
        deserialize_bytes,
        deserialize_byte_buf,
        deserialize_unit,
        deserialize_unit_struct(_name: &str),
        deserialize_seq,
        deserialize_tuple(_len: usize),
        deserialize_tuple_struct(_name: &str, _len: usize),
        deserialize_map,
        deserialize_struct(_name: &str, _fields: &[&str]),
        deserialize_identifier,
        deserialize_shared,
        deserialize_array,
        deserialize_hinted
    );
    fn deserialize_newtype_struct<V: DataVisitor<'de>>(self, _: &str, v: V) -> Result<V::Value, E> {
        v.visit_newtype_struct(self)
    }
    fn deserialize_option<V: DataVisitor<'de>>(self, v: V) -> Result<V::Value, E> {
        match self.0 {
            Content::Unit => v.visit_none(),
            Content::Some(x) => v.visit_some(x.replay()),
            _ => v.visit_some(self),
        }
    }
    fn deserialize_ignored_any<V: DataVisitor<'de>>(self, v: V) -> Result<V::Value, E> {
        v.visit_unit()
    }
    fn deserialize_enum<V: DataVisitor<'de>>(
        self,
        _: &str,
        _: &[&str],
        v: V,
    ) -> Result<V::Value, E> {
        let (key, val) = match self.0 {
            Content::Str(_) => (self.0, None),
            Content::Map(m) if m.len() == 1 => {
                let (k, v) = &m[0];
                (k, Some(v))
            }
            _ => return Err(E::custom("expected externally tagged enum")),
        };
        v.visit_enum(Tagged(key, Payload(val, PhantomData)))
    }
}
struct Items<'a, 'de, E>(std::slice::Iter<'a, Content<'de>>, PhantomData<E>);
impl<'de, E: Error> SeqAccess<'de> for Items<'_, 'de, E> {
    type Error = E;
    fn next_element_seed<T: DeserializeSeed<'de>>(&mut self, t: T) -> Result<Option<T::Value>, E> {
        self.0.next().map(|v| t.deserialize(v.replay())).transpose()
    }
}
struct Entries<'a, 'de, E>(
    std::slice::Iter<'a, (Content<'de>, Content<'de>)>,
    Option<&'a Content<'de>>,
    PhantomData<E>,
);
impl<'de, E: Error> MapAccess<'de> for Entries<'_, 'de, E> {
    type Error = E;
    fn next_key_seed<T: DeserializeSeed<'de>>(&mut self, t: T) -> Result<Option<T::Value>, E> {
        let Some((k, v)) = self.0.next() else {
            return Ok(None);
        };
        self.1 = Some(v);
        t.deserialize(k.replay()).map(Some)
    }
    fn next_value_seed<T: DeserializeSeed<'de>>(&mut self, t: T) -> Result<T::Value, E> {
        t.deserialize(
            self.1
                .take()
                .ok_or_else(|| E::custom("expected map value"))?
                .replay(),
        )
    }
}
pub(super) struct Tagged<'a, 'de, E>(pub(super) &'a Content<'de>, pub(super) Payload<'a, 'de, E>);
impl<'a, 'de, E: Error> EnumAccess<'de> for Tagged<'a, 'de, E> {
    type Error = E;
    type Variant = Payload<'a, 'de, E>;
    fn variant_seed<T: DeserializeSeed<'de>>(self, t: T) -> Result<(T::Value, Self::Variant), E> {
        Ok((t.deserialize(self.0.replay())?, self.1))
    }
}
#[doc(hidden)]
pub struct Payload<'a, 'de, E>(pub Option<&'a Content<'de>>, pub PhantomData<E>);
impl<'de, E: Error> VariantAccess<'de> for Payload<'_, 'de, E> {
    type Error = E;
    fn unit_variant(self) -> Result<(), E> {
        match self.0 {
            None | Some(Content::Unit) => Ok(()),
            _ => Err(E::custom("expected unit variant")),
        }
    }
    fn newtype_variant_seed<T: DeserializeSeed<'de>>(self, t: T) -> Result<T::Value, E> {
        t.deserialize(self.0.unwrap_or(&Content::Unit).replay())
    }
    fn tuple_variant<V: DataVisitor<'de>>(self, _: usize, v: V) -> Result<V::Value, E> {
        self.0.unwrap_or(&Content::Unit).replay().deserialize_seq(v)
    }
    fn struct_variant<V: DataVisitor<'de>>(self, _: &[&str], v: V) -> Result<V::Value, E> {
        self.0.unwrap_or(&Content::Unit).replay().deserialize_map(v)
    }
}

use super::{
    content::{Content, Payload, Tagged},
    *,
};
use std::borrow::Cow;

type Entry<'de> = (Content<'de>, Content<'de>);

/// All flattened fields share these slots. Named fields claim entries in place;
/// catch-all maps read the remaining entries without taking them.
#[doc(hidden)]
pub struct FlatInput<'a, 'de, E>(pub &'a mut [Option<Entry<'de>>], pub PhantomData<E>);

struct Fields<'a, 'de, 'f, E> {
    entries: std::slice::IterMut<'a, Option<Entry<'de>>>,
    names: Option<&'f [&'f str]>,
    pending: Option<Cow<'a, Content<'de>>>,
    marker: PhantomData<E>,
}
impl<'de, E: Error> MapAccess<'de> for Fields<'_, 'de, '_, E> {
    type Error = E;
    fn next_key_seed<K: DeserializeSeed<'de>>(&mut self, seed: K) -> Result<Option<K::Value>, E> {
        for slot in self.entries.by_ref() {
            let Some((k, _)) = slot.as_ref() else {
                continue;
            };
            if let Some(names) = self.names {
                if !names.iter().any(|name| k.named(name)) {
                    continue;
                }
                let (key, value) = slot.take().unwrap();
                self.pending = Some(Cow::Owned(value));
                return seed.deserialize(key.replay()).map(Some);
            }
            let (key, value) = slot.as_ref().unwrap();
            self.pending = Some(Cow::Borrowed(value));
            return seed.deserialize(key.replay()).map(Some);
        }
        Ok(None)
    }
    fn next_value_seed<V: DeserializeSeed<'de>>(&mut self, seed: V) -> Result<V::Value, E> {
        let value = self
            .pending
            .take()
            .ok_or_else(|| E::custom("expected flattened map value"))?;
        seed.deserialize(value.replay())
    }
}
macro_rules! reject {
    ($($method:ident $(($($p:ident:$t:ty),*))?),*) => {$(
        fn $method<V:DataVisitor<'de>>(self,$($($p:$t,)*)?_:V)->Result<V::Value,E>{
            Err(E::custom("expected a flattened map or struct"))
        }
    )*};
}
impl<'de, E: Error> Deserializer<'de> for FlatInput<'_, 'de, E> {
    type Error = E;
    reject!(
        deserialize_bool,
        deserialize_i8,
        deserialize_i16,
        deserialize_i32,
        deserialize_i64,
        deserialize_u8,
        deserialize_u16,
        deserialize_u32,
        deserialize_u64,
        deserialize_f32,
        deserialize_f64,
        deserialize_char,
        deserialize_str,
        deserialize_string,
        deserialize_bytes,
        deserialize_byte_buf,
        deserialize_seq,
        deserialize_tuple(_n: usize),
        deserialize_tuple_struct(_name: &str, _n: usize),
        deserialize_identifier,
        deserialize_shared,
        deserialize_hinted,
        deserialize_array,
        deserialize_ignored_any
    );
    fn deserialize_any<V: DataVisitor<'de>>(self, v: V) -> Result<V::Value, E> {
        self.deserialize_map(v)
    }
    fn deserialize_map<V: DataVisitor<'de>>(self, v: V) -> Result<V::Value, E> {
        v.visit_map(Fields {
            entries: self.0.iter_mut(),
            names: None,
            pending: None,
            marker: PhantomData,
        })
    }
    fn deserialize_struct<V: DataVisitor<'de>>(
        self,
        _: &str,
        fields: &[&str],
        v: V,
    ) -> Result<V::Value, E> {
        v.visit_map(Fields {
            entries: self.0.iter_mut(),
            names: Some(fields),
            pending: None,
            marker: PhantomData,
        })
    }
    fn deserialize_enum<V: DataVisitor<'de>>(
        self,
        name: &str,
        names: &[&str],
        v: V,
    ) -> Result<V::Value, E> {
        for slot in self.0 {
            if slot
                .as_ref()
                .is_some_and(|(k, _)| names.iter().any(|n| k.named(n)))
            {
                let (k, val) = slot.take().unwrap();
                return v.visit_enum(Tagged(&k, Payload(Some(&val), PhantomData)));
            }
        }
        Err(E::custom(format!(
            "no variant of {name} found in flattened data"
        )))
    }
    fn deserialize_option<V: DataVisitor<'de>>(self, v: V) -> Result<V::Value, E> {
        v.visit_flattened_option(self)
    }
    fn deserialize_unit<V: DataVisitor<'de>>(self, v: V) -> Result<V::Value, E> {
        v.visit_unit()
    }
    fn deserialize_unit_struct<V: DataVisitor<'de>>(self, _: &str, v: V) -> Result<V::Value, E> {
        v.visit_unit()
    }
    fn deserialize_newtype_struct<V: DataVisitor<'de>>(self, _: &str, v: V) -> Result<V::Value, E> {
        v.visit_newtype_struct(self)
    }
}

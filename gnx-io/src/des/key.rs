use super::*;
use std::borrow::Cow;

/// Decode either a textual field/variant identifier or a format's positional index.
#[doc(hidden)]
pub struct KeySeed(pub &'static [&'static str]);
impl Expecting for KeySeed {
    fn expected(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str("a field or variant identifier")
    }
}
impl<'de> DeserializeSeed<'de> for KeySeed {
    type Value = Cow<'de, str>;
    fn deserialize<D: Deserializer<'de>>(self, d: D) -> Result<Self::Value, D::Error> {
        d.deserialize_identifier(self)
    }
}
impl<'de> DataVisitor<'de> for KeySeed {
    type Value = Cow<'de, str>;
    fn visit_u64<E: Error>(self, i: u64) -> Result<Self::Value, E> {
        usize::try_from(i)
            .ok()
            .and_then(|i| self.0.get(i))
            .copied()
            .map(Cow::Borrowed)
            .ok_or_else(|| E::custom(format!("invalid identifier index: {i}")))
    }
    fn visit_borrowed_str<E: Error>(self, v: &'de str) -> Result<Self::Value, E> {
        Ok(Cow::Borrowed(v))
    }
    fn visit_string<E: Error>(self, v: String) -> Result<Self::Value, E> {
        Ok(Cow::Owned(v))
    }
    fn visit_str<E: Error>(self, v: &str) -> Result<Self::Value, E> {
        Ok(Cow::Owned(v.into()))
    }
    fn visit_borrowed_bytes<E: Error>(self, v: &'de [u8]) -> Result<Self::Value, E> {
        std::str::from_utf8(v).map(Cow::Borrowed).map_err(E::custom)
    }
    fn visit_bytes<E: Error>(self, v: &[u8]) -> Result<Self::Value, E> {
        std::str::from_utf8(v)
            .map(|s| Cow::Owned(s.into()))
            .map_err(E::custom)
    }
}

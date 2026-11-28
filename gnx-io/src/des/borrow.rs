use super::*;
use std::borrow::Cow;

struct Borrow;
impl Expecting for Borrow {
    fn expected(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str("string or bytes")
    }
}
struct Text;
impl Expecting for Text {
    fn expected(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str("a string")
    }
}
impl<'de> DataVisitor<'de> for Text {
    type Value = Cow<'de, str>;
    fn visit_borrowed_str<E: Error>(self, s: &'de str) -> Result<Self::Value, E> {
        Ok(Cow::Borrowed(s))
    }
    fn visit_str<E: Error>(self, s: &str) -> Result<Self::Value, E> {
        Ok(Cow::Owned(s.into()))
    }
    fn visit_string<E: Error>(self, s: String) -> Result<Self::Value, E> {
        Ok(Cow::Owned(s))
    }
}
impl<'de> DataVisitor<'de> for Borrow {
    type Value = Cow<'de, [u8]>;
    fn visit_borrowed_bytes<E: Error>(self, s: &'de [u8]) -> Result<Self::Value, E> {
        Ok(Cow::Borrowed(s))
    }
    fn visit_bytes<E: Error>(self, s: &[u8]) -> Result<Self::Value, E> {
        Ok(Cow::Owned(s.into()))
    }
    fn visit_byte_buf<E: Error>(self, s: Vec<u8>) -> Result<Self::Value, E> {
        Ok(Cow::Owned(s))
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut a: A) -> Result<Self::Value, A::Error> {
        let mut v = Vec::new();
        while let Some(b) = a.next_element()? {
            v.push(b);
        }
        Ok(Cow::Owned(v))
    }
}
pub fn borrow_cow_str<'de, D: Deserializer<'de>>(d: D) -> Result<Cow<'de, str>, D::Error> {
    d.deserialize_str(Text)
}
pub fn borrow_cow_bytes<'de, D: Deserializer<'de>>(d: D) -> Result<Cow<'de, [u8]>, D::Error> {
    d.deserialize_any(Borrow)
}

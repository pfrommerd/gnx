use super::{Deserialize, Deserializer, Expecting, DataVisitor, SeqAccess, MapAccess};

use gnx::util::Error;

use std::marker::PhantomData;
use std::hash::Hash;
use std::collections::HashMap;


impl<'de, T: Deserialize<'de>> Deserialize<'de> for Vec<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        // let mut seq = deserializer.deserialize_seq(visitor);
        struct VecVisitor<T>(PhantomData<T>);
        impl<'de, T> Expecting for VecVisitor<T> {
            fn expected(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(fmt, "a sequence of {}s", std::any::type_name::<T>())
            }
        }
        impl<'de, T: Deserialize<'de>> DataVisitor<'de> for VecVisitor<T> {
            type Value = Vec<T>;
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let mut v = Vec::new();
                while let Some(item) = seq.next_element()? {
                    v.push(item);
                }
                Ok(v)
            }
        }
        deserializer.deserialize_seq(VecVisitor(PhantomData))
    }
}

impl<'de, K: Deserialize<'de> + Eq + Hash, V: Deserialize<'de>> Deserialize<'de> for HashMap<K, V> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct HashMapVisitor<K, V>(PhantomData<(K, V)>);
        impl<'de, K, V> Expecting for HashMapVisitor<K, V> {
            fn expected(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(fmt, "a map of {}s and {}s", std::any::type_name::<K>(), std::any::type_name::<V>())
            }
        }
        impl<'de, K: Deserialize<'de> + Eq + Hash, V: Deserialize<'de>> DataVisitor<'de> for HashMapVisitor<K, V> {
            type Value = HashMap<K, V>;
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                let mut h = HashMap::new();
                while let Some((k, v)) = map.next_entry()? {
                    h.insert(k, v);
                }
                Ok(h)
            }
        }
        deserializer.deserialize_map(HashMapVisitor(PhantomData))
    }
}


impl<'de> Deserialize<'de> for String {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct StringVisitor;
        impl<'de> Expecting for StringVisitor {
            fn expected(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(fmt, "a string")
            }
        }
        impl<'de> DataVisitor<'de> for StringVisitor {
            type Value = String;
            fn visit_str<E: Error>(self, value: &str) -> Result<Self::Value, E> {
                Ok(value.to_string())
            }
        }
        deserializer.deserialize_str(StringVisitor)
    }
}
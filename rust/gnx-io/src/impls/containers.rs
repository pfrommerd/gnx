use crate::*;

use std::marker::PhantomData;
use std::hash::Hash;
use std::collections::HashMap;

impl<T: GraphSerialize> GraphSerialize for Vec<T> {
    fn serialize<S: GraphSerializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(self.iter())
    }
}

impl<'de, T: GraphDeserialize<'de>> GraphDeserialize<'de> for Vec<T> {
    fn deserialize<D: GraphDeserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        // let mut seq = deserializer.deserialize_seq(visitor);
        struct VecVisitor<T>(PhantomData<T>);
        impl<'de, T> Expecting for VecVisitor<T> {
            fn expected(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(fmt, "a sequence of {}s", std::any::type_name::<T>())
            }
        }
        impl<'de, T: GraphDeserialize<'de>> DesVisitor<'de> for VecVisitor<T> {
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


impl<K: GraphSerialize, V: GraphSerialize> GraphSerialize for HashMap<K, V> {
    fn serialize<S: GraphSerializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_map(self.iter())
    }
}

impl<'de, K: GraphDeserialize<'de> + Eq + Hash, V: GraphDeserialize<'de>> GraphDeserialize<'de> for HashMap<K, V> {
    fn deserialize<D: GraphDeserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct HashMapVisitor<K, V>(PhantomData<(K, V)>);
        impl<'de, K, V> Expecting for HashMapVisitor<K, V> {
            fn expected(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(fmt, "a map of {}s and {}s", std::any::type_name::<K>(), std::any::type_name::<V>())
            }
        }
        impl<'de, K: GraphDeserialize<'de> + Eq + Hash, V: GraphDeserialize<'de>> DesVisitor<'de> for HashMapVisitor<K, V> {
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
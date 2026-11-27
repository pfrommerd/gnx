use super::{DataVisitor, Deserialize, Deserializer, Expecting, MapAccess, SeqAccess};

use gnx_graph::Error;

use std::collections::HashMap;
use std::hash::Hash;
use std::marker::PhantomData;

macro_rules! scalar {
    ($t:ty, $method:ident, $($visit:item),+ $(,)?) => {
        impl<'de> Deserialize<'de> for $t {
            fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                struct ScalarVisitor;
                impl Expecting for ScalarVisitor {
                    fn expected(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
                        fmt.write_str(stringify!($t))
                    }
                }
                impl DataVisitor<'_> for ScalarVisitor {
                    type Value = $t;
                    $($visit)*
                }
                deserializer.$method(ScalarVisitor)
            }
        }
    };
}

macro_rules! integer {
    ($t:ty, $method:ident) => {
        scalar!(
            $t,
            $method,
            fn visit_i64<E: Error>(self, n: i64) -> Result<$t, E> {
                n.try_into().map_err(E::custom)
            },
            fn visit_u64<E: Error>(self, n: u64) -> Result<$t, E> {
                n.try_into().map_err(E::custom)
            }
        );
    };
}
macro_rules! float {
    ($t:ty, $method:ident) => {
        scalar!(
            $t,
            $method,
            fn visit_f64<E: Error>(self, n: f64) -> Result<$t, E> {
                Ok(n as $t)
            },
            fn visit_i64<E: Error>(self, n: i64) -> Result<$t, E> {
                Ok(n as $t)
            },
            fn visit_u64<E: Error>(self, n: u64) -> Result<$t, E> {
                Ok(n as $t)
            }
        );
    };
}

integer!(usize, deserialize_u64);
integer!(u8, deserialize_u8);
integer!(u16, deserialize_u16);
integer!(u32, deserialize_u32);
integer!(u64, deserialize_u64);
integer!(i8, deserialize_i8);
integer!(i16, deserialize_i16);
integer!(i32, deserialize_i32);
integer!(i64, deserialize_i64);

scalar!(
    bool,
    deserialize_bool,
    fn visit_bool<E: Error>(self, value: bool) -> Result<bool, E> {
        Ok(value)
    }
);

float!(f32, deserialize_f32);
float!(f64, deserialize_f64);

impl<'de> Deserialize<'de> for gnx_base::DType {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        String::deserialize(deserializer)?
            .parse()
            .map_err(D::Error::custom)
    }
}

impl<'de, T: Deserialize<'de>> Deserialize<'de> for Option<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct OptionVisitor<T>(PhantomData<T>);
        impl<T> Expecting for OptionVisitor<T> {
            fn expected(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
                fmt.write_str(std::any::type_name::<Option<T>>())
            }
        }
        impl<'de, T: Deserialize<'de>> DataVisitor<'de> for OptionVisitor<T> {
            type Value = Option<T>;
            fn visit_none<E: Error>(self) -> Result<Self::Value, E> {
                Ok(None)
            }
            fn visit_some<D: Deserializer<'de>>(self, d: D) -> Result<Self::Value, D::Error> {
                T::deserialize(d).map(Some)
            }
        }
        deserializer.deserialize_option(OptionVisitor(PhantomData))
    }
}

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
                write!(
                    fmt,
                    "a map of {}s and {}s",
                    std::any::type_name::<K>(),
                    std::any::type_name::<V>()
                )
            }
        }
        impl<'de, K: Deserialize<'de> + Eq + Hash, V: Deserialize<'de>> DataVisitor<'de>
            for HashMapVisitor<K, V>
        {
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

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

scalar!(
    (),
    deserialize_unit,
    fn visit_unit<E: Error>(self) -> Result<(), E> {
        Ok(())
    }
);

impl<'de: 'a, 'a> Deserialize<'de> for &'a str {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct Borrowed;
        impl Expecting for Borrowed {
            fn expected(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a borrowed string")
            }
        }
        impl<'de> DataVisitor<'de> for Borrowed {
            type Value = &'de str;
            fn visit_borrowed_str<E: Error>(self, s: &'de str) -> Result<Self::Value, E> {
                Ok(s)
            }
        }
        d.deserialize_str(Borrowed)
    }
}

impl<'de> Deserialize<'de> for gnx_base::DType {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        String::deserialize(deserializer)?
            .parse()
            .map_err(D::Error::custom)
    }
}

impl<'de, T: Deserialize<'de>> Deserialize<'de> for Option<T> {
    fn missing_field<E: Error>(_: &str) -> Result<Self, E> {
        Ok(None)
    }

    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct OptionVisitor<T>(PhantomData<T>);
        impl<T> Expecting for OptionVisitor<T> {
            fn expected(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
                fmt.write_str(std::any::type_name::<Option<T>>())
            }
        }
        impl<'de, T: Deserialize<'de>> DataVisitor<'de> for OptionVisitor<T> {
            type Value = Option<T>;
            fn visit_flattened_option<D: Deserializer<'de>>(
                self,
                d: D,
            ) -> Result<Self::Value, D::Error> {
                Ok(T::deserialize(d).ok())
            }
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

impl<'de, T: Deserialize<'de>> Deserialize<'de> for Box<T> {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        T::deserialize(d).map(Box::new)
    }
}
impl<'de, T: ?Sized> Deserialize<'de> for PhantomData<T> {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        <()>::deserialize(d).map(|_| PhantomData)
    }
}
impl<'de, T: ?Sized + ToOwned> Deserialize<'de> for std::borrow::Cow<'_, T>
where
    T::Owned: Deserialize<'de>,
{
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        T::Owned::deserialize(d).map(std::borrow::Cow::Owned)
    }
}
impl<'de, T: Deserialize<'de>, const N: usize> Deserialize<'de> for [T; N] {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct ArrayVisitor<T, const N: usize>(PhantomData<T>);
        impl<T, const N: usize> Expecting for ArrayVisitor<T, N> {
            fn expected(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(f, "an array of length {N}")
            }
        }
        impl<'de, T: Deserialize<'de>, const N: usize> DataVisitor<'de> for ArrayVisitor<T, N> {
            type Value = [T; N];
            fn visit_seq<A: SeqAccess<'de>>(self, mut a: A) -> Result<Self::Value, A::Error> {
                let mut v = Vec::new();
                for _ in 0..N {
                    v.push(
                        a.next_element()?
                            .ok_or_else(|| Error::custom("array too short"))?,
                    );
                }
                if a.next_element::<super::IgnoredAny>()?.is_some() {
                    return Err(Error::custom("array too long"));
                }
                Ok(v.try_into().unwrap_or_else(|_| unreachable!()))
            }
        }
        d.deserialize_tuple(N, ArrayVisitor(PhantomData))
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

// Generate all prefix arities, including the one-element tuple (distinct from a newtype).
macro_rules! tuples {
    (@impl $(($T:ident, $i:tt)),+) => {
        impl<'de, $($T: Deserialize<'de>),+> Deserialize<'de> for ($($T,)+) {
            fn deserialize<__D: Deserializer<'de>>(d: __D) -> Result<Self, __D::Error> {
                struct TupleVisitor<T>(PhantomData<T>);
                impl<T> Expecting for TupleVisitor<T> {
                    fn expected(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                        f.write_str("a tuple")
                    }
                }
                impl<'de, $($T: Deserialize<'de>),+> DataVisitor<'de> for TupleVisitor<($($T,)+)> {
                    type Value = ($($T,)+);
                    fn visit_seq<__A: SeqAccess<'de>>(
                        self,
                        mut seq: __A,
                    ) -> Result<Self::Value, __A::Error> {
                        let value = ($(
                            seq.next_element::<$T>()?.ok_or_else(|| {
                                Error::custom(concat!("missing tuple element ", stringify!($i)))
                            })?,
                        )+);
                        if seq.next_element::<super::IgnoredAny>()?.is_some() {
                            return Err(Error::custom("too many tuple elements"));
                        }
                        Ok(value)
                    }
                }
                d.deserialize_tuple([$(stringify!($T)),+].len(), TupleVisitor(PhantomData))
            }
        }
    };
    (@prefix [$($done:tt),*] $next:tt $(, $rest:tt)*) => {
        tuples!(@impl $($done,)* $next);
        tuples!(@prefix [$($done,)* $next] $($rest),*);
    };
    (@prefix [$($done:tt),*]) => {};
}
tuples!(@prefix []
    (A, 0), (B, 1), (C, 2), (D, 3), (E, 4), (F, 5), (G, 6), (H, 7),
    (I, 8), (J, 9), (K, 10), (L, 11), (M, 12), (N, 13), (O, 14), (P, 15)
);

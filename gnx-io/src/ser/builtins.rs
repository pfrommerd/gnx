use paste;
use std::collections::HashMap;

use super::{Serialize, Serializer};

macro_rules! impl_graph_serialize {
    ($($T:ty)*) => { paste::paste! {
        $(
            impl Serialize for $T {
                fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                    serializer.[< serialize_ $T:lower >](*self)
                }
            }
        )*
    } };
}

impl_graph_serialize!(
    bool i8 i16 i32 i64 u8 u16 u32 u64 f32 f64
);

impl Serialize for usize {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_u64(*self as u64)
    }
}

impl Serialize for gnx_base::DType {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str((*self).into())
    }
}

impl Serialize for str {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self)
    }
}
impl Serialize for () {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_unit()
    }
}
impl Serialize for String {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self)
    }
}

impl<T: ?Sized + Serialize> Serialize for Box<T> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        (**self).serialize(s)
    }
}
impl<T: ?Sized> Serialize for std::marker::PhantomData<T> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_unit_struct("PhantomData")
    }
}
impl<T: ?Sized + ToOwned + Serialize> Serialize for std::borrow::Cow<'_, T> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        self.as_ref().serialize(s)
    }
}
impl<T: Serialize> Serialize for [T] {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.collect_seq(self)
    }
}
impl<T: Serialize, const N: usize> Serialize for [T; N] {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use super::SerializeTuple;
        let mut tuple = s.serialize_tuple(N)?;
        for item in self {
            tuple.serialize_element(item)?;
        }
        tuple.end()
    }
}
impl<T: Serialize> Serialize for Vec<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(self.iter())
    }
}

impl<K: Serialize, V: Serialize> Serialize for HashMap<K, V> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_map(self.iter())
    }
}

impl<T: Serialize> Serialize for Option<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Some(value) => serializer.serialize_some(value),
            None => serializer.serialize_none(),
        }
    }
}

// Generate all prefix arities, including the one-element tuple (distinct from a newtype).
macro_rules! tuples {
    (@impl $(($T:ident, $i:tt)),+) => {
        impl<$($T: Serialize),+> Serialize for ($($T,)+) {
            fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
                use super::SerializeTuple;
                let mut tuple = s.serialize_tuple([$(stringify!($T)),+].len())?;
                $(tuple.serialize_element(&self.$i)?;)+
                tuple.end()
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

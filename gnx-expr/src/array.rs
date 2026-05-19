use std::fmt::{Debug, Display};

use ordered_float::OrderedFloat;

use crate::expr::trace::{ConcreteValue, Traceable, Tracer, TracerCell, ValueInfo};

#[derive(Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Dim {
    Fixed(usize),
    // unknown size, can be converted
    // to a fixed size by truncating if too big or padding if too small.
    Dynamic,
    // Dependent on another dimension
    // to resolve the dynamic size.
    // Contains the index of the dependent dimension.
    // The other dimension can potentially also be jagged.
    Jagged(usize),
}

impl Display for Dim {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Dim::Fixed(n) => write!(f, "{}", n),
            Dim::Dynamic => write!(f, "?"),
            Dim::Jagged(i) => write!(f, "jagged({})", i),
        }
    }
}

#[derive(Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Shape(Vec<Dim>);

impl Shape {
    pub fn fixed(dims: impl IntoIterator<Item = usize>) -> Self {
        Shape(dims.into_iter().map(Dim::Fixed).collect())
    }
}

impl FromIterator<Dim> for Shape {
    fn from_iter<T: IntoIterator<Item = Dim>>(iter: T) -> Self {
        Shape(iter.into_iter().collect())
    }
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "(")?;
        for (i, dim) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", dim)?;
        }
        write!(f, ")")
    }
}

impl std::fmt::Debug for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}

#[rustfmt::skip]
#[derive(Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum DType {
    Bool,
    U8, U16, U32, U64,
    I8, I16, I32, I64,
    F16, F32, F64,
    C64, C128,
}

#[derive(Clone, Hash, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Item {
    Bool(bool),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    F16(OrderedFloat<f32>),
    F32(OrderedFloat<f32>),
    F64(OrderedFloat<f64>),
    C64(OrderedFloat<f32>, OrderedFloat<f32>),
    C128(OrderedFloat<f64>, OrderedFloat<f64>),
}

impl std::fmt::Display for Item {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Item::Bool(b) => write!(f, "{}", b),
            Item::U8(u) => write!(f, "{}", u),
            Item::U16(u) => write!(f, "{}", u),
            Item::U32(u) => write!(f, "{}", u),
            Item::U64(u) => write!(f, "{}", u),
            Item::I8(i) => write!(f, "{}", i),
            Item::I16(i) => write!(f, "{}", i),
            Item::I32(i) => write!(f, "{}", i),
            Item::I64(i) => write!(f, "{}", i),
            Item::F16(v) => write!(f, "{}", v),
            Item::F32(v) => write!(f, "{}", v),
            Item::F64(v) => write!(f, "{}", v),
            Item::C64(r, i) => write!(f, "{} + {}i", r, i),
            Item::C128(r, i) => write!(f, "{} + {}i", r, i),
        }
    }
}

use std::borrow::Cow;

#[derive(Clone, Hash, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Data<'r> {
    Bool(Cow<'r, [bool]>),
    U8(Cow<'r, [u8]>),
    U16(Cow<'r, [u16]>),
    U32(Cow<'r, [u32]>),
    U64(Cow<'r, [u64]>),
    I8(Cow<'r, [i8]>),
    I16(Cow<'r, [i16]>),
    I32(Cow<'r, [i32]>),
    I64(Cow<'r, [i64]>),
    F16(Cow<'r, [OrderedFloat<f32>]>),
    F32(Cow<'r, [OrderedFloat<f32>]>),
    F64(Cow<'r, [OrderedFloat<f64>]>),
    C64(Cow<'r, [OrderedFloat<f32>]>, Cow<'r, [OrderedFloat<f32>]>),
    C128(Cow<'r, [OrderedFloat<f64>]>, Cow<'r, [OrderedFloat<f64>]>),
}

fn fmt_display_slice<T: Display>(f: &mut std::fmt::Formatter<'_>, slice: &[T]) -> std::fmt::Result {
    write!(f, "[")?;
    let mut first = true;
    for x in slice {
        if !first {
            write!(f, ", ")?;
        }
        first = false;
        write!(f, "{}", x)?;
    }
    write!(f, "]")
}

impl<'r> std::fmt::Display for Data<'r> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Data::Bool(v) => fmt_display_slice(f, v.as_ref()),
            Data::U8(v) => fmt_display_slice(f, v.as_ref()),
            Data::U16(v) => fmt_display_slice(f, v.as_ref()),
            Data::U32(v) => fmt_display_slice(f, v.as_ref()),
            Data::U64(v) => fmt_display_slice(f, v.as_ref()),
            Data::I8(v) => fmt_display_slice(f, v.as_ref()),
            Data::I16(v) => fmt_display_slice(f, v.as_ref()),
            Data::I32(v) => fmt_display_slice(f, v.as_ref()),
            Data::I64(v) => fmt_display_slice(f, v.as_ref()),
            Data::F16(v) => fmt_display_slice(f, v.as_ref()),
            Data::F32(v) => fmt_display_slice(f, v.as_ref()),
            Data::F64(v) => fmt_display_slice(f, v.as_ref()),
            Data::C64(re, im) => {
                write!(f, "[")?;
                let mut first = true;
                for (r, i) in re.iter().zip(im.iter()) {
                    if !first {
                        write!(f, ", ")?;
                    }
                    first = false;
                    write!(f, "{} + {}i", r, i)?;
                }
                write!(f, "]")
            }
            Data::C128(re, im) => {
                write!(f, "[")?;
                let mut first = true;
                for (r, i) in re.iter().zip(im.iter()) {
                    if !first {
                        write!(f, ", ")?;
                    }
                    first = false;
                    write!(f, "{} + {}i", r, i)?;
                }
                write!(f, "]")
            }
        }
    }
}

impl From<&'static str> for DType {
    fn from(s: &'static str) -> Self {
        match s {
            "bool" => DType::Bool,
            "uint8" => DType::U8,
            "uint16" => DType::U16,
            "uint32" => DType::U32,
            "uint64" => DType::U64,
            "int8" => DType::I8,
            "int16" => DType::I16,
            "int32" => DType::I32,
            "int64" => DType::I64,
            "float16" => DType::F16,
            "float32" => DType::F32,
            "float64" => DType::F64,
            "complex64" => DType::C64,
            "complex128" => DType::C128,
            _ => panic!("Unsupported data type: {}", s),
        }
    }
}

impl Into<&'static str> for DType {
    fn into(self) -> &'static str {
        match self {
            DType::Bool => "bool",
            DType::U8 => "uint8",
            DType::U16 => "uint16",
            DType::U32 => "uint32",
            DType::U64 => "uint64",
            DType::I8 => "int8",
            DType::I16 => "int16",
            DType::I32 => "int32",
            DType::I64 => "int64",
            DType::F16 => "float16",
            DType::F32 => "float32",
            DType::F64 => "float64",
            DType::C64 => "complex64",
            DType::C128 => "complex128",
        }
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let dtype_str: &'static str = (*self).into();
        write!(f, "{}", dtype_str)
    }
}

pub trait DataImpl: Send + Sync + Debug + Display {
    fn info(&self) -> &ArrayInfo;
}
pub type DataHandle = Box<dyn DataImpl>;

impl ConcreteValue for DataHandle {
    fn to_info(&self) -> ValueInfo {
        ValueInfo::Array(self.info().clone())
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct ArrayInfo {
    shape: Shape,
    dtype: DType,
}

impl ArrayInfo {
    pub fn new(shape: Shape, dtype: DType) -> Self {
        ArrayInfo { shape, dtype }
    }
}

impl Display for ArrayInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}", self.shape, self.dtype)
    }
}

pub struct Array(Tracer<Array>);

impl From<Array> for Tracer<Array> {
    fn from(value: Array) -> Self {
        value.0
    }
}
impl From<Tracer<Array>> for Array {
    fn from(value: Tracer<Array>) -> Self {
        Array(value)
    }
}

impl Traceable for Array {
    type Concrete = DataHandle;
    type Info = ArrayInfo;
}


impl Array {
    pub fn dtype(&self) -> DType {
        self.0.info().dtype
    }
    pub fn shape(&self) -> &Shape {
        &self.0.info().shape
    }
}

/// Mutable array traced via [`TracerCell`] (JAX-style ref).
pub struct ArrayRef(TracerCell<Array>);

impl ArrayRef {
    pub fn new(base: impl Into<Tracer<Array>>) -> Self {
        ArrayRef(TracerCell::new(base.into()))
    }

    pub fn cell(&self) -> &TracerCell<Array> {
        &self.0
    }

    pub fn get(&self) -> Array {
        Array(self.0.get())
    }

    pub fn set(&self, value: impl Into<Tracer<Array>>) {
        self.0.set(value.into());
    }

    pub fn dtype(&self) -> DType {
        self.0.get().info().dtype
    }

    pub fn shape(&self) -> Shape {
        self.0.get().info().shape.clone()
    }
}
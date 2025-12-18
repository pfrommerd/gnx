use std::fmt::{Debug, Display};

use ordered_float::OrderedFloat;

use crate::trace::{Tracer, Traceable, ConcreteValue, ValueInfo};

pub enum Dim {
    Fixed(usize),
    // unknown size, can be converted
    // to a fixed size by truncating if too big or padding if too small.
    Dynamic,
    // Dependent on another dimension
    // to resolve the dynamic size.
    // Contains the index of the dependent dimension.
    // The other dimension can potentially also be jagged.
    Jagged(usize)
}

#[derive(Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Shape(Vec<usize>);

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let shape_str = self
            .0
            .iter()
            .map(|dim| dim.to_string())
            .collect::<Vec<String>>()
            .join(", ");
        write!(f, "({})", shape_str)
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

pub trait MutDataImpl: Send + Sync + Debug + Display {
    fn info(&self) -> &ArrayInfo;
}
pub type MutDataHandle = Box<dyn MutDataImpl>;

impl ConcreteValue for MutDataHandle {
    fn to_info(&self) -> ValueInfo {
        ValueInfo::ArrayRef(ArrayRefInfo(self.info().clone()))
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct ArrayInfo {
    shape: Shape,
    dtype: DType,
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

// Used for the tracer
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct ArrayRefInfo(ArrayInfo);

impl Display for ArrayRefInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

pub struct ArrayRef(Tracer<ArrayRef>);

impl From<ArrayRef> for Tracer<ArrayRef> {
    fn from(value: ArrayRef) -> Self {
        value.0
    }
}
impl From<Tracer<ArrayRef>> for ArrayRef {
    fn from(value: Tracer<ArrayRef>) -> Self {
        ArrayRef(value)
    }
}

impl Traceable for ArrayRef {
    type Concrete = MutDataHandle;
    type Info = ArrayRefInfo;
}

impl ArrayRef {
    pub fn dtype(&self) -> DType {
        self.0.info().0.dtype
    }
    pub fn shape(&self) -> &Shape {
        &self.0.info().0.shape
    }
}
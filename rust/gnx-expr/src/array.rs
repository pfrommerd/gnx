use std::sync::Arc;

use crate::{BorrowFrom, Traceable, Tracer, Value, ValueInfo};

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
#[derive(Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum DType {
    Bool,
    U8, U16, U32, U64,
    I8, I16, I32, I64,
    F16, F32, F64,
    C64, C128,
}

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
    F16(f32),
    F32(f32),
    F64(f64),
    C64(f32, f32),
    C128(f64, f64),
}

use std::borrow::Cow;
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
    F16(Cow<'r, [f32]>),
    F32(Cow<'r, [f32]>),
    F64(Cow<'r, [f64]>),
    C64(Cow<'r, [f32]>, Cow<'r, [f32]>),
    C128(Cow<'r, [f64]>, Cow<'r, [f64]>),
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

pub trait DataImpl {}
pub type DataHandle = Arc<dyn DataImpl>;

pub struct Sharding {}

pub struct ArrayInfo {
    shape: Shape,
    dtype: DType,
    sharding: Option<Sharding>,
}

pub struct Array(Tracer<Array>);

impl Array {
    pub fn sharding(&self) -> Option<&Sharding> {
        self.0.info().sharding.as_ref()
    }
    pub fn dtype(&self) -> DType {
        self.0.info().dtype
    }
    pub fn shape(&self) -> &Shape {
        &self.0.info().shape
    }
}

impl BorrowFrom<ValueInfo> for ArrayInfo {
    fn borrow_from(value: &ValueInfo) -> &Self {
        todo!()
    }
}

impl Traceable for Array {
    type Concrete = Value;
    type Info = ArrayInfo;
}
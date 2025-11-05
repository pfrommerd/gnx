use std::sync::Arc;

use super::expr::{ArrayRefTracer, ArrayTracer};

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

pub struct ArrayType {
    shape: Shape,
    dtype: DType,
    sharding: Option<Sharding>,
}

pub struct Array(ArrayTracer);

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

// Wraps ArrayInfo so that Array/ArrayRef have different info types

#[allow(unused)]
pub struct ArrayRefType(ArrayType);

#[allow(unused)]
pub struct ArrayRef(ArrayRefTracer);

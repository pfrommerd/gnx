use crate::array::{ArrayRefType, ArrayType, DataHandle};
use std::borrow::Borrow;

// Generic type information
// and value storage for all supported
// expression types.
// Note that Array and ArrayRef are distinct types.

pub enum TypeInfo {
    Array(ArrayType),
    ArrayRef(ArrayRefType),
}

pub enum TypeValue {
    Array(DataHandle),
}

// Conversion types
// Note that these panic on failure
impl From<DataHandle> for TypeValue {
    fn from(data: DataHandle) -> TypeValue {
        TypeValue::Array(data)
    }
}
impl From<TypeValue> for DataHandle {
    fn from(value: TypeValue) -> DataHandle {
        match value {
            TypeValue::Array(handle) => handle,
            // _ => panic!("TypeValue is not an Array"),
        }
    }
}
impl Borrow<DataHandle> for TypeValue {
    fn borrow(&self) -> &DataHandle {
        match self {
            TypeValue::Array(handle) => handle,
            // _ => panic!("TypeValue is not an Array"),
        }
    }
}

impl From<ArrayType> for TypeInfo {
    fn from(info: ArrayType) -> TypeInfo {
        TypeInfo::Array(info)
    }
}
impl From<TypeInfo> for ArrayType {
    fn from(info: TypeInfo) -> ArrayType {
        match info {
            TypeInfo::Array(arr) => arr,
            _ => panic!("TypeInfo is not an Array"),
        }
    }
}
impl Borrow<ArrayType> for TypeInfo {
    fn borrow(&self) -> &ArrayType {
        match self {
            TypeInfo::Array(arr) => arr,
            _ => panic!("TypeInfo is not an Array"),
        }
    }
}

impl From<ArrayRefType> for TypeInfo {
    fn from(info: ArrayRefType) -> TypeInfo {
        TypeInfo::ArrayRef(info)
    }
}
impl From<TypeInfo> for ArrayRefType {
    fn from(info: TypeInfo) -> ArrayRefType {
        match info {
            TypeInfo::ArrayRef(arr_ref) => arr_ref,
            _ => panic!("TypeInfo is not an ArrayRef"),
        }
    }
}
impl Borrow<ArrayRefType> for TypeInfo {
    fn borrow(&self) -> &ArrayRefType {
        match self {
            TypeInfo::ArrayRef(arr_ref) => arr_ref,
            _ => panic!("TypeInfo is not an ArrayRef"),
        }
    }
}

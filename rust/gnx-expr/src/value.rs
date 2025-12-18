use crate::array::{DataHandle, MutDataHandle, ArrayRefInfo, ArrayInfo};
use crate::backend::DeviceHandle;
use crate::device::DeviceInfo;

use crate::trace::Tracer;
use std::hash::{Hash, Hasher};
use std::any::Any;
use std::fmt::{Debug, Display};


pub trait ConcreteValue: Any + Send + Sync + Debug + Display {
    fn to_info(&self) -> ValueInfo;
}

// A generic concrete value.
// Array, ArrayRef, Device are enum variants
// for convenience.
#[derive(Debug)]
pub enum Value {
    // The physical (immutable) data for this array.
    Array(DataHandle),
    // The mutable data underlying this array.
    ArrayRef(MutDataHandle),
    // A handle to a device.
    Device(DeviceHandle),
    Other(Box<dyn ConcreteValue>)
}

impl ConcreteValue for Value {
    fn to_info(&self) -> ValueInfo {
        match self {
            Value::Array(data) => data.to_info(),
            Value::ArrayRef(data) => data.to_info(),
            Value::Device(device) => device.to_info(),
            Value::Other(value) => value.to_info(),
        }
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Array(data) => Display::fmt(data, f),
            Value::ArrayRef(data) => Display::fmt(data, f),
            Value::Device(device) => Display::fmt(device, f),
            Value::Other(value) => Display::fmt(value, f),
        }
    }
}

impl Value {
    pub fn new<T: ConcreteValue>(value: T) -> Self {
        let value = match castaway::cast!(value, Self) {
            Ok(value) => return value,
            Err(value) => value
        };
        let value = match castaway::cast!(value, DataHandle) {
            Ok(value) => return Value::Array(value),
            Err(value) => value
        };
        let value = match castaway::cast!(value, MutDataHandle) {
            Ok(value) => return Value::ArrayRef(value),
            Err(value) => value
        };
        let value = match castaway::cast!(value, DeviceHandle) {
            Ok(value) => return Value::Device(value),
            Err(value) => value
        };
        Value::Other(Box::new(value))
    }
    pub fn downcast<T: ConcreteValue>(self) -> Result<T, Self> {
        match self {
            Value::Array(data) => castaway::cast!(data, T).map_err(Value::Array),
            Value::ArrayRef(data) => castaway::cast!(data, T).map_err(Value::ArrayRef),
            Value::Device(device) => castaway::cast!(device, T).map_err(Value::Device),
            Value::Other(value) => {
                let v: &dyn Any = &value;
                if v.is::<T>() {
                    let value: Box<dyn Any> = value;
                    Ok(*value.downcast::<T>().unwrap())
                } else {
                    Err(Value::Other(value))
                }
            }
        }
    }
    pub fn downcast_ref<T: 'static>(&self) -> Result<&T, ()> {
        match self {
            Value::Array(data) => castaway::cast!(data, &T).ok(),
            Value::ArrayRef(data) => castaway::cast!(data, &T).ok(),
            Value::Device(device) => castaway::cast!(device, &T).ok(),
            Value::Other(value) => {
                let v: &dyn Any = value;
                v.downcast_ref()
            }
        }.ok_or(())
    }
    pub fn downcast_mut<T: 'static>(&mut self) -> Result<&mut T, ()> {
        match self {
            Value::Array(data) => castaway::cast!(data, &mut T).ok(),
            Value::ArrayRef(data) => castaway::cast!(data, &mut T).ok(),
            Value::Device(device) => castaway::cast!(device, &mut T).ok(),
            Value::Other(value) => {
                let v: &mut dyn Any = value;
                v.downcast_mut()
            }
        }.ok_or(())
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum ValueInfo {
    Array(ArrayInfo),
    ArrayRef(ArrayRefInfo),
    Device(DeviceInfo),
    Other(Box<dyn AnyInfo>)
}

impl ValueInfo {
    pub fn new<T: AnyInfo>(value: T) -> Self {
        let value = match castaway::cast!(value, Self) {
            Ok(value) => return value,
            Err(value) => value
        };
        let value = match castaway::cast!(value, ArrayInfo) {
            Ok(value) => return ValueInfo::Array(value),
            Err(value) => value
        };
        let value = match castaway::cast!(value, ArrayRefInfo) {
            Ok(value) => return ValueInfo::ArrayRef(value),
            Err(value) => value
        };
        let value = match castaway::cast!(value, DeviceInfo) {
            Ok(value) => return ValueInfo::Device(value),
            Err(value) => value
        };
        ValueInfo::Other(Box::new(value))
    }

    pub fn downcast<T: AnyInfo>(self) -> Result<T, Self> {
        match self {
            ValueInfo::Array(info) => castaway::cast!(info, T).map_err(ValueInfo::Array),
            ValueInfo::ArrayRef(info) => castaway::cast!(info, T).map_err(ValueInfo::ArrayRef),
            ValueInfo::Device(info) => castaway::cast!(info, T).map_err(ValueInfo::Device),
            ValueInfo::Other(info) => {
                let v: &dyn Any = &info;
                if v.is::<T>() {
                    let info: Box<dyn Any> = info;
                    Ok(*info.downcast::<T>().unwrap())
                } else {
                    Err(ValueInfo::Other(info))
                }
            }
        }
    }
    pub fn downcast_ref<T: AnyInfo>(&self) -> Result<&T, ()> {
        match self {
            ValueInfo::Array(info) => castaway::cast!(info, &T).ok(),
            ValueInfo::ArrayRef(info) => castaway::cast!(info, &T).ok(),
            ValueInfo::Device(info) => castaway::cast!(info, &T).ok(),
            ValueInfo::Other(info) => {
                let v: &dyn Any = info;
                v.downcast_ref()
            }
        }.ok_or(())
    }
    pub fn downcast_mut<T: AnyInfo>(&mut self) -> Result<&mut T, ()> {
        match self {
            ValueInfo::Array(info) => castaway::cast!(info, &mut T).ok(),
            ValueInfo::ArrayRef(info) => castaway::cast!(info, &mut T).ok(),
            ValueInfo::Device(info) => castaway::cast!(info, &mut T).ok(),
            ValueInfo::Other(info) => {
                let v: &mut dyn Any = info;
                v.downcast_mut()
            }
        }.ok_or(())
    }
}


impl Display for ValueInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValueInfo::Array(info) => Display::fmt(info, f),
            ValueInfo::ArrayRef(info) => Display::fmt(info, f),
            ValueInfo::Device(info) => Display::fmt(info, f),
            ValueInfo::Other(info) => Display::fmt(info, f),
        }
    }
}

pub trait Traceable : Into<Tracer<Self>> + From<Tracer<Self>> {
    type Concrete: ConcreteValue;
    type Info: AnyInfo;
}


pub struct Generic(Tracer<Generic>);

impl From<Generic> for Tracer<Generic> {
    fn from(value: Generic) -> Self {
        value.0
    }
}

impl From<Tracer<Generic>> for Generic {
    fn from(value: Tracer<Generic>) -> Self {
        Generic(value)
    }
}

impl Traceable for Generic {
    type Concrete = Value;
    type Info = ValueInfo;
}

// AnyInfo

pub trait AnyInfo: Any + Send + Sync + Debug + Display {
    fn dyn_hash(&self, state: &mut dyn std::hash::Hasher);
    fn dyn_eq(&self, other: &dyn AnyInfo) -> bool;
    fn dyn_clone(&self) -> Box<dyn AnyInfo>;
}

impl<T> AnyInfo for T 
    where T: Any + Send + Sync,
          T: Debug + Display + Clone + Hash + Eq {
    fn dyn_hash(&self, mut state: &mut dyn Hasher) {
        Hash::hash(self, &mut state);
    }
    fn dyn_eq(&self, other: &dyn AnyInfo) -> bool {
        let s: &dyn Any = other;
        match s.downcast_ref::<Self>() {
            Some(other) => self == other,
            None => false,
        }
    }
    fn dyn_clone(&self) -> Box<dyn AnyInfo> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn AnyInfo> {
    fn clone(&self) -> Self {
        self.dyn_clone()
    }
}
impl Hash for Box<dyn AnyInfo> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.type_id().hash(state);
        self.dyn_hash(state);
    }
}
impl PartialEq for Box<dyn AnyInfo> {
    fn eq(&self, other: &Self) -> bool {
        self.dyn_eq(other.as_ref())
    }
}
impl Eq for Box<dyn AnyInfo> {}
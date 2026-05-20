use super::trace::Tracer;
use std::any::{Any, TypeId};
use std::fmt::{Debug, Display};
use std::hash::{Hash, Hasher};

pub trait ConcreteValue: Any + Send + Sync + Debug + Display {
    fn to_info(&self) -> ValueInfo;
}

#[derive(Debug)]
pub struct Value(Box<dyn ConcreteValue>);
// Allows us to catch when a Value is downcast into a different type.
unsafe impl castaway::LifetimeFree for Value {}

impl ConcreteValue for Value {
    fn to_info(&self) -> ValueInfo { self.0.to_info() }
}

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(self.0.as_ref(), f)
    }
}

impl Value {
    pub fn new<T: ConcreteValue>(value: T) -> Self {
        match castaway::cast!(value, Self) {
            Ok(value) => return value,
            Err(value) => Value(Box::new(value)),
        }
    }
    pub fn downcast_into<T: ConcreteValue>(self) -> Result<T, Self> {
        // If T is Value, just return self
        castaway::cast!(self, T).map_err(|_value| {
            todo!()
        })
    }
    pub fn downcast<T: 'static>(&self) -> Result<&T, ()> {
        // If T is &Value, just return self
        castaway::cast!(self, &T).map_err(|_value| {
            todo!()
        })
    }
}

pub struct ValueInfo(Box<dyn AnyInfo>);

impl ValueInfo {
    pub fn new<T: AnyInfo>(value: T) -> Self {
        // if T is ValueInfo, use it directly
        match castaway::cast!(value, Self) {
            Ok(value) => return value,
            Err(value) => ValueInfo(Box::new(value)),
        }
    }

    pub unsafe fn downcast_into_unchecked<T: AnyInfo>(self) -> T {
        // If T is ValueInfo, use it directly. Otherwise upcast to dyn Any
        let value: Box<dyn Any> = match castaway::cast!(self, T) {
            Ok(value) => return value,
            Err(value) => value.0,
        };
        // TODO: use downcast_unchecked on Any when stabilized.
        // SAFETY: We know that the value is of type T because we checked the type_id above.
        let value: Box<T> = unsafe { value.downcast().unwrap_unchecked() };
        *value
    }

    pub fn downcast_into<T: AnyInfo>(self) -> Result<T, Self> {
        // If T is ValueInfo, just return self
        castaway::cast!(self, T).or_else(|_value| {
            let value: &dyn Any = &_value.0;
            if value.type_id() == TypeId::of::<T>() {
                let value: Box<dyn Any> = _value.0;
                let value: Box<T> = value.downcast().unwrap();
                return Ok(*value)
            } else {
                return Err(_value)
            }
        })
    }
    pub unsafe fn downcast_unchecked<T: AnyInfo>(&self) -> &T {
        // If T is &ValueInfo, use it directly. Otherwise upcast to dyn Any
        let value: &dyn Any = match castaway::cast!(self, &T) {
            Ok(value) => return value,
            Err(value) => &value.0,
        };
        // SAFETY: We know that the value is of type T because we checked the type_id above.
        let value: &T = unsafe { value.downcast_ref().unwrap_unchecked() };
        value
    }
    pub fn downcast<T: AnyInfo>(&self) -> Result<&T, ()> {
        // If T is &ValueInfo, just return self
        castaway::cast!(self, &T).or_else(|_value| {
            // upcast to dyn Any
            let value: &dyn Any = &_value.0;
            value.downcast_ref::<T>().ok_or(())
        })
    }
}

impl Display for ValueInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(self.0.as_ref(), f)
    }
}

pub trait Traceable: Into<Tracer<Self>> + From<Tracer<Self>> {
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

// Any type which implements Send + Sync + Debug + Display
// can be used as a ValueInfo.
pub trait AnyInfo: Any + Send + Sync + Debug + Display {
    fn dyn_hash(&self, state: &mut dyn std::hash::Hasher);
    fn dyn_eq(&self, other: &dyn AnyInfo) -> bool;
    fn dyn_clone(&self) -> Box<dyn AnyInfo>;
}

impl<T> AnyInfo for T
where
    T: Any + Send + Sync,
    T: Debug + Display + Clone + Hash + Eq,
{
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

impl Debug for ValueInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self.0.as_ref(), f)
    }
}

impl Clone for ValueInfo {
    fn clone(&self) -> Self {
        ValueInfo(self.0.dyn_clone())
    }
}
impl Hash for ValueInfo {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.type_id().hash(state);
        self.0.dyn_hash(state);
    }
}
impl PartialEq for ValueInfo {
    fn eq(&self, other: &Self) -> bool {
        self.0.dyn_eq(other.0.as_ref())
    }
}
impl Eq for ValueInfo {}

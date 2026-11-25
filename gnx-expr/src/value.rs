use std::any::{Any, TypeId};
use std::fmt::{Debug, Display};
use std::sync::Arc;

use crate::trace::Tracer;
use crate::var::{InfoCast, VarInfo};

pub trait ConcreteValue: Any + Send + Sync + Debug + Display {
    fn to_info(&self) -> VarInfo;
}

/// An immutable concrete value. Cloning retains its storage, without copying it.
#[derive(Clone, Debug)]
pub struct Value(Arc<dyn ConcreteValue>);
// Allows us to catch when a Value is downcast into a different type.
unsafe impl castaway::LifetimeFree for Value {}

impl ConcreteValue for Value {
    fn to_info(&self) -> VarInfo {
        self.0.to_info()
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(self.0.as_ref(), f)
    }
}

impl Value {
    pub fn new<T: ConcreteValue>(value: T) -> Self {
        match castaway::cast!(value, Self) {
            Ok(value) => value,
            Err(value) => Value(Arc::new(value)),
        }
    }
    /// Take the payload if it has the requested type and is uniquely held.
    pub fn downcast_into<T: ConcreteValue>(self) -> Result<T, Self> {
        // If T is Value, just return self
        castaway::cast!(self, T).or_else(|value| {
            let payload: &dyn Any = value.0.as_ref();
            if payload.type_id() == TypeId::of::<T>() {
                let payload: Arc<dyn Any + Send + Sync> = value.0;
                let payload = payload.downcast::<T>().ok().expect("checked value type");
                Arc::try_unwrap(payload).map_err(|shared| Value(shared))
            } else {
                Err(value)
            }
        })
    }
    #[allow(clippy::result_unit_err)]
    pub fn downcast<T: 'static>(&self) -> Result<&T, ()> {
        // If T is Value, just return self
        castaway::cast!(self, &T).or_else(|value| {
            let payload: &dyn Any = value.0.as_ref();
            payload.downcast_ref::<T>().ok_or(())
        })
    }
}

/// A kind of traced value: what its concrete values are, and what its
/// info must be.
pub trait Traceable: Into<Tracer<Self>> + From<Tracer<Self>> {
    type Concrete: ConcreteValue;
    /// Either `VarInfo`, for any info, or the archived type of one info.
    type Info: InfoCast;
}

/// A traced value of any kind.
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
    type Info = VarInfo;
}

use super::trace::Tracer;
use crate::{DynEq, DynHash};
use std::any::{Any, TypeId};
use std::fmt::{Debug, Display};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

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
            Ok(value) => return value,
            Err(value) => Value(Arc::new(value)),
        }
    }
    /// Take the payload if it has the requested type and is uniquely held.
    pub fn downcast_into<T: ConcreteValue>(self) -> Result<T, Self> {
        // If T is Value, just return self
        castaway::cast!(self, T).or_else(|_value| {
            let value: &dyn Any = _value.0.as_ref();
            if value.type_id() == TypeId::of::<T>() {
                let value: Arc<dyn Any + Send + Sync> = _value.0;
                let value = value.downcast::<T>().ok().expect("checked value type");
                Arc::try_unwrap(value).map_err(|shared| Value(shared))
            } else {
                Err(_value)
            }
        })
    }
    pub fn downcast<T: 'static>(&self) -> Result<&T, ()> {
        // If T is Value, just return self
        castaway::cast!(self, &T).or_else(|_value| {
            let value: &dyn Any = _value.0.as_ref();
            value.downcast_ref::<T>().ok_or(())
        })
    }
}

/// The dialect-defined metadata of a variable: the type of a value or cell.
/// An owned, cheaply cloned [`Info`] of any concrete type.
#[derive(Clone)]
pub struct VarInfo(Arc<dyn Info>);

impl From<&VarInfo> for VarInfo {
    fn from(info: &VarInfo) -> Self {
        info.clone()
    }
}

impl VarInfo {
    /// The symbol in each of the payload's symbol slots. See [`Info::symbols`].
    pub fn symbols(&self) -> &[crate::Symbol] {
        (*self.0).symbols()
    }
    /// Replace the symbol in every slot. See [`Info::remap_symbols`].
    pub fn remap_symbols(&self, mut f: impl FnMut(crate::Symbol) -> crate::Symbol) -> Self {
        (*self.0)
            .remap_symbols(&mut f)
            .unwrap_or_else(|| self.clone())
    }
    pub fn new<T: Info>(value: T) -> Self {
        // if T is VarInfo, use it directly
        match castaway::cast!(value, Self) {
            Ok(value) => return value,
            Err(value) => VarInfo(Arc::new(value)),
        }
    }

    // SAFETY: The caller has checked that the value is of type T.
    pub unsafe fn downcast_into_unchecked<T: Info + Clone>(self) -> T {
        // If T is VarInfo, use it directly. Otherwise upcast to dyn Any
        let value: Arc<dyn Any + Send + Sync> = match castaway::cast!(self, T) {
            Ok(value) => return value,
            Err(value) => value.0,
        };
        let value: Arc<T> = unsafe { value.downcast().unwrap_unchecked() };
        Arc::unwrap_or_clone(value)
    }

    pub fn downcast_into<T: Info + Clone>(self) -> Result<T, Self> {
        // If T is VarInfo, just return self
        castaway::cast!(self, T).or_else(|_value| {
            let value: &dyn Any = _value.0.as_ref();
            if value.type_id() == TypeId::of::<T>() {
                let value: Arc<dyn Any + Send + Sync> = _value.0;
                let value: Arc<T> = unsafe { value.downcast().unwrap_unchecked() };
                return Ok(Arc::unwrap_or_clone(value));
            } else {
                return Err(_value);
            }
        })
    }
    pub unsafe fn downcast_unchecked<T: Info>(&self) -> &T {
        // If T is &VarInfo, use it directly. Otherwise upcast to dyn Any
        let value: &dyn Any = match castaway::cast!(self, &T) {
            Ok(value) => return value,
            Err(value) => value.0.as_ref(),
        };
        // SAFETY: We know that the value is of type T because we checked the type_id above.
        let value: &T = unsafe { value.downcast_ref().unwrap_unchecked() };
        value
    }
    pub fn downcast<T: Info>(&self) -> Result<&T, ()> {
        // If T is &VarInfo, just return self
        castaway::cast!(self, &T).or_else(|_value| {
            let value: &dyn Any = _value.0.as_ref();
            value.downcast_ref::<T>().ok_or(())
        })
    }
}

impl Display for VarInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(self.0.as_ref(), f)
    }
}

pub trait Traceable: Into<Tracer<Self>> + From<Tracer<Self>> {
    type Concrete: ConcreteValue;
    type Info: Info;
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
    type Info = VarInfo;
}

/// Dialect-defined metadata of a variable.
///
/// Metadata can refer to region-local [`Symbol`](crate::Symbol)s through an
/// ordered list of symbol slots. The dialect owns the meaning of each slot.
/// An info without symbols needs only `impl Info for T {}`.
pub trait Info: Any + Send + Sync + Debug + Display + DynHash + DynEq {
    /// The symbol in each slot, in slot order. A symbol can fill several slots.
    fn symbols(&self) -> &[crate::Symbol] {
        &[]
    }
    /// Rebuild this info with `f` applied to the symbol in each slot, in slot
    /// order. `None` leaves the info unchanged, which is only correct for an
    /// info without symbols.
    fn remap_symbols(&self, _f: &mut dyn FnMut(crate::Symbol) -> crate::Symbol) -> Option<VarInfo> {
        None
    }
}

impl Info for VarInfo {
    fn symbols(&self) -> &[crate::Symbol] {
        (*self.0).symbols()
    }
    fn remap_symbols(&self, f: &mut dyn FnMut(crate::Symbol) -> crate::Symbol) -> Option<VarInfo> {
        (*self.0).remap_symbols(f)
    }
}

macro_rules! symbol_free_info {
    ($($t:ty),* $(,)?) => { $(impl Info for $t {})* };
}
symbol_free_info!(
    bool,
    char,
    u8,
    u16,
    u32,
    u64,
    usize,
    i8,
    i16,
    i32,
    i64,
    isize,
    &'static str,
    String,
);

impl Debug for VarInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self.0.as_ref(), f)
    }
}

impl Hash for VarInfo {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (*self.0).type_id().hash(state);
        (*self.0).dyn_hash(state);
    }
}
impl PartialEq for VarInfo {
    fn eq(&self, other: &Self) -> bool {
        (*self.0).dyn_eq(other.0.as_ref())
    }
}
impl Eq for VarInfo {}

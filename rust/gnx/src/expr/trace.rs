use std::borrow::Borrow;
use std::sync::Arc;

use super::{Op, TypeInfo, TypeValue};
use crate::array::{ArrayRefType, ArrayType, DataHandle};

#[allow(unused)]
pub struct Invocation {
    op: Op,
    inputs: Vec<GenericTracer>,
    total_outputs: usize,
}

#[allow(unused)]
pub enum Trace {
    // A realized value
    Physical(TypeValue),
    Placeholder,
    Input,
    Returned {
        invoke: Arc<Invocation>,
        index: usize,
    },
    // An invalid trace
    Error,
}

#[allow(unused)]
pub struct Traced {
    trace: Trace,
    is_abstract: bool,
    propagate_errors: bool,
    info: TypeInfo,
}

// A tracer wraps an Arc<Trace>
// and is typed for external API usage, but
// the type is effectively erased in storage by the
// Into<GenericType> + TryFrom<GenericType>, and
// Into<GenericValue> + TryFrom<GenericValue> impls
#[derive(Clone)]
pub struct Tracer<Info, Value>
where
    Info: Into<TypeInfo> + From<TypeInfo>,
    Value: Into<TypeValue> + From<TypeValue>,
    TypeInfo: Borrow<Info>,
    TypeValue: Borrow<Value>,
{
    traced: Arc<Traced>,
    _phantom: std::marker::PhantomData<(Info, Value)>,
}

impl<Info, Value> Tracer<Info, Value>
where
    Info: Into<TypeInfo> + From<TypeInfo>,
    Value: Into<TypeValue> + From<TypeValue>,
    TypeInfo: Borrow<Info>,
    TypeValue: Borrow<Value>,
{
    pub fn info(&self) -> &Info {
        self.traced.info.borrow()
    }
    pub fn value(&self) -> Option<&Value> {
        match &self.traced.trace {
            Trace::Physical(v) => Some(v.borrow()),
            _ => None,
        }
    }
}

pub type ArrayTracer = Tracer<ArrayType, DataHandle>;
pub type ArrayRefTracer = Tracer<ArrayRefType, DataHandle>;
pub type GenericTracer = Tracer<TypeInfo, TypeValue>;

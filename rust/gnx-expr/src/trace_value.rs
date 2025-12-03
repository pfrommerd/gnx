use crate::trace::{Invocation, Value};
use crate::util::OptionArcCell;

use std::sync::Arc;
use std::sync::OnceLock;

// A trace value that can be updated
// with a concrete value. Once updated
// with a value, the reference to the invocation is dropped!
// These are used for trace values 
// and can be eval'd to a concrete value.
#[derive(Clone)]
pub struct UpdatableValue {
    // Will be set to None if the trace is a concrete value.
    invocation: OptionArcCell<Invocation>,
    ret_idx: usize,
    // The concrete value for this tracer.
    value: OnceLock<Value>
}

// An abstract trace value cannot be updated with a concrete value.
// These do *not* have an ArcCell and so do not need to
// close the underlying arc in order to access the invocation.
pub enum AbstractValue {
    Placeholder,
    Returned { invocation: Arc<Invocation>, ret_idx: usize }
}

pub enum TraceValue {
    Abstract(AbstractValue),
    Updatable(UpdatableValue),
}

impl TraceValue {
    pub fn placeholder() -> Self {
        TraceValue::Abstract(AbstractValue::Placeholder)
    }
    pub fn concrete(value: Value) -> Self {
        TraceValue::Updatable(UpdatableValue {
            invocation: OptionArcCell::from(None),
            ret_idx: 0,
            value: value.into(),
        })
    }
    pub fn abstract_returned(invocation: Arc<Invocation>, ret_idx: usize) -> Self {
        TraceValue::Abstract(AbstractValue::Returned {
            invocation, ret_idx
        })
    }
    pub fn updatable_returned(invocation: Arc<Invocation>, ret_idx: usize) -> Self {
        TraceValue::Updatable(UpdatableValue {
            invocation: invocation.into(),
            ret_idx: ret_idx,
            value: OnceLock::new(),
        })
    }

    pub fn is_abstract(&self) -> bool {
        match self {
            TraceValue::Abstract(_) => true,
            TraceValue::Updatable(_) => false
        }
    }
}
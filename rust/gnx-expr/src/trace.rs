use std::sync::{Arc, OnceLock, RwLock, Weak};
use std::marker::PhantomData;

use crate::array::{Array, ArrayRef, ArrayInfo, DataHandle, MutDataHandle};
use crate::expr::Op;
use crate::trace_value::TraceValue;

#[derive(Clone, Debug)]
pub enum Value {
    // The physical (immutable) data for this array.
    Array(DataHandle),
    // The mutable data underlying this array.
    ArrayRef(MutDataHandle),
    // A handle to a device.
    Device,
    // A generic resource handle.
    Handle
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum ValueInfo {
    Array(ArrayInfo),
    Device(()),
    Handle(())
}

#[derive(Clone, Debug)]
pub enum ValueInfoRef<'r> {
    Array(&'r ArrayInfo),
    Device(()),
    Handle(())
}


impl Value {
    fn info(&self) -> ValueInfoRef<'_> {
        match self {
            Value::Array(data) => ValueInfoRef::Array(data.info()),
            _ => todo!(),
        }
    }
}

pub struct Trace {
    // Contains either the invocation, ret_idx,
    // is a placeholder, or is a concrete value
    value: TraceValue,
    info: ValueInfo,
    // The users of this trace
    used_by: RwLock<Vec<UsedIn>>
}

pub struct Invocation {
    op: Op,
    inputs: Vec<Tracer<Generic>>,
    outputs: OnceLock<Vec<WeakTracer<Generic>>>
}

#[derive(Clone)]
pub struct UsedIn {
    // Weak to prevent circular references.
    invocation: Weak<Invocation>,
    idx: usize,
}

impl Invocation {
    // Will return a vector of the output tracers.
    pub fn new(op: Op,
            inputs: Vec<Tracer<Generic>>,
            outputs: Vec<ValueInfo>) -> Vec<Tracer<Generic>> {
        let is_abstract = inputs.iter()
            .any(|x| x.shared.value.is_abstract());
        let invocation = Arc::new(Invocation {
            op,
            inputs,
            outputs: OnceLock::new(),
        });
        // Add UsedIn to the inputs
        for (idx, input) in invocation.inputs.iter().enumerate() {
            let mut w = input.shared.used_by.write().unwrap();
            w.push(UsedIn {
                invocation: Arc::downgrade(&invocation),
                idx
            })
        }
        let outputs: Vec<Tracer<Generic>> = outputs.into_iter().enumerate().map(|(index, info)| {
            Tracer::new(Arc::new(Trace {
                value: if is_abstract { 
                    TraceValue::abstract_returned(invocation.clone(), index)
                } else {
                    TraceValue::updatable_returned(invocation.clone(), index)
                },
                info: info,
                used_by: RwLock::new(Vec::new()),
            }))
        }).collect();
        // Add the outputs as outputs of the invocation
        invocation.outputs.set(
            outputs.iter().map(|o| o.weak()).collect()
        ).ok().unwrap();
        outputs
    }

    pub fn op(&self) -> &Op { &self.op }
}

pub trait BorrowFrom<T> {
    fn borrow_from(value: &T) -> &Self;
}

pub trait Traceable {
    type Concrete: Into<Value> + BorrowFrom<Value>;
    type Info: Into<ValueInfo> + BorrowFrom<ValueInfo>;
}

impl BorrowFrom<Value> for DataHandle {
    fn borrow_from(value: &Value) -> &Self {
        match value {
            Value::Array(data) => data,
            _ => panic!("Expected Array, got {:?}", value),
        }
    }
}
impl BorrowFrom<Value> for MutDataHandle {
    fn borrow_from(value: &Value) -> &Self {
        match value {
            Value::ArrayRef(data) => data,
            _ => panic!("Expected ArrayRef, got {:?}", value),
        }
    }
}

impl BorrowFrom<ValueInfo> for ArrayInfo {
    fn borrow_from(value: &ValueInfo) -> &Self {
        match value {
            ValueInfo::Array(info) => info,
            _ => panic!("Expected ArrayInfo, got {:?}", value),
        }
    }
}
impl BorrowFrom<Value> for Value {
    fn borrow_from(value: &Value) -> &Self { value }
}
impl BorrowFrom<ValueInfo> for ValueInfo {
    fn borrow_from(value: &ValueInfo) -> &Self { value }
}

impl From<DataHandle> for Value {
    fn from(value: DataHandle) -> Self { Value::Array(value) }
}
impl From<MutDataHandle> for Value {
    fn from(value: MutDataHandle) -> Self { Value::ArrayRef(value) }
}
impl From<ArrayInfo> for ValueInfo {
    fn from(value: ArrayInfo) -> Self { ValueInfo::Array(value) }
}

#[derive(Clone)]
pub struct Tracer<T: Traceable> {
    shared: Arc<Trace>,
    _phantom: PhantomData<T>
}

#[derive(Clone)]
pub struct WeakTracer<T: Traceable> {
    shared: Weak<Trace>,
    _phantom: PhantomData<T>
}

impl<T: Traceable> Tracer<T> {
    // Create a placeholder tracer (used for abstract inputs, etc.)
    fn new(shared: Arc<Trace>) -> Self {
        Tracer { shared, _phantom: PhantomData }
    }

    pub fn placeholder(info: T::Info) -> Self {
        todo!()
    }
    // Create a tracer from a concrete value.
    pub fn concrete(value: T::Concrete) -> Self {
        let value: Value = value.into();
        todo!()
    }

    pub fn is_abstract(&self) -> bool {
        self.shared.value.is_abstract()
    }

    pub fn info(&self) -> &T::Info {
        T::Info::borrow_from(&self.shared.info)
    }

    pub fn weak(&self) -> WeakTracer<T> {
        WeakTracer { shared: Arc::downgrade(&self.shared), _phantom: PhantomData }
    }

    // Cast to a generic tracer.
    pub fn generic(self) -> Tracer<Generic> {
        Tracer { shared: self.shared, _phantom: PhantomData }
    }
    // Cast to a particular type of tracer.
    pub fn cast<U: Traceable>(self) -> Tracer<U> {
        // Try to borrow from the shared info.
        U::Info::borrow_from(&self.shared.info);
        Tracer { shared: self.shared, _phantom: PhantomData }
    }
}

impl<T: Traceable> WeakTracer<T> {
    pub fn upgrade(&self) -> Option<Tracer<T>> {
        self.shared.upgrade().map(|shared| Tracer { shared, _phantom: PhantomData })
    }

    pub fn generic(&self) -> WeakTracer<Generic> {
        WeakTracer { shared: self.shared.clone(), _phantom: PhantomData }
    }
}

pub struct Generic;


impl Traceable for Generic {
    type Concrete = Value;
    type Info = ValueInfo;
}
impl Traceable for Array {
    type Concrete = DataHandle;
    type Info = ArrayInfo;
}
impl Traceable for ArrayRef {
    type Concrete = MutDataHandle;
    type Info = ArrayInfo;
}
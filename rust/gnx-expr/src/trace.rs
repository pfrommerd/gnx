use std::sync::{Arc, OnceLock, RwLock, Weak};
use std::marker::PhantomData;
use std::fmt::{Debug, Display};

use crate::expr::Op;
use crate::util::DgArc;

pub use crate::value::{
    Traceable, Generic,
    Value, ValueInfo,
    ConcreteValue,
};

// An abstract trace value cannot be updated with a concrete value.
// These do *not* have an ArcCell and so do not need to
// close the underlying arc in order to access the invocation.
enum AbstractTrace {
    Placeholder,
    Returned { invocation: Arc<Invocation>, ret_idx: usize }
}


struct UpdatableTrace {
    // Will be set to None if the trace is a concrete value.
    invocation: DgArc<Invocation>,
    ret_idx: usize,
    // The concrete value for this tracer.
    value: OnceLock<Value>
}

enum TraceInner {
    Abstract(AbstractTrace),
    Updatable(UpdatableTrace),
    Constant(Value),
}

pub struct Trace {
    inner: TraceInner,
    info: ValueInfo,
    // The users of this trace
    used_by: RwLock<Vec<UsedIn>>
}

impl Trace {
    fn debug_fmt(&self, name: &str, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct(name).finish()
    }
}

// For debugging purposes, make this a bit cleaner
// than the auto-derived Debug impl would be
impl Debug for Trace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.debug_fmt("Trace", f)
    }
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
    pub fn op(&self) -> &Op { &self.op }
}

impl Trace {
    fn abstract_retval(invocation: Arc<Invocation>, ret_idx: usize, info: ValueInfo) -> Self {
        Trace {
            inner: TraceInner::Abstract(AbstractTrace::Returned { invocation, ret_idx }),
            info,
            used_by: RwLock::new(Vec::new())
        }
    }
    fn updatable_retval(invocation: Arc<Invocation>, ret_idx: usize, info: ValueInfo) -> Self {
        Trace {
            inner: TraceInner::Updatable(UpdatableTrace {
                invocation: invocation.into(),
                ret_idx, value: OnceLock::new()
            }),
            info,
            used_by: RwLock::new(Vec::new())
        }
    }
    fn placeholder(info: ValueInfo) -> Self {
        Trace {
            inner: TraceInner::Abstract(AbstractTrace::Placeholder),
            info,
            used_by: RwLock::new(Vec::new())
        }
    }
    fn concrete(value: Value, info: ValueInfo) -> Self {
        Trace {
            inner: TraceInner::Updatable(UpdatableTrace {
                invocation: DgArc::new(),
                ret_idx: 0, value: value.into()
            }),
            info,
            used_by: RwLock::new(Vec::new())
        }
    }

    fn is_abstract(&self) -> bool { matches!(self.inner, TraceInner::Abstract(_)) }

    fn info(&self) -> &ValueInfo {
        &self.info
    }

    fn update(&self, value: Value) {
        match &self.inner {
            TraceInner::Updatable(updatable) => {
                updatable.value.set(value).ok().expect("Value has already been set");
                // Downgrade the invocation to allow
                // for dropping the associated graph.
                updatable.invocation.downgrade();
            }
            TraceInner::Abstract(_) => panic!("Cannot update an abstract Tracer"),
            TraceInner::Constant(_) => panic!("Cannot update a constant Tracer"),
        }
    }

    fn try_concrete(&self) -> Option<&Value> {
        match &self.inner {
            TraceInner::Updatable(updatable) => updatable.value.get(),
            TraceInner::Constant(value) => Some(value),
            _ => None,
        }
    }
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

// For debugging purposes, make this a bit cleaner
impl<T: Traceable> Debug for Tracer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.shared.debug_fmt(std::any::type_name::<Tracer<T>>(), f)
    }
}


impl<T: Traceable> Tracer<T> {
    fn new(trace: Trace) -> Self {
        Tracer { shared: Arc::new(trace), _phantom: PhantomData }
    }

    pub fn invoke(op: Op,
            inputs: Vec<Tracer<Generic>>,
            outputs: Vec<ValueInfo>) -> Vec<Tracer<Generic>> {
        let is_abstract = inputs.iter()
            .any(|x| x.shared.is_abstract());
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
        let outputs: Vec<Tracer<Generic>> = if is_abstract { 
            outputs.into_iter().enumerate().map(|(index, info)| {
                Tracer::new(Trace::abstract_retval(invocation.clone(), index, info))
            }).collect()
        } else {
            outputs.into_iter().enumerate().map(|(index, info)| {
                Tracer::new(Trace::abstract_retval(invocation.clone(), index, info))
            }).collect()
        };
        // Add the outputs as outputs of the invocation
        invocation.outputs.set(
            outputs.iter().map(|o| o.weak()).collect()
        ).ok().unwrap();
        outputs
    }

    pub fn placeholder(info: T::Info) -> Self {
        Tracer::new(Trace::placeholder(ValueInfo::new(info)))
    }

    // Create a tracer from a concrete value.
    pub fn concrete(value: T::Concrete, info: T::Info) -> Self {
        Tracer::new(Trace::concrete(Value::new(value), ValueInfo::new(info)))
    }

    pub fn is_abstract(&self) -> bool {
        self.shared.is_abstract()
    }

    pub fn info(&self) -> &T::Info {
        self.shared.info.downcast_ref().unwrap()
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
        Tracer { shared: self.shared, _phantom: PhantomData }
    }

    // Try to extract the concrete value from this tracer.
    pub fn try_concrete(&self) -> Option<&T::Concrete> {
        self.shared.try_concrete().map(|value| value.downcast_ref().expect("Tracer contained incorrect type"))
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
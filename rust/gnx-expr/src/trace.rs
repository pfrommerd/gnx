use std::collections::BTreeMap;
use std::sync::{Arc, Weak};
use std::marker::{PhantomData, PhantomPinned};

use crate::util::{ArcCell, SyncUnsafeCell};
use crate::expr::Op;

pub enum Value {
    Array,
    ArrayRef,
    // A generic resource handle.
    Handle
}

impl Value {
    fn matches_info(&self, info: &ValueInfo) -> bool { true }
}

#[derive(Clone)]
pub enum ValueInfo {
    Array
}

pub struct Invocation {
    op: Box<dyn Op>,
    // Any closed-over tracers
    closure: Vec<Tracer<Generic>>,
    // any explicit inputs to the invocation
    inputs: Vec<Tracer<Generic>>,
    // This cell is populated upon construction
    // of the invocation and then never modified again!
    // Thus we use a sync unsafe cell to allow concurrent
    // zero-overhead access to the weak outputs after construction.
    outputs: SyncUnsafeCell<Vec<WeakTracer<Generic>>>,
}

pub struct Returned {
    invocation: Arc<Invocation>,
    index: usize,
    info: ValueInfo,
}

impl Invocation {
    // Will return a vector of the output tracers.
    pub fn new<O: Op>(op: O,
            closure: Vec<Tracer<Generic>>,
            inputs: Vec<Tracer<Generic>>,
            outputs: Vec<ValueInfo>) -> (Arc<Invocation>, Vec<Tracer<Generic>>) {
        let invocation = Arc::new(Invocation {
            op: Box::new(op),
            closure: closure,
            inputs: inputs,
            outputs: SyncUnsafeCell::new(Vec::new())
        });
        let outputs: Vec<Tracer<Generic>> = outputs.into_iter().enumerate().map(|(index, info)| {
            Tracer::new(Trace::Op(Returned {
                invocation: invocation.clone(),
                index: index,
                info: info,
            }))
        }).collect();
        // Populate invocation.outputs with the weak tracers.
        // SAFETY: This is the only time we mutably access self.outputs!
        unsafe {
            let weak_outputs = &mut *invocation.outputs.get();
            std::mem::swap(weak_outputs,
                &mut outputs.iter().map(|output| output.weak()).collect()
            );
        }
        (invocation, outputs)
    }

    pub fn op(&self) -> &dyn Op { &*self.op }
    pub fn inputs(&self) -> &Vec<Tracer<Generic>> { &self.inputs }
    pub fn outputs(&self) -> &Vec<WeakTracer<Generic>> {
        // SAFETY: We only immutably access self.outputs after construction.
        unsafe {
            let weak_outputs = &*self.outputs.get();
            weak_outputs
        }
    }
}


#[allow(unused)]
pub enum Trace {
    // A realized value and the info it was realized with.
    // a value can match multiple possible info types
    // (e.g. a 1-D array with known shape can match an info containing a dynamic shape)
    Physical(Value, ValueInfo),
    Placeholder(ValueInfo),
    Op(Returned),
    Error(ValueInfo),
}

impl Trace {
    pub fn info(&self) -> &ValueInfo {
        match self {
            Trace::Physical(_, info) => info,
            Trace::Placeholder(info) => info,
            Trace::Op(returned) => &returned.info,
            Trace::Error(info) => info,
        }
    }
}

// Acts like a Cell<Trace>. Uses an ArcCell
// internally to allow atomic updates to the trace
// without the overhead of a Mutex.
#[derive(Clone)]
pub struct TraceCell {
    trace: ArcCell<Trace>,
    // A trace cell does not allow changing the value info.
    // This allows us to get a reference to the info 
    // without cloning the underlying Arc<Trace>.
    info: ValueInfo,
    _pinned: PhantomPinned
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TracerId(usize);

impl TraceCell {
    pub fn new(trace: Trace) -> Self {
        let info = trace.info().clone();
        TraceCell { trace: ArcCell::new(Arc::new(trace)), info, _pinned: PhantomPinned }
    }
    pub fn info(&self) -> &ValueInfo {
        &self.info
    }
    pub fn get(&self) -> Arc<Trace> {
        self.trace.get()
    }
    // Realize the trace with a given value.
    pub fn realize(&self, value: Value) {
        if !value.matches_info(&self.info) {
            panic!("Internal error: realized tracer value does not match tracer info");
        }
        let trace = Arc::new(Trace::Physical(value, self.info.clone()));
        self.trace.set(trace);
    }
}

pub trait BorrowFrom<T> {
    fn borrow_from(value: &T) -> &Self;
} 

impl<T> BorrowFrom<T> for T {
    fn borrow_from(value: &T) -> &Self { value }
}

pub trait Traceable {
    type Concrete: BorrowFrom<Value>;
    type Info: BorrowFrom<ValueInfo>;
}

#[derive(Clone)]
pub struct Tracer<T: Traceable> {
    shared: Arc<TraceCell>,
    _phantom: PhantomData<T>
}

#[derive(Clone)]
pub struct WeakTracer<T: Traceable> {
    shared: Weak<TraceCell>,
    _phantom: PhantomData<T>
}

impl<T: Traceable> Tracer<T> {
    pub fn new(trace: Trace) -> Self {
        let cell = TraceCell::new(trace);
        Tracer { shared: Arc::new(cell), _phantom: PhantomData }
    }
    pub fn weak(&self) -> WeakTracer<T> {
        WeakTracer { shared: Arc::downgrade(&self.shared), _phantom: PhantomData }
    }
    pub fn generic(&self) -> Tracer<Generic> {
        Tracer { shared: self.shared.clone(), _phantom: PhantomData }
    }
    // Always succeeds but the resulting tracer may panic when calling info()
    pub fn cast<U: Traceable>(&self) -> Tracer<U> {
        Tracer { shared: self.shared.clone(), _phantom: PhantomData }
    }

    pub fn info(&self) -> &T::Info {
        T::Info::borrow_from(self.shared.info())
    }

    pub fn trace(&self) -> Arc<Trace> {
        self.shared.trace.get()
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
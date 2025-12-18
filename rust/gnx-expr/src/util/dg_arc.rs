use std::sync::{Arc, RwLock, Weak};

pub enum MaybeArcInner<T: ?Sized> {
    Arc(Arc<T>),
    Weak(Weak<T>),
}

impl<T: ?Sized> Clone for MaybeArcInner<T> {
    fn clone(&self) -> Self {
        match self {
            MaybeArcInner::Arc(arc) => MaybeArcInner::Arc(arc.clone()),
            MaybeArcInner::Weak(weak) => MaybeArcInner::Weak(weak.clone()),
        }
    }
}

// A downgradable arc that can be
// downgraded to a weak reference.
pub struct DgArc<T: ?Sized> {
    inner: RwLock<MaybeArcInner<T>>,
}

impl<T: ?Sized> Clone for DgArc<T> {
    fn clone(&self) -> Self {
        let w = self.inner.read().unwrap();
        DgArc { inner: RwLock::new(w.clone()) }
    }
}

impl<T> From<T> for DgArc<T> {
    fn from(value: T) -> Self {
        DgArc { inner: RwLock::new(MaybeArcInner::Arc(Arc::new(value))) }
    }
}
impl<T: ?Sized> From<Arc<T>> for DgArc<T> {
    fn from(value: Arc<T>) -> Self {
        DgArc { inner: RwLock::new(MaybeArcInner::Arc(value)) }
    }
}

impl<T> DgArc<T> {
    pub fn new() -> Self {
        DgArc { inner: RwLock::new(MaybeArcInner::Weak(Weak::new())) }
    }
}

impl<T: ?Sized> DgArc<T> {
    pub fn downgrade(&self) { 
        let mut w = self.inner.write().unwrap();
        match &*w {
            MaybeArcInner::Arc(arc) => {
                *w = MaybeArcInner::Weak(Arc::downgrade(arc));
            }
            MaybeArcInner::Weak(_) => {}
        }
    }
    pub fn upgrade(&self) {
        let mut w = self.inner.write().unwrap();
        match &*w {
            MaybeArcInner::Arc(_) => {},
            MaybeArcInner::Weak(weak) => {
                let v = weak.upgrade();
                if let Some(v) = v {
                    *w = MaybeArcInner::Arc(v);
                }
            }
        }
    }

    pub fn get(&self) -> Option<Arc<T>> {
        let w = self.inner.read().unwrap();
        match &*w {
            MaybeArcInner::Arc(arc) => Some(arc.clone()),
            MaybeArcInner::Weak(weak) => weak.upgrade(),
        }
    }
}

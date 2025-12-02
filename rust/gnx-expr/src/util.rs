use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::cell::UnsafeCell;
use std::marker::PhantomData;
use std::any::TypeId;
use std::mem;

use std::hash::{Hash, Hasher};
use std::any::Any;

// An ArcCell, a vendored version of
// crossbeam::atomic::ArcCell
// but without the crossbeam dependency.
// It can be used to store an Arc and provide an atomic
// way of changing the Arc.
#[derive(Debug)]
pub struct ArcCell<T>(AtomicUsize, PhantomData<Arc<T>>);

impl<T> Drop for ArcCell<T> {
    fn drop(&mut self) {
        self.take();
    }
}

impl<T> Clone for ArcCell<T> {
    fn clone(&self) -> Self {
        ArcCell::new(self.get())
    }
}

impl<T> ArcCell<T> {
    /// Creates a new `ArcCell`.
    pub fn new(t: Arc<T>) -> ArcCell<T> {
        ArcCell(AtomicUsize::new(unsafe { mem::transmute(t) }), PhantomData)
    }

    fn take(&self) -> Arc<T> {
        loop {
            match self.0.swap(0, Ordering::Acquire) {
                0 => {}
                n => return unsafe { mem::transmute(n) },
            }
        }
    }

    fn put(&self, t: Arc<T>) {
        debug_assert_eq!(self.0.load(Ordering::SeqCst), 0);
        self.0
            .store(unsafe { mem::transmute(t) }, Ordering::Release);
    }

    /// Stores a new value in the `ArcCell`, returning the previous
    /// value.
    pub fn set(&self, t: Arc<T>) -> Arc<T> {
        let old = self.take();
        self.put(t);
        old
    }

    /// Returns a copy of the value stored by the `ArcCell`.
    pub fn get(&self) -> Arc<T> {
        let t = self.take();
        // NB: correctness here depends on Arc's clone impl not panicking
        let out = t.clone();
        self.put(t);
        out
    }
}

impl<T: Default> Default for ArcCell<T> {
    fn default() -> Self {
        ArcCell::new(Arc::default())
    }
}

impl<T> From<Arc<T>> for ArcCell<T> {
    fn from(value: Arc<T>) -> Self {
        ArcCell::new(value)
    }
}

impl<T> From<T> for ArcCell<T> {
    fn from(value: T) -> Self {
        ArcCell::new(Arc::new(value))
    }
}
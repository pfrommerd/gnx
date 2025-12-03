use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::marker::PhantomData;
use std::mem;

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

// An ArcCell, a vendored version of
// crossbeam::atomic::ArcCell
// but without the crossbeam dependency.
// It can be used to store an Arc and provide an atomic
// way of changing the Arc.
#[derive(Debug)]
pub struct OptionArcCell<T>(
    AtomicUsize,
    PhantomData<Arc<T>>
);

impl<T> Drop for OptionArcCell<T> {
    fn drop(&mut self) {
        self.take();
    }
}

impl<T> Clone for OptionArcCell<T> {
    fn clone(&self) -> Self {
        OptionArcCell::new(self.get())
    }
}

impl<T> OptionArcCell<T> {
    /// Creates a new `ArcCell`.
    pub fn new(t: Option<Arc<T>>) -> OptionArcCell<T> {
        match t {
            Some(t) => {
                let ptr: usize = unsafe { mem::transmute(t) };
                OptionArcCell(AtomicUsize::new(ptr + 1), PhantomData)
            },
            None => {
                OptionArcCell(AtomicUsize::new(1), PhantomData)
            }
        }
    }

    fn take(&self) -> Option<Arc<T>> {
        loop {
            match self.0.swap(0, Ordering::Acquire) {
                // someone else is currently swapping,
                // loop until we can swap
                0 => {},
                1 => return None,
                n => return Some(unsafe { mem::transmute(n - 1) }),
            }
        }
    }

    fn put(&self, t: Option<Arc<T>>) {
        // Require that the current value is 0 (i.e. no value is stored in the cell)
        debug_assert_eq!(self.0.load(Ordering::SeqCst), 0);
        match t {
            Some(t) => {
                let ptr: usize = unsafe { mem::transmute(t) };
                self.0.store(ptr + 1, Ordering::Release);
            }
            None => {
                self.0.store(1, Ordering::Release);
            }
        }
    }

    /// Stores a new value in the `OptionArcCell`,
    /// returning the previous value, if any.
    pub fn set(&self, t: Option<Arc<T>>) -> Option<Arc<T>> {
        let old = self.take();
        self.put(t);
        old
    }

    /// Returns a copy of the value stored by the `ArcCell`.
    pub fn get(&self) -> Option<Arc<T>> {
        let t = self.take();
        // NB: correctness here depends on Arc's clone impl not panicking
        let out = t.clone();
        self.put(t);
        out
    }
}

impl<T> Default for OptionArcCell<T> {
    fn default() -> Self {
        OptionArcCell::new(None)
    }
}

impl<T> From<Arc<T>> for OptionArcCell<T> {
    fn from(value: Arc<T>) -> Self {
        OptionArcCell::new(Some(value))
    }
}

impl<T> From<Option<Arc<T>>> for OptionArcCell<T> {
    fn from(value: Option<Arc<T>>) -> Self {
        OptionArcCell::new(value)
    }
}

impl<T> From<T> for OptionArcCell<T> {
    fn from(value: T) -> Self {
        OptionArcCell::new(Some(Arc::new(value)))
    }
}
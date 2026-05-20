use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::Deref;
use std::sync::{Arc, RwLock};

use crate::{Generic, ValueInfo};

use super::context::{CellUpdate, ContextID, TraceContext};
use super::{TraceRef, Traceable, Tracer};

/// Stable identity for a [`TraceCell`].
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct CellKey(usize);

impl CellKey {
    pub fn from(cell: &TraceCellRef) -> Self {
        CellKey(Arc::as_ptr(&cell.0) as usize)
    }
}

/// Mutable traced slot (JAX-style ref). [`TraceRef`] values are immutable.
pub struct TraceCell {
    context_id: ContextID,
    type_info: ValueInfo,
    cell: RwLock<TraceRef>
}

/// Shared handle to a mutable [`TraceCell`].
#[repr(transparent)]
#[derive(Clone)]
pub struct TraceCellRef(Arc<TraceCell>);

impl TraceCellRef {
    pub fn new(base: TraceRef) -> Self {
        let context_id = TraceContext::current_id();
        TraceCellRef(Arc::new(TraceCell {
            context_id,
            type_info: base.info().clone(),
            cell: RwLock::new(base),
        }))
    }

    pub fn type_info(&self) -> &ValueInfo {
        &self.0.type_info
    }

    /// Resolve the trace for this cell in the active context.
    pub fn get(&self) -> TraceRef {
        self.0.get(self)
    }

    /// Bind the cell: update home storage in-context, otherwise via [`TraceContext`].
    pub fn set(&self, value: TraceRef) {
        self.0.set(self, value);
    }
}

impl Deref for TraceCellRef {
    type Target = TraceCell;

    fn deref(&self) -> &TraceCell {
        &self.0
    }
}

impl From<Arc<TraceCell>> for TraceCellRef {
    fn from(value: Arc<TraceCell>) -> Self {
        TraceCellRef(value)
    }
}

impl From<TraceCellRef> for Arc<TraceCell> {
    fn from(value: TraceCellRef) -> Self {
        value.0
    }
}

impl Debug for TraceCellRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&*self.0, f)
    }
}

impl Debug for TraceCell {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let value = self.cell.read().unwrap().clone();
        f.debug_struct("TraceCell")
            .field("ctx", &self.context_id)
            .field("type_info", &self.type_info)
            .field("value", &value)
            .finish()
    }
}

impl TraceCell {
    pub fn context_id(&self) -> ContextID {
        self.context_id
    }

    fn get(&self, cell_ref: &TraceCellRef) -> TraceRef {
        if self.context_id == TraceContext::current_id() {
            return self.cell.read().unwrap().clone();
        }
        TraceContext::active()
            .cell_override(CellKey::from(cell_ref))
            .unwrap_or_else(|| self.cell.read().unwrap().clone())
    }

    fn set(&self, cell_ref: &TraceCellRef, value: TraceRef) {
        if self.context_id == TraceContext::current_id() {
            *self.cell.write().unwrap() = value;
            return;
        }
        let ctx = TraceContext::active();
        let effective = self.get(cell_ref);
        if value.context_id() == effective.context_id() {
            ctx.override_cell(CellKey::from(cell_ref), value);
        } else {
            ctx.register_update(CellUpdate {
                cell: cell_ref.clone(),
                value,
            });
        }
    }
}

/// Typed handle to a [`TraceCell`] (analogous to [`Tracer`] for [`Trace`]).
pub struct TracerCell<T: Traceable> {
    shared: TraceCellRef,
    _phantom: PhantomData<T>,
}

impl<T: Traceable> TracerCell<T> {
    pub fn new(trace: Tracer<T>) -> Self {
        TracerCell {
            shared: TraceCellRef::new(trace.trace_ref().clone()),
            _phantom: PhantomData,
        }
    }

    pub fn cell_ref(&self) -> &TraceCellRef {
        &self.shared
    }

    pub fn context_id(&self) -> ContextID {
        self.shared.context_id()
    }

    pub fn get(&self) -> Tracer<T> {
        Tracer::unchecked_new(self.shared.get())
    }

    pub fn set(&self, value: Tracer<T>) {
        self.shared.set(value.trace_ref().clone());
    }

    pub fn try_cast_into<U: Traceable>(self) -> Result<TracerCell<U>, ()> {
        match self.shared.type_info().downcast::<U::Info>() {
            Ok(_) => Ok(TracerCell { shared: self.shared, _phantom: PhantomData }),
            Err(_) => Err(()),
        }
    }

    pub fn try_cast<U: Traceable>(&self) -> Result<&TracerCell<U>, &Self> {
        match self.shared.type_info().downcast::<U::Info>() {
            Ok(_) => Ok(unsafe { std::mem::transmute(self) }),
            Err(_) => Err(self),
        }
    }
}

// SAFETY: TracerCell<Generic> and TraceCellRef share the same address (TraceCellRef is field 0).
impl From<&TraceCellRef> for &TracerCell<Generic> {
    fn from(r: &TraceCellRef) -> Self {
        unsafe { std::mem::transmute(r) }
    }
}

impl<T: Traceable> From<TracerCell<T>> for TraceCellRef {
    fn from(c: TracerCell<T>) -> Self { c.shared }
}

impl From<TraceCellRef> for TracerCell<Generic> {
    fn from(c: TraceCellRef) -> Self {
        TracerCell { shared: c, _phantom: PhantomData }
    }
}

impl<T: Traceable> Clone for TracerCell<T> {
    fn clone(&self) -> Self {
        TracerCell {
            shared: self.shared.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<T: Traceable> std::fmt::Debug for TracerCell<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let value = self.shared.get().clone();
        f.debug_struct("TracerCell").field("shared", &self.shared).field("value", &value).finish()
    }
}

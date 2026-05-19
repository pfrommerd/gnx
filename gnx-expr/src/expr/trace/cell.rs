use std::fmt::Debug;
use std::sync::{Arc, RwLock};

use std::marker::PhantomData;

use super::context::{CellUpdate, ContextID, TraceContext};
use super::{TraceRef, Traceable, Tracer};

/// Stable identity for a [`TraceCell`].
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct CellKey(usize);

impl CellKey {
    pub fn from(cell: &TraceCellRef) -> Self {
        CellKey(Arc::as_ptr(cell) as usize)
    }
}

/// Home-context storage for a cell's current trace binding.
struct Cell {
    trace: TraceRef,
}

/// Mutable traced slot (JAX-style ref). [`TraceRef`] values are immutable.
pub struct TraceCell {
    context_id: ContextID,
    base: TraceRef,
    cell: RwLock<Cell>,
}

pub type TraceCellRef = Arc<TraceCell>;

impl Debug for TraceCell {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TraceCell")
            .field("ctx", &self.context_id)
            .field("base", &self.base)
            .finish()
    }
}

impl TraceCell {
    pub fn new(base: TraceRef) -> TraceCellRef {
        let context_id = TraceContext::current_id();
        Arc::new(TraceCell {
            context_id,
            base: base.clone(),
            cell: RwLock::new(Cell { trace: base }),
        })
    }

    pub fn context_id(&self) -> ContextID {
        self.context_id
    }

    pub fn base(&self) -> &TraceRef {
        &self.base
    }

    fn home_trace(&self) -> TraceRef {
        self.cell.read().unwrap().trace.clone()
    }

    /// Resolve the trace for this cell in the active context.
    pub fn get(self: &Arc<Self>) -> TraceRef {
        if self.context_id == TraceContext::current_id() {
            return self.home_trace();
        }
        TraceContext::active()
            .cell_override(CellKey::from(self))
            .unwrap_or_else(|| self.home_trace())
    }

    /// Bind the cell: update home storage in-context, otherwise via [`TraceContext`].
    pub fn set(self: &Arc<Self>, value: TraceRef) {
        if self.context_id == TraceContext::current_id() {
            self.cell.write().unwrap().trace = value;
            return;
        }
        let ctx = TraceContext::active();
        let effective = self.get();
        if value.context_id() == effective.context_id() {
            ctx.override_cell(CellKey::from(self), value);
        } else {
            ctx.register_update(CellUpdate {
                cell: self.clone(),
                value,
            });
        }
    }
}

/// Typed handle to a [`TraceCell`] (analogous to [`Tracer`] for [`Trace`]).
pub struct TracerCell<T: Traceable> {
    cell: TraceCellRef,
    _phantom: PhantomData<T>,
}

impl<T: Traceable> TracerCell<T> {
    pub fn new(trace: Tracer<T>) -> Self {
        TracerCell {
            cell: TraceCell::new(trace.trace_ref().clone()),
            _phantom: PhantomData,
        }
    }

    pub fn cell_ref(&self) -> &TraceCellRef {
        &self.cell
    }

    pub fn context_id(&self) -> ContextID {
        self.cell.context_id()
    }

    pub fn get(&self) -> Tracer<T> {
        Tracer::new(self.cell.get())
    }

    pub fn set(&self, value: Tracer<T>) {
        self.cell.set(value.trace_ref().clone());
    }
}

impl<T: Traceable> Clone for TracerCell<T> {
    fn clone(&self) -> Self {
        TracerCell {
            cell: self.cell.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<T: Traceable> std::fmt::Debug for TracerCell<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TracerCell").field("cell", &self.cell).finish()
    }
}

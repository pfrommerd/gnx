use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::Deref;
use std::sync::{Arc, RwLock};

use crate::{Generic, InfoCast, Invocation, VarInfo};

use super::context::{ContextID, TraceContext};
use super::{Trace, Traceable, Tracer};

/// Stable identity for a [`TraceCell`].
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct CellKey(usize);

impl CellKey {
    pub fn from(cell: &CellTrace) -> Self {
        CellKey(Arc::as_ptr(&cell.0) as usize)
    }
}

/// Mutable traced slot (JAX-style ref). [`Trace`] values are immutable.
pub struct TraceCell {
    context_id: ContextID,
    type_info: VarInfo,
    /// The current "home" trace value, held under its own lock. Input cells
    /// (`is_input`) initialize this to a placeholder trace.
    home: RwLock<Trace>,
    /// Whether this cell is an abstract input/placeholder. Input cells are
    /// always treated as maybe-aliased (their relationship to other inputs is
    /// unknown at trace time).
    is_input: bool,
}

/// Shared handle to a mutable [`TraceCell`].
#[repr(transparent)]
#[derive(Clone)]
pub struct CellTrace(Arc<TraceCell>);

impl CellTrace {
    pub fn new(base: Trace) -> Self {
        CellTrace(Arc::new(TraceCell {
            context_id: TraceContext::current_id(),
            type_info: base.info().clone(),
            home: RwLock::new(base),
            is_input: false,
        }))
    }

    pub fn placeholder(info: VarInfo) -> Self {
        CellTrace(Arc::new(TraceCell {
            context_id: TraceContext::current_id(),
            type_info: info.clone(),
            home: RwLock::new(Trace::placeholder(info)),
            is_input: true,
        }))
    }

    pub fn context_id(&self) -> &ContextID {
        &self.0.context_id
    }

    pub fn type_info(&self) -> &VarInfo {
        &self.0.type_info
    }

    pub fn is_placeholder(&self) -> bool {
        self.0.is_input
    }

    /// Whether aliasing analysis cannot assume this cell is unique.
    pub fn is_maybe_aliased(&self) -> bool {
        !self.0.context_id.is_current() || self.is_placeholder()
    }

    /// The current home trace value, without recording a structural cell read.
    ///
    /// Used during invocation/capture to thread non-aliased cell values and to
    /// classify foreign cells. For maybe-aliased cells this is only a structural
    /// handle (its producer lives in another context or is a placeholder);
    /// ordering is carried by serialization edges, not this value.
    pub fn read_home(&self) -> Trace {
        self.0.home.read().unwrap().clone()
    }

    /// Resolve the trace for this cell in the active context.
    pub fn get(&self) -> Trace {
        // If we are in the current context,
        // return the actual trace.
        if self.0.context_id.is_current() && !self.is_placeholder() {
            return self.read_home();
        }
        Invocation::read_cell(self.clone())
    }

    /// Rebind a local slot, or record an ordered write to an external cell.
    pub fn set(&self, value: Trace) {
        assert_eq!(
            self.type_info(),
            value.info(),
            "cell assignment type mismatch"
        );
        if self.0.context_id.is_current() && !self.is_placeholder() {
            *self.0.home.write().unwrap() = value;
        } else {
            Invocation::write_cell(self.clone(), value);
        }
    }
}

impl Deref for CellTrace {
    type Target = TraceCell;

    fn deref(&self) -> &TraceCell {
        &self.0
    }
}

impl From<Arc<TraceCell>> for CellTrace {
    fn from(value: Arc<TraceCell>) -> Self {
        CellTrace(value)
    }
}

impl From<CellTrace> for Arc<TraceCell> {
    fn from(value: CellTrace) -> Self {
        value.0
    }
}

impl Debug for CellTrace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&*self.0, f)
    }
}

impl Debug for TraceCell {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut dbg = f.debug_struct("TraceCell");
        dbg.field("ctx", &self.context_id)
            .field("type_info", &self.type_info);
        if self.is_input {
            dbg.field("value", &"placeholder");
        } else {
            dbg.field("value", &self.home.read().unwrap());
        }
        dbg.finish()
    }
}

/// Typed handle to a [`TraceCell`] (analogous to [`Tracer`] for [`TraceData`]).
pub struct CellTracer<T: Traceable> {
    shared: CellTrace,
    _phantom: PhantomData<T>,
}

impl<T: Traceable> CellTracer<T> {
    /// Allocate an independent slot in the template's owning context, even
    /// when first requested by a nested trace. Copies the home initializer,
    /// not a read in the nested trace. Intended for immutable cell templates.
    pub fn independent(&self) -> Self {
        assert!(!self.shared.is_placeholder());
        Self {
            shared: CellTrace(Arc::new(TraceCell {
                context_id: self.shared.context_id().clone(),
                type_info: self.shared.type_info().clone(),
                home: RwLock::new(self.shared.read_home()),
                is_input: false,
            })),
            _phantom: PhantomData,
        }
    }
    pub fn new(trace: Tracer<T>) -> Self {
        CellTracer {
            shared: CellTrace::new(trace.trace_ref().clone()),
            _phantom: PhantomData,
        }
    }

    pub fn placeholder(info: VarInfo) -> Self {
        CellTracer {
            shared: CellTrace::placeholder(info),
            _phantom: PhantomData,
        }
    }

    pub fn cell_ref(&self) -> &CellTrace {
        &self.shared
    }

    pub fn context_id(&self) -> &ContextID {
        self.shared.context_id()
    }

    pub fn get(&self) -> Tracer<T> {
        unsafe { Tracer::unchecked_new(self.shared.get()) }
    }

    pub fn set(&self, value: Tracer<T>) {
        self.shared.set(value.trace_ref().clone());
    }

    pub fn try_cast<U: Traceable>(self) -> Result<CellTracer<U>, ()> {
        match U::Info::cast(self.shared.type_info()) {
            Some(_) => Ok(CellTracer {
                shared: self.shared,
                _phantom: PhantomData,
            }),
            None => Err(()),
        }
    }

    pub fn try_cast_ref<U: Traceable>(&self) -> Result<&CellTracer<U>, &Self> {
        match U::Info::cast(self.shared.type_info()) {
            Some(_) => Ok(unsafe { std::mem::transmute(self) }),
            None => Err(self),
        }
    }
}

// SAFETY: CellTracer<Generic> and CellTrace share the same address (CellTrace is field 0).
impl From<&CellTrace> for &CellTracer<Generic> {
    fn from(r: &CellTrace) -> Self {
        unsafe { std::mem::transmute(r) }
    }
}

impl<T: Traceable> From<CellTracer<T>> for CellTrace {
    fn from(c: CellTracer<T>) -> Self {
        c.shared
    }
}

impl From<CellTrace> for CellTracer<Generic> {
    fn from(c: CellTrace) -> Self {
        CellTracer {
            shared: c,
            _phantom: PhantomData,
        }
    }
}

impl<T: Traceable> Clone for CellTracer<T> {
    fn clone(&self) -> Self {
        CellTracer {
            shared: self.shared.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<T: Traceable> std::fmt::Debug for CellTracer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CellTracer")
            .field("shared", &self.shared)
            .finish()
    }
}

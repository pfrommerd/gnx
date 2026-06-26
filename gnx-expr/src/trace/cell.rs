use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::Deref;
use std::sync::{Arc, RwLock};

use crate::builtins::{READ_OP, UPDATE_OP};
use crate::trace::TraceInput;
use crate::{Effect, Generic, Invocation, Op, ValueInfo};

use super::context::{ContextID, TraceContext};
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
    /// The current "home" trace value, held under its own lock. Input cells
    /// (`is_input`) initialize this to a placeholder trace.
    home: RwLock<TraceRef>,
    /// Whether this cell is an abstract input/placeholder. Input cells are
    /// always treated as maybe-aliased (their relationship to other inputs is
    /// unknown at trace time).
    is_input: bool,
}

/// Shared handle to a mutable [`TraceCell`].
#[repr(transparent)]
#[derive(Clone)]
pub struct TraceCellRef(Arc<TraceCell>);

impl TraceCellRef {
    pub fn new(base: TraceRef) -> Self {
        TraceCellRef(Arc::new(TraceCell {
            context_id: TraceContext::current_id(),
            type_info: base.info().clone(),
            home: RwLock::new(base),
            is_input: false,
        }))
    }

    pub fn placeholder(info: ValueInfo) -> Self {
        TraceCellRef(Arc::new(TraceCell {
            context_id: TraceContext::current_id(),
            type_info: info.clone(),
            home: RwLock::new(TraceRef::placeholder(info)),
            is_input: true,
        }))
    }

    pub fn context_id(&self) -> &ContextID {
        &self.0.context_id
    }

    pub fn type_info(&self) -> &ValueInfo {
        &self.0.type_info
    }

    pub fn is_placeholder(&self) -> bool {
        self.0.is_input
    }

    /// Whether aliasing analysis cannot assume this cell is unique.
    pub fn is_maybe_aliased(&self) -> bool {
        !self.0.context_id.is_current() || self.is_placeholder()
    }

    /// The current home trace value, without emitting a [`READ_OP`].
    ///
    /// Used during invocation/capture to thread non-aliased cell values and to
    /// classify foreign cells. For maybe-aliased cells this is only a structural
    /// handle (its producer lives in another context or is a placeholder);
    /// ordering is carried by serialization edges, not this value.
    pub fn read_home(&self) -> TraceRef {
        self.0.home.read().unwrap().clone()
    }

    /// Resolve the trace for this cell in the active context.
    pub fn get(&self) -> TraceRef {
        // If we are in the current context,
        // return the actual trace.
        if self.0.context_id.is_current() && !self.is_placeholder() {
            return self.read_home();
        }
        // Otherwise, bind an invocation with this cell.
        let [trace] = Invocation::invoke(
            Op::new(&READ_OP),
            vec![],
            vec![self.clone()],
            vec![self.type_info().clone()],
        )
        .try_into()
        .unwrap();
        trace.into_trace_ref()
    }

    /// Bind the cell: update home storage in-context, otherwise via [`TraceContext`].
    pub fn set(&self, value: TraceRef) {
        if self.0.context_id.is_current() && !self.is_placeholder() {
            let mut guard = self.0.home.write().unwrap();
            *guard = value;
        } else {
            // Maybe-aliased cell: record the write as an UPDATE_OP invocation so
            // that aliasing serialization edges are established.
            Invocation::invoke_with_subexprs(
                Op::new(&UPDATE_OP),
                vec![],
                vec![TraceInput::Cell(self.clone(), Effect::Write)],
                vec![],
            );
            let _ = value;
        }
    }

    // Callback from an invocation to update the cell.
    pub(crate) fn direct_update_from_invoke(
        &self,
        invocation: &Arc<Invocation>,
        subexpr: Option<usize>,
        index: usize,
    ) {
        assert!(
            !self.is_maybe_aliased(),
            "aliased cells cannot be updated directly"
        );
        let mut guard = self.0.home.write().unwrap();
        // set the cell value to an updated trace.
        *guard =
            TraceRef::updated_cell(invocation.clone(), self.type_info().clone(), subexpr, index);
    }

    pub(crate) fn downgrade(&self) -> std::sync::Weak<TraceCell> {
        Arc::downgrade(&self.0)
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

    pub fn placeholder(info: ValueInfo) -> Self {
        TracerCell {
            shared: TraceCellRef::placeholder(info),
            _phantom: PhantomData,
        }
    }

    pub fn cell_ref(&self) -> &TraceCellRef {
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

    pub fn try_cast<U: Traceable>(self) -> Result<TracerCell<U>, ()> {
        match self.shared.type_info().downcast::<U::Info>() {
            Ok(_) => Ok(TracerCell {
                shared: self.shared,
                _phantom: PhantomData,
            }),
            Err(_) => Err(()),
        }
    }

    pub fn try_cast_ref<U: Traceable>(&self) -> Result<&TracerCell<U>, &Self> {
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
    fn from(c: TracerCell<T>) -> Self {
        c.shared
    }
}

impl From<TraceCellRef> for TracerCell<Generic> {
    fn from(c: TraceCellRef) -> Self {
        TracerCell {
            shared: c,
            _phantom: PhantomData,
        }
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
        f.debug_struct("TracerCell")
            .field("shared", &self.shared)
            .finish()
    }
}

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use super::cell::CellKey;
use super::TraceRef;

static NEXT_CONTEXT_ID: AtomicU64 = AtomicU64::new(1);

/// Identifies the [`TraceContext`] that created a traced value or cell.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct ContextID(u64);

impl ContextID {
    /// Root context used when no [`TraceContext`] is on the stack.
    pub const ROOT: ContextID = ContextID(0);

    fn fresh() -> Self {
        ContextID(NEXT_CONTEXT_ID.fetch_add(1, Ordering::Relaxed))
    }
}

#[derive(Debug)]
struct TraceContextState {
    overrides: Mutex<HashMap<CellKey, TraceRef>>,
    updates: Mutex<Vec<CellUpdate>>,
}

/// A recorded cross-context write to a [`TraceCell`](super::TraceCell).
#[derive(Debug, Clone)]
pub struct CellUpdate {
    pub cell: super::TraceCellRef,
    pub value: TraceRef,
}

/// Active tracing region: cell overrides and update log for capture.
#[derive(Clone, Debug)]
pub struct TraceContext {
    id: ContextID,
    state: Arc<TraceContextState>,
}

impl TraceContext {
    pub fn id(&self) -> ContextID {
        self.id
    }

    fn new() -> Self {
        TraceContext {
            id: ContextID::fresh(),
            state: Arc::new(TraceContextState {
                overrides: Mutex::new(HashMap::new()),
                updates: Mutex::new(Vec::new()),
            }),
        }
    }

    fn root() -> Self {
        ROOT_CONTEXT.with(|c| c.borrow().clone())
    }

    /// Push a new context onto the thread-local stack and return a guard that pops on drop.
    pub fn enter() -> TraceContextGuard {
        let ctx = TraceContext::new();
        CONTEXT_STACK.with(|stack| stack.borrow_mut().push(ctx.clone()));
        TraceContextGuard { ctx }
    }

    /// The innermost active context on the stack, if any.
    pub fn current() -> Option<TraceContext> {
        CONTEXT_STACK.with(|stack| stack.borrow().last().cloned())
    }

    /// Active context: stack top, or the root context when the stack is empty.
    pub fn active() -> TraceContext {
        Self::current().unwrap_or_else(TraceContext::root)
    }

    /// Id of the active context (never requires an explicit `enter()`).
    pub fn current_id() -> ContextID {
        Self::active().id
    }

    pub(crate) fn override_cell(&self, cell: CellKey, trace: TraceRef) {
        self.state
            .overrides
            .lock()
            .unwrap()
            .insert(cell, trace);
    }

    pub(crate) fn cell_override(&self, cell: CellKey) -> Option<TraceRef> {
        self.state.overrides.lock().unwrap().get(&cell).cloned()
    }

    pub(crate) fn register_update(&self, update: CellUpdate) {
        self.state.updates.lock().unwrap().push(update);
    }

    /// Cell updates recorded in this context (for capture).
    pub fn updates(&self) -> Vec<CellUpdate> {
        self.state.updates.lock().unwrap().clone()
    }
}

/// RAII guard: pops this context from the thread-local stack on drop.
pub struct TraceContextGuard {
    ctx: TraceContext,
}

impl TraceContextGuard {
    pub fn context(&self) -> &TraceContext {
        &self.ctx
    }
}

impl Drop for TraceContextGuard {
    fn drop(&mut self) {
        CONTEXT_STACK.with(|stack| {
            let mut s = stack.borrow_mut();
            if let Some(top) = s.last() {
                if top.id == self.ctx.id {
                    s.pop();
                }
            }
        });
    }
}

thread_local! {
    static CONTEXT_STACK: RefCell<Vec<TraceContext>> = RefCell::new(Vec::new());
    static ROOT_CONTEXT: RefCell<TraceContext> = RefCell::new(TraceContext {
        id: ContextID::ROOT,
        state: Arc::new(TraceContextState {
            overrides: Mutex::new(HashMap::new()),
            updates: Mutex::new(Vec::new()),
        }),
    });
}

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};

use super::TraceRef;
use super::cell::CellKey;

/// Identifies the [`TraceContext`] that created a traced value or cell.
/// Does not hold a strong reference to the context, so it can be dropped.
/// Is None for the root context.
#[derive(Clone, Debug)]
pub struct ContextID(Option<Weak<TraceContextState>>);

impl ContextID {
    pub const ROOT: ContextID = ContextID(None);
}

impl std::hash::Hash for ContextID {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0
            .as_ref()
            .map(|x| Weak::as_ptr(x) as usize)
            .hash(state);
    }
}

impl PartialEq for ContextID {
    fn eq(&self, other: &Self) -> bool {
        self.0.as_ref().map(|x| Weak::as_ptr(x) as usize)
            == other.0.as_ref().map(|x| Weak::as_ptr(x) as usize)
    }
}

impl Eq for ContextID {}

impl PartialOrd for ContextID {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        let ours = self.0.as_ref().map(|x| Weak::as_ptr(x) as usize);
        let theirs = other.0.as_ref().map(|x| Weak::as_ptr(x) as usize);
        ours.partial_cmp(&theirs)
    }
}

impl Ord for ContextID {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let ours = self.0.as_ref().map(|x| Weak::as_ptr(x) as usize);
        let theirs = other.0.as_ref().map(|x| Weak::as_ptr(x) as usize);
        ours.cmp(&theirs)
    }
}

impl From<&TraceContext> for ContextID {
    fn from(ctx: &TraceContext) -> Self {
        ContextID(Some(Arc::downgrade(&ctx.state)))
    }
}

impl ContextID {
    // If the context is not alive, a tracer has been leaked!
    pub fn is_alive(&self) -> bool {
        match &self.0 {
            None => true,
            Some(weak) => Weak::upgrade(weak).is_some(),
        }
    }
    // If this is the current active context
    pub fn is_current(&self) -> bool {
        CONTEXT_STACK.with(|stack| match stack.borrow().last() {
            // If there is a context, check if it is the same as this context
            Some(ctx) => Some(Arc::as_ptr(&ctx.state)) == self.0.as_ref().map(Weak::as_ptr),
            // If there is no context, check if this is the root context
            None => self.0.is_none(),
        })
    }
}

#[derive(Debug)]
struct TraceContextState {
    overrides: Mutex<HashMap<CellKey, TraceRef>>,
}

/// Active tracing region: cell overrides and update log for capture.
#[derive(Clone, Debug)]
pub struct TraceContext {
    state: Arc<TraceContextState>,
}

impl TraceContext {
    pub fn id(&self) -> ContextID {
        ContextID::from(self)
    }

    /// Push a new context onto the thread-local stack and return a guard that pops on drop.
    pub fn enter() -> TraceContextGuard {
        let ctx = TraceContext {
            state: Arc::new(TraceContextState {
                overrides: Mutex::new(HashMap::new()),
            }),
        };
        CONTEXT_STACK.with(|stack| stack.borrow_mut().push(ctx.clone()));
        TraceContextGuard { ctx }
    }

    /// The innermost active context on the stack, if any.
    pub fn current() -> Option<TraceContext> {
        CONTEXT_STACK.with(|stack| stack.borrow().last().cloned())
    }

    /// Id of the current context (or the root id if no context is on the stack).
    pub fn current_id() -> ContextID {
        CONTEXT_STACK.with(|stack| {
            stack
                .borrow()
                .last()
                .map(|ctx| ctx.id())
                .unwrap_or(ContextID::ROOT)
        })
    }

    pub(crate) fn override_cell(&self, cell: CellKey, trace: TraceRef) {
        self.state.overrides.lock().unwrap().insert(cell, trace);
    }

    // Gets the current override for a cell. Will return a placeholder
    // if no override is found.
    pub(crate) fn cell_override(&self, cell: CellKey) -> Option<TraceRef> {
        match self.state.overrides.lock().unwrap().get(&cell) {
            Some(trace) => Some(trace.clone()),
            None => None,
        }
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
                if Arc::as_ptr(&top.state) == Arc::as_ptr(&self.ctx.state) {
                    s.pop();
                }
            }
        });
    }
}

thread_local! {
    static CONTEXT_STACK: RefCell<Vec<TraceContext>> = RefCell::new(Vec::new());
    static ROOT_CONTEXT: RefCell<TraceContext> = RefCell::new(TraceContext {
        state: Arc::new(TraceContextState {
            overrides: Mutex::new(HashMap::new()),
        }),
    });
}

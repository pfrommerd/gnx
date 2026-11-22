use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use std::thread::ThreadId;

use super::Invocation;

/// Identifies a trace by its thread, per-thread sequence number, and stack offset.
/// The ID is only a token; the trace state itself remains thread-local.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContextID(Option<(ThreadId, usize, usize)>);

impl From<&TraceContext> for ContextID {
    fn from(ctx: &TraceContext) -> Self {
        Self(Some((
            ctx.state.thread_id,
            ctx.state.local_id,
            ctx.state.stack_offset,
        )))
    }
}

impl ContextID {
    pub const ROOT: Self = Self(None);

    pub fn is_alive(&self) -> bool {
        match self.0 {
            None => true,
            Some((thread, _, offset)) if thread == std::thread::current().id() => CONTEXT_STACK
                .with_borrow(|(_, stack)| stack.get(offset).is_some_and(|ctx| ctx.id() == *self)),
            Some(_) => false,
        }
    }

    pub fn is_current(&self) -> bool {
        *self == TraceContext::current_id()
    }
}

#[derive(Debug)]
struct TraceContextState {
    // The thread id and stack offset
    // at which this context was created.
    thread_id: ThreadId,
    symbol_scope: Rc<RefCell<crate::VarScope>>,
    symbols: RefCell<std::collections::BTreeSet<crate::Symbol>>,
    parent: Option<TraceContext>,
    // Counter within the thread for this context.
    // ThreadID and local_id together uniquely identify a context
    // and will not be reused (unlike Rc::as_ptr, which may be reused).
    local_id: usize,
    // The stack offset at which this context can be found.
    stack_offset: usize,
    // The most recent write into a maybe-aliased cell in this context.
    last_aliased_write: RefCell<Option<Arc<Invocation>>>,
    // All reads from maybe-aliased cells since `last_aliased_write`. A
    // subsequent aliased write must be ordered after all of them (WAR), while
    // the reads themselves may be reordered among each other (RAR).
    reads_since_write: RefCell<Vec<Arc<Invocation>>>,
}

/// Active tracing region tracking mutable-access ordering in the trace DAG.
#[derive(Clone, Debug)]
pub struct TraceContext {
    state: Rc<TraceContextState>,
}

impl TraceContext {
    /// Allocate a symbol in this tracing region. Active nested traces
    /// share an allocator to distinguish local and inherited symbols; there
    /// is no process-global symbol namespace.
    pub fn create_symbol(&self) -> crate::Symbol {
        let s = self.state.symbol_scope.borrow_mut().create_symbol();
        self.state.symbols.borrow_mut().insert(s);
        s
    }
    pub(super) fn owns_symbol(&self, s: crate::Symbol) -> bool {
        self.state.symbols.borrow().contains(&s)
    }
    /// Whether `s` was allocated in this region or an enclosing one.
    pub fn has_symbol(&self, s: crate::Symbol) -> bool {
        self.owns_symbol(s) || self.state.parent.as_ref().is_some_and(|p| p.has_symbol(s))
    }
    pub fn id(&self) -> ContextID {
        ContextID::from(self)
    }

    /// Push a new context onto the thread-local stack and return a guard that pops on drop.
    pub fn enter() -> TraceContextGuard {
        CONTEXT_STACK.with_borrow_mut(|(id, stack)| {
            let thread_id = std::thread::current().id();
            let local_id = *id;
            *id += 1;
            let stack_offset = stack.len();
            let ctx = TraceContext {
                state: Rc::new(TraceContextState {
                    thread_id,
                    symbol_scope: stack
                        .last()
                        .map(|p| p.state.symbol_scope.clone())
                        .unwrap_or_default(),
                    symbols: Default::default(),
                    parent: stack.last().cloned(),
                    local_id,
                    stack_offset,
                    last_aliased_write: RefCell::new(None),
                    reads_since_write: RefCell::new(Vec::new()),
                }),
            };
            stack.push(ctx.clone());
            TraceContextGuard { ctx }
        })
    }

    /// The innermost active context on the stack, if any.
    pub fn current() -> Option<TraceContext> {
        CONTEXT_STACK.with_borrow(|(_, stack)| stack.last().cloned())
    }

    /// Id of the current context (or the root id if no context is on the stack).
    pub fn current_id() -> ContextID {
        CONTEXT_STACK
            .with_borrow(|(_, stack)| stack.last().map(|ctx| ctx.id()).unwrap_or(ContextID::ROOT))
    }

    /// The most recent write into a maybe-aliased cell, if any.
    pub fn last_aliased_write(&self) -> Option<Arc<Invocation>> {
        self.state.last_aliased_write.borrow().clone()
    }

    /// The serialization parents a new maybe-aliased access must be ordered
    /// after. A write depends on every read since the last write plus the last
    /// write itself (WAR + WAW); a read depends only on the last write (RAW).
    pub fn aliasing_parents(&self, write: bool) -> Vec<Arc<Invocation>> {
        let mut parents = Vec::new();
        if write {
            parents.extend(self.state.reads_since_write.borrow().iter().cloned());
        }
        if let Some(w) = self.state.last_aliased_write.borrow().clone() {
            parents.push(w);
        }
        parents
    }

    /// Record that `inv` read from maybe-aliased cells (and did not write).
    pub fn note_aliased_read(&self, inv: Arc<Invocation>) {
        self.state.reads_since_write.borrow_mut().push(inv);
    }

    /// Record that `inv` wrote into maybe-aliased cells, becoming the new
    /// ordering anchor and clearing the pending reads.
    pub fn note_aliased_write(&self, inv: Arc<Invocation>) {
        *self.state.last_aliased_write.borrow_mut() = Some(inv);
        self.state.reads_since_write.borrow_mut().clear();
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
        CONTEXT_STACK.with_borrow_mut(|(_, stack)| {
            if let Some(top) = stack.last() {
                if Rc::as_ptr(&top.state) == Rc::as_ptr(&self.ctx.state) {
                    stack.pop();
                }
            }
        });
    }
}

thread_local! {
    // contains the local id counter and the stack of contexts
    static CONTEXT_STACK: RefCell<(usize, Vec<TraceContext>)> = RefCell::new((0, Vec::new()));
}

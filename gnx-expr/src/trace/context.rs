use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use std::thread::ThreadId;

use super::Invocation;

/// Identifies a [`TraceContext`] by its thread id,
/// the local counter number for that thread,
/// stack offset, and the address of the associated Rc
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContextID(Option<(ThreadId, usize, usize, *const TraceContextState)>);

impl ContextID {
    pub const ROOT: ContextID = ContextID(None);

    pub(crate) fn new(
        thread_id: ThreadId,
        local_id: usize,
        stack_offset: usize,
        arc: &Rc<TraceContextState>,
    ) -> Self {
        ContextID(Some((thread_id, local_id, stack_offset, Rc::as_ptr(arc))))
    }

    pub(crate) fn thread_id(&self) -> Option<ThreadId> {
        self.0.as_ref().map(|x| x.0)
    }

    pub(crate) fn local_id(&self) -> Option<usize> {
        self.0.as_ref().map(|x| x.1)
    }

    pub(crate) fn stack_offset(&self) -> Option<usize> {
        self.0.as_ref().map(|x| x.2)
    }

    pub(crate) fn address(&self) -> Option<*const TraceContextState> {
        self.0.as_ref().map(|x| x.3)
    }
}

impl From<&TraceContext> for ContextID {
    fn from(ctx: &TraceContext) -> Self {
        ContextID::new(
            ctx.state.thread_id,
            ctx.state.local_id,
            ctx.state.stack_offset,
            &ctx.state,
        )
    }
}

impl ContextID {
    // If the context is not alive, a tracer has been leaked!
    pub fn is_alive(&self) -> bool {
        match &self.0 {
            None => true,
            Some((_, _, stack_offset, rc_ptr)) => CONTEXT_STACK.with_borrow(|(_, stack)| {
                stack.get(*stack_offset).map(|x| Rc::as_ptr(&x.state)) == Some(*rc_ptr)
            }),
        }
    }
    // If this is the current active context
    pub fn is_current(&self) -> bool {
        CONTEXT_STACK.with_borrow(|(_, stack)| match stack.last() {
            // If there is a context, check if it is the same as this context
            Some(ctx) => Some(Rc::as_ptr(&ctx.state)) == self.address(),
            // If there is no context, check if this is the root context
            None => self.0.is_none(),
        })
    }
}

#[derive(Debug)]
struct TraceContextState {
    // The thread id and stack offset
    // at which this context was created.
    thread_id: ThreadId,
    // Counter within the thread for this context.
    // ThreadID and local_id together uniquely identify a context
    // and will not be reused (unlike Rc::as_ptr, which may be reused).
    local_id: usize,
    // The stack offset at which this context can be found.
    stack_offset: usize,
    // Contains the previous aliased write
    // which took place in this context.
    last_aliased_write: RefCell<Option<Arc<Invocation>>>,
    // Contains the previous aliased read
    // which took place in this context.
    last_aliased_read: RefCell<Option<Arc<Invocation>>>,
}

/// Active tracing region: cell overrides and update log for capture.
#[derive(Clone, Debug)]
pub struct TraceContext {
    state: Rc<TraceContextState>,
}

impl TraceContext {
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
                    local_id,
                    stack_offset,
                    last_aliased_write: RefCell::new(None),
                    last_aliased_read: RefCell::new(None),
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

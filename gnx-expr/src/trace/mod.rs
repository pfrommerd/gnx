mod capture;
mod cell;
mod context;
mod trace_ref;

pub use capture::{Capture, TracerKey};
pub use cell::{CellKey, TraceCell, TraceCellRef, TracerCell};
pub use context::{ContextID, TraceContext, TraceContextGuard};
pub use trace_ref::{Trace, TraceRef, Tracer};

use std::fmt::Debug;
use std::sync::{Arc, OnceLock, Weak};

use crate::{Effect, Expr, Op};

pub use super::value::{ConcreteValue, Generic, Traceable, Value, ValueInfo};

/// A traced value referenced by an [`Invocation`]: immutable trace or mutable cell.
#[derive(Clone, Debug)]
pub enum TraceObject {
    Ref(TraceRef),
    Cell(TraceCellRef),
}

impl TraceObject {
    /// The trace value backing this object: the ref itself, or a cell's current
    /// home value.
    pub fn resolve(&self) -> TraceRef {
        match self {
            TraceObject::Ref(r) => r.clone(),
            TraceObject::Cell(c) => c.read_home(),
        }
    }
}

impl<T: Traceable> From<Tracer<T>> for TraceObject {
    fn from(t: Tracer<T>) -> Self {
        TraceObject::Ref(t.into_trace_ref())
    }
}

impl<T: Traceable> From<TracerCell<T>> for TraceObject {
    fn from(c: TracerCell<T>) -> Self {
        TraceObject::Cell(c.into())
    }
}

impl From<TraceRef> for TraceObject {
    fn from(r: TraceRef) -> Self {
        TraceObject::Ref(r)
    }
}

impl From<TraceCellRef> for TraceObject {
    fn from(c: TraceCellRef) -> Self {
        TraceObject::Cell(c)
    }
}

// An input type for an invocation.
// Either a TraceRef or a TraceCellRef with an effect
// (Unusued, Read, Write, ReadWrite).
// The effect specified is used to get() or set() the cell with the appropriate TraceRef.
#[derive(Clone)]
pub enum TraceInput {
    Ref(TraceRef),
    Cell(TraceCellRef, Effect),
}

impl From<TraceObject> for TraceInput {
    fn from(value: TraceObject) -> Self {
        match value {
            TraceObject::Ref(r) => TraceInput::Ref(r),
            TraceObject::Cell(c) => TraceInput::Cell(c, Effect::Read),
        }
    }
}

impl From<TraceRef> for TraceInput {
    fn from(value: TraceRef) -> Self {
        TraceInput::Ref(value)
    }
}

impl<T: Traceable> From<Tracer<T>> for TraceInput {
    fn from(value: Tracer<T>) -> Self {
        TraceInput::Ref(value.into_trace_ref())
    }
}

impl From<TraceCellRef> for TraceInput {
    fn from(value: TraceCellRef) -> Self {
        TraceInput::Cell(value, Effect::Read)
    }
}

impl<T: Traceable> From<TracerCell<T>> for TraceInput {
    fn from(value: TracerCell<T>) -> Self {
        TraceInput::Cell(value.into(), Effect::Read)
    }
}

// The output *type* for the invocation.
pub enum TraceOutput {
    Ref(ValueInfo),
    NewCell(ValueInfo),
    /// The returned value is just an alias to an input.
    InputCell(usize),
    /// The returned value is an alias to a closure value.
    ClosureCell {
        subexpr: usize,
        idx: usize,
    },
}

#[derive(Debug)]
pub enum InvParent {
    Ref(TraceRef),
    // If the cell is read from,
    // we include the TraceRef that was read.
    Cell(TraceCellRef, Option<TraceRef>),
}

impl InvParent {
    pub fn trace(&self) -> Option<&TraceRef> {
        match self {
            InvParent::Ref(r) => Some(r),
            InvParent::Cell(_, Some(r)) => Some(r),
            InvParent::Cell(_, None) => None,
        }
    }
}

// Children of an invocation are weak references
// to the outputs of the invocation.
#[derive(Debug)]
pub enum InvChild {
    Ref(Weak<Trace>),
    NewCell(Weak<TraceCell>),
    InputCell {
        idx: usize,
        cell: Weak<TraceCell>,
    },
    ClosureCell {
        subexpr: usize,
        idx: usize,
        cell: Weak<TraceCell>,
    },
}

pub struct Invocation {
    // The "owning" context for this invocation.
    context_id: ContextID,
    op: Op,
    /// Serialization edges for maybe-aliased cell access. When this invocation
    /// reads from or writes into potentially-aliased cells, these are the prior
    /// invocations it must be ordered after (see [`TraceContext::aliasing_parents`]).
    /// A read depends on the last aliased write (RAW); a write additionally
    /// depends on every read since that write (WAR + WAW). Empty when this
    /// invocation touches no maybe-aliased cells.
    aliasing_parents: Vec<Arc<Invocation>>,
    // Subexpressions and the associated closure variables.
    subexprs: Vec<(Arc<Expr>, Vec<InvParent>)>,
    inputs: Vec<InvParent>,
    // To avoid locking, the outputs are stored in an UnsafeCell.
    // This is safe to access post-construction since the outputs are initialized within the invoke() method.
    outputs: OnceLock<Vec<InvChild>>,
}

impl Debug for Invocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Invocation")
            .field("op", &self.op)
            .field("ctx", &self.context_id)
            .field("inputs", &self.inputs)
            .finish()
    }
}

impl Invocation {
    pub fn op(&self) -> &Op {
        &self.op
    }

    pub fn context_id(&self) -> &ContextID {
        &self.context_id
    }

    pub fn subexprs(&self) -> &[(Arc<Expr>, Vec<InvParent>)] {
        &self.subexprs
    }

    pub fn inputs(&self) -> &[InvParent] {
        &self.inputs
    }

    pub fn aliasing_parents(&self) -> &[Arc<Invocation>] {
        &self.aliasing_parents
    }

    pub fn outputs(&self) -> &[InvChild] {
        let outputs = self.outputs.get().unwrap();
        outputs
    }

    pub fn invoke<I>(
        op: Op,
        subexprs: Vec<(Arc<Expr>, Vec<TraceInput>)>,
        inputs: Vec<I>,
        outputs: Vec<ValueInfo>,
    ) -> Vec<Tracer<Generic>>
    where
        I: Into<TraceInput>,
    {
        Self::invoke_with_subexprs(
            op,
            subexprs,
            inputs.into_iter().map(Into::into).collect(),
            outputs.into_iter().map(TraceOutput::Ref).collect(),
        )
        .into_iter()
        .map(|object| match object {
            TraceObject::Ref(r) => Tracer::from(r),
            TraceObject::Cell(_) => panic!("invoke_refs produced a cell"),
        })
        .collect()
    }

    pub fn invoke_with_subexprs(
        op: Op,
        subexprs: Vec<(Arc<Expr>, Vec<TraceInput>)>,
        inputs: Vec<TraceInput>,
        outputs: Vec<TraceOutput>,
    ) -> Vec<TraceObject> {
        let context_id = TraceContext::current_id();
        // Non-aliased cells written by this invocation, recorded so we can
        // thread their value (SSA) after the invocation is constructed.
        // Each entry is (cell, subexpr index (None if a direct input), position).
        let mut nonaliased_writes: Vec<(TraceCellRef, Option<usize>, usize)> = Vec::new();
        // Whether this invocation reads from / writes into any maybe-aliased cells.
        let mut read_aliased = false;
        let mut write_aliased = false;
        fn handle_input(
            input: TraceInput,
            subexpr: Option<usize>,
            index: usize,
            nonaliased_writes: &mut Vec<(TraceCellRef, Option<usize>, usize)>,
            read_aliased: &mut bool,
            write_aliased: &mut bool,
        ) -> InvParent {
            match input {
                TraceInput::Ref(r) => InvParent::Ref(r),
                TraceInput::Cell(c, e) => {
                    let aliased = c.is_maybe_aliased();
                    match e {
                        Effect::Unused => InvParent::Cell(c, None),
                        Effect::Read => {
                            if aliased {
                                *read_aliased = true;
                            }
                            let read = c.read_home();
                            InvParent::Cell(c, Some(read))
                        }
                        Effect::Write => {
                            if aliased {
                                *write_aliased = true;
                            } else {
                                nonaliased_writes.push((c.clone(), subexpr, index));
                            }
                            InvParent::Cell(c, None)
                        }
                        Effect::ReadWrite => {
                            if aliased {
                                *read_aliased = true;
                                *write_aliased = true;
                            } else {
                                nonaliased_writes.push((c.clone(), subexpr, index));
                            }
                            let read = c.read_home();
                            InvParent::Cell(c, Some(read))
                        }
                    }
                }
            }
        }
        // Convert all inputs to InvParent types, which carry the read traces for
        // cells that are read from or written to.
        let inputs = inputs
            .into_iter()
            .enumerate()
            .map(|(index, input)| {
                handle_input(
                    input,
                    None,
                    index,
                    &mut nonaliased_writes,
                    &mut read_aliased,
                    &mut write_aliased,
                )
            })
            .collect();
        let subexprs = subexprs
            .into_iter()
            .enumerate()
            .map(|(sub_idx, (expr, closure))| {
                (
                    expr,
                    closure
                        .into_iter()
                        .enumerate()
                        .map(|(index, input)| {
                            handle_input(
                                input,
                                Some(sub_idx),
                                index,
                                &mut nonaliased_writes,
                                &mut read_aliased,
                                &mut write_aliased,
                            )
                        })
                        .collect(),
                )
            })
            .collect();
        // Serialization parents for maybe-aliased access (empty otherwise).
        let aliasing_parents = if read_aliased || write_aliased {
            TraceContext::current()
                .map(|ctx| ctx.aliasing_parents(write_aliased))
                .unwrap_or_default()
        } else {
            Vec::new()
        };
        let invocation = Arc::new(Invocation {
            op,
            subexprs,
            context_id,
            aliasing_parents,
            inputs,
            outputs: OnceLock::new(),
        });
        let mut values = Vec::new();
        let mut children = Vec::new();
        for (idx, output) in outputs.into_iter().enumerate() {
            match output {
                TraceOutput::Ref(info) => {
                    let trace = TraceRef::returned(invocation.clone(), idx, info);
                    children.push(InvChild::Ref(Arc::downgrade(&trace.0)));
                    values.push(TraceObject::Ref(trace));
                }
                TraceOutput::NewCell(info) => {
                    let trace =
                        TraceCellRef::new(TraceRef::returned(invocation.clone(), idx, info));
                    children.push(InvChild::NewCell(trace.downgrade()));
                    values.push(TraceObject::Cell(trace));
                }
                TraceOutput::InputCell(idx) => {
                    let cell = match invocation.inputs.get(idx) {
                        Some(InvParent::Cell(cell, _)) => cell.clone(),
                        _ => panic!("TraceOutput::InputCell referenced a non-cell input"),
                    };
                    children.push(InvChild::InputCell {
                        idx,
                        cell: cell.downgrade(),
                    });
                    values.push(TraceObject::Cell(cell));
                }
                TraceOutput::ClosureCell { subexpr, idx } => {
                    let cell = match invocation
                        .subexprs
                        .get(subexpr)
                        .and_then(|(_, c)| c.get(idx))
                    {
                        Some(InvParent::Cell(cell, _)) => cell.clone(),
                        _ => panic!("TraceOutput::ClosureCell referenced a non-cell closure"),
                    };
                    children.push(InvChild::ClosureCell {
                        subexpr,
                        idx,
                        cell: cell.downgrade(),
                    });
                    values.push(TraceObject::Cell(cell));
                }
            }
        }
        invocation
            .outputs
            .set(children)
            .expect("new invocation outputs are set once");

        // Thread the value of non-aliased written cells (SSA): each picks up a
        // fresh UpdatedCellState trace pointing at this invocation.
        for (cell, subexpr, index) in nonaliased_writes {
            cell.direct_update_from_invoke(&invocation, subexpr, index);
        }

        // Update the maybe-aliased scheduler state so subsequent accesses are
        // ordered relative to this one.
        if read_aliased || write_aliased {
            if let Some(ctx) = TraceContext::current() {
                if write_aliased {
                    ctx.note_aliased_write(invocation.clone());
                } else {
                    ctx.note_aliased_read(invocation.clone());
                }
            }
        }

        values
    }
}

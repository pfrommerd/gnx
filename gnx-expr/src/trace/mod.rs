mod capture;
mod cell;
mod context;
mod trace_ref;

pub use capture::{Capture, TracerKey};
pub use cell::{CellKey, TraceCell, TraceCellRef, TracerCell};
pub use context::{ContextID, TraceContext, TraceContextGuard};
pub use trace_ref::{Trace, TraceRef, Tracer};

use std::collections::BTreeMap;
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
    /// If we are writing into or reading from potentially
    /// aliased cells, this will be set to the "parent" invocation.
    /// This is used to enforce ordering of aliased writes and reads.
    /// (If we write to aliased cells, this is set to Context.last_aliased_read,
    ///  and if we read from any aliased cells, this is set to Context.last_aliased_write)
    aliasing_parent: Option<Arc<Invocation>>,
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

    pub fn aliasing_parent(&self) -> Option<&Arc<Invocation>> {
        self.aliasing_parent.as_ref()
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
        // Unaliased cells that are updated by this invocation.
        // contains the cell, subexpr index (none if input), and index
        let mut updated_cells: BTreeSet<TraceCellRef> = BTreeSet::new();
        // Whether any aliased inputs were read or writen.
        let mut read_aliased = false;
        let mut write_aliased = false;
        fn handle_input(
            input: TraceInput,
            mutated_cells: &mut Vec<TraceCellRef>,
            read_aliased: &mut bool,
            write_aliased: &mut bool,
        ) -> InvParent {
            match input {
                TraceInput::Ref(r) => InvParent::Ref(r),
                TraceInput::Cell(c, e) => {
                    if !c.context_id().is_current() {
                        *foreign_cells = true;
                    }
                    match e {
                        Effect::Unused => InvParent::Cell(c, None),
                        Effect::Read => {
                            let read = c.read_home();
                            InvParent::Cell(c, Some(read))
                        }
                        Effect::Write => {
                            mutated_cells.push(c.clone());
                            InvParent::Cell(c, None)
                        }
                        Effect::ReadWrite => {
                            mutated_cells.push(c.clone());
                            let read = c.read_home();
                            InvParent::Cell(c, Some(read))
                        }
                    }
                }
            }
        }
        // Convert all inputs to TraceParent types, which include
        // the parent refs for cells that are read from or written to.
        let inputs = inputs
            .into_iter()
            .map(|input| {
                handle_input(
                    input,
                    &mut updated_cells,
                    &mut read_aliased,
                    &mut write_aliased,
                )
            })
            .collect();
        let subexprs = subexprs
            .into_iter()
            .map(|(expr, closure)| {
                (
                    expr,
                    closure
                        .into_iter()
                        .map(|input| handle_input(input, &mut mutated_cells, &mut foreign_cells))
                        .collect(),
                )
            })
            .collect();
        let parent_write = if foreign_cells {
            TraceContext::current().and_then(|ctx| ctx.last_global_write())
        } else {
            None
        };
        let invocation = Arc::new(Invocation {
            op,
            subexprs,
            context_id,
            inputs,
            parent_write,
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

        let had_writes = !mutated_cells.is_empty();
        for cell in mutated_cells {
            let value = TraceRef::updated(invocation.clone(), cell.type_info().clone());
            cell.set_from_invocation(value);
        }

        if had_writes {
            if let Some(ctx) = TraceContext::current() {
                ctx.register_global_write(invocation.clone());
            }
        }

        values
    }
}

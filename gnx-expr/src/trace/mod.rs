mod capture;
mod cell;
mod context;
mod trace_ref;

pub use capture::{Capture, TracerKey};
pub use cell::{CellKey, TraceCell, TraceCellRef, TracerCell};
pub use context::{ContextID, TraceContext, TraceContextGuard};
pub use trace_ref::{Trace, TraceRef, Tracer};

use std::cell::UnsafeCell;
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

/// Closure or input lists for [`Invocation::invoke`] and [`crate::Capture::from_context`].
pub type TraceOperands = Vec<TraceObject>;

impl TraceObject {
    /// Effective trace value for dependency walking and context checks.
    pub fn resolve(&self) -> TraceRef {
        match self {
            TraceObject::Ref(r) => r.clone(),
            TraceObject::Cell(c) => c.get(),
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
// The effect specified is used to get() or set()
// the cell with the appropriate TraceRef.
pub enum TraceInput {
    Ref(TraceRef),
    Cell(TraceCellRef, Effect),
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
    context_id: ContextID,
    op: Op,
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

    pub fn outputs(&self) -> &[InvChild] {
        let outputs = self.outputs.get().unwrap();
        outputs
    }

    pub fn invoke(op: Op, inputs: Vec<TraceInput>, outputs: Vec<TraceOutput>) -> Vec<TraceObject> {
        Self::invoke_with_subexprs(op, Vec::new(), inputs, outputs)
    }

    pub fn invoke_with_subexprs(
        op: Op,
        subexprs: Vec<(Arc<Expr>, Vec<TraceInput>)>,
        inputs: Vec<TraceInput>,
        outputs: Vec<TraceOutput>,
    ) -> Vec<TraceObject> {
        // Cells that are updated by this invocation.
        let context_id = TraceContext::current_id();
        let mut mutated_cells = Vec::new();
        let mut handle_input = |input: TraceInput| match input {
            TraceInput::Ref(r) => InvParent::Ref(r),
            TraceInput::Cell(c, e) => match e {
                Effect::Unused => InvParent::Cell(c, None),
                Effect::Read => InvParent::Cell(c, Some(c.get())),
                Effect::Write => {
                    mutated_cells.push(c.clone());
                    InvParent::Cell(c, None)
                }
                Effect::ReadWrite => {
                    mutated_cells.push(c.clone());
                    InvParent::Cell(c, Some(c.get()))
                }
            },
        };
        // Convert all inputs to TraceParent types, which include
        // the parent refs for cells that are read from or written to.
        let inputs = inputs.into_iter().map(handle_input).collect();
        let subexprs = subexprs
            .into_iter()
            .map(|(expr, closure)| (expr, closure.into_iter().map(handle_input).collect()))
            .collect();
        let invocation = Arc::new(Invocation {
            op,
            subexprs,
            context_id,
            inputs,
            outputs: OnceLock::new(),
        });
        let handle_output = |(idx, output): (usize, TraceOutput)| -> (TraceObject, InvChild) match output {
            TraceOutput::Ref(info) => {
                let trace = TraceRef::returned(invocation.clone(), idx, info);
                (TraceObject::Ref(trace), InvChild::Ref(Arc::downgrade(&trace.0)))
            },
            TraceOutput::NewCell(info) => {
                let trace = TraceCellRef::new(TraceRef::returned(invocation.clone(), idx, info));
                (TraceObject::Cell(trace), InvChild::NewCell(Arc::downgrade(&trace.0)))
            },
            TraceOutput::InputCell(idx) => InvChild::InputCell {
                idx,
                cell: Weak::new(),
            },
            TraceOutput::ClosureCell { subexpr, idx } => InvChild::ClosureCell {
                subexpr,
                idx,
                cell: Weak::new(),
            },
        };
        let (values, outputs) = outputs.into_iter().map(handle_output).unzip();
        // For any mutated input/closure cells, set them
        // to a new tracer that depends on the invocation.
        for cell in mutated_cells {
            cell.set(TraceRef::updated(
                invocation.clone(),
                cell.type_info().clone(),
                context_id,
            ));
        }
        // Construct output objects.
        let outputs = outputs
            .into_iter()
            .enumerate()
            .map(|(idx, output)| match output {
                TraceOutput::Ref(info) => {
                    TraceObject::Ref(TraceRef::returned(invocation.clone(), idx, info))
                }
                TraceOutput::NewCell(info) => TraceObject::Cell(TraceCellRef::new(
                    TraceRef::returned(invocation.clone(), idx, info, context_id),
                )),
            })
            .collect();

        // Return the outputs as tracers.
        outputs
            .into_iter()
            .enumerate()
            .map(|(index, info)| {
                Tracer::from(TraceRef::produced(
                    abstract_,
                    invocation.clone(),
                    Position::Return(index),
                    info,
                    context_id,
                ))
            })
            .collect()
    }
}

use std::fmt::Display;
use std::any::Any;
use std::hash::{Hash, Hasher};

use std::sync::Arc;

use super::util::{DynHash, DynEq};

pub trait Op: Display + DynHash + DynEq + Any + 'static {

}

// dyn Op implements PartialEq, Eq, and Hash
impl PartialEq for dyn Op {
    fn eq(&self, other: &dyn Op) -> bool { self.dyn_eq(other) }
}
impl Eq for dyn Op {}
impl Hash for dyn Op {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.dyn_hash(state)
    }
}

#[derive(Clone, Hash, PartialEq, Eq)]
pub struct Var {
    id: Option<usize>, // If None, var is a "hole"
}

#[derive(Clone, Hash)]
pub struct Eqn {
    op: Arc<dyn Op>,
    // closed-over inputs
    closure: Vec<Var>,
    inputs: Vec<Var>,
    outputs: Vec<Var>,
}

// TODO: Why can't we derive PartialEq?
impl PartialEq for Eqn {
    fn eq(&self, other: &Self) -> bool {
        self.op.dyn_eq(&other.op) &&
        self.inputs == other.inputs &&
        self.outputs == other.outputs
    }
}
impl Eq for Eqn {}

impl Eqn {
    pub fn new<O: Op>(op: O,
        closure: Vec<Var>,
        inputs: Vec<Var>,
        outputs: Vec<Var>
    ) -> Self {
        Eqn { op: Arc::new(op), closure, inputs, outputs }
    }
    pub fn op(&self) -> &dyn Op { &*self.op }
    pub fn inputs(&self) -> &Vec<Var> { &self.inputs }
    pub fn outputs(&self) -> &Vec<Var> { &self.outputs }
}

// Tracers can be "captured" into exprs
pub struct Expr {
    // any closed-over inputs
    closure: Vec<Var>,
    // explicit inputs
    inputs: Vec<Var>,
    eqns: Vec<Eqn>,
    outputs: Vec<Var>,
}

use crate::trace::{Tracer, Generic};

pub struct Capture {
    expr: Expr,
    // the tracers that were closed-over by the expr
    closure: Vec<Tracer<Generic>>
}
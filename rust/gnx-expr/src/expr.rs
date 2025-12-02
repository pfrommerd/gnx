use std::fmt::Display;
use std::any::Any;
use std::hash::{Hash, Hasher};

use std::sync::Arc;
use std::collections::BTreeMap;
use std::borrow::Cow;

use crate::ValueInfo;
use crate::array::{Item, Data};

pub enum Attrs {
    Scalar(Item),
    Data(Data<'static>),
    String(Cow<'static, str>),
    Info(ValueInfo),
    Expr(Expr),
    List(Vec<Attrs>),
    Map(BTreeMap<Cow<'static, str>, Attrs>),
}

pub struct Op {
    pub dialect: &'static str,
    pub name: &'static str,
    pub attrs: Attrs
}

#[derive(Clone, Hash, PartialEq, Eq)]
pub struct Var {
    id: Option<usize>, // If None, var is a "hole"
}

#[derive(Clone, Hash)]
pub struct Eqn {
    op: Op,
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
    pub fn new(op: Op,
        closure: Vec<Var>,
        inputs: Vec<Var>,
        outputs: Vec<Var>
    ) -> Self {
        Eqn { op: Arc::new(op), closure, inputs, outputs }
    }
    pub fn op(&self) -> &Op { &self.op }
    pub fn inputs(&self) -> &Vec<Var> { &self.inputs }
    pub fn outputs(&self) -> &Vec<Var> { &self.outputs }
}

// Tracers can be "captured" into exprs
pub struct Expr {
    // any closed-over inputs
    // note that these are distinct vars from 
    // the closure vars in the eqns
    captured_inputs: Vec<Var>,
    explicit_inputs: Vec<Var>,
    eqns: Vec<Eqn>,
    outputs: Vec<Var>,
}

// A closed expression is an expression
// with captured_inputs bound.
pub struct ClosedExpr<'r> {
    expr: Expr,
    captured_inputs: Cow<'r, [Var]>
}

use crate::trace::{Tracer, Generic};

pub struct Capture {
    expr: Expr,
    // the tracers that were closed-over by the expr
    closure: Vec<Tracer<Generic>>
}
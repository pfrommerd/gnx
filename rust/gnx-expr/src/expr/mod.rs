mod attr;

pub use attr::*;

use std::borrow::Cow;
use std::ops::Deref;
use std::sync::Arc;
use crate::trace::{Tracer, Generic};

#[derive(Clone, Hash, PartialEq, Eq)]
pub enum OpString {
    Static(&'static str),
    Shared(Arc<String>)
}

impl Deref for OpString {
    type Target = str;
    fn deref(&self) -> &Self::Target {
        match self {
            OpString::Static(s) => s,
            OpString::Shared(s) => s.as_str(),
        }
    }
}

#[derive(Clone, Hash, PartialEq, Eq)]
pub struct Op {
    pub dialect: OpString,
    pub name: OpString,
    pub attrs: AttrMap
}

#[derive(Clone, Hash, PartialEq, Eq)]
pub struct Var {
    id: Option<usize>, // If None, var is a "hole"
}

#[derive(Clone, Hash, PartialEq, Eq)]
pub struct Eqn {
    op: Op,
    // closed-over inputs
    closure: Vec<Var>,
    inputs: Vec<Var>,
    outputs: Vec<Var>,
}

impl Eqn {
    pub fn new(op: Op,
        closure: Vec<Var>,
        inputs: Vec<Var>,
        outputs: Vec<Var>
    ) -> Self {
        Eqn { op, closure, inputs, outputs }
    }
    pub fn op(&self) -> &Op { &self.op }
    pub fn inputs(&self) -> &Vec<Var> { &self.inputs }
    pub fn outputs(&self) -> &Vec<Var> { &self.outputs }
}

// Tracers can be "captured" into exprs
#[derive(Clone, Hash, PartialEq, Eq)]
pub struct Expr {
    // any closed-over inputs
    // note that these are distinct vars from 
    // the closure vars in the eqns
    closure_inputs: Vec<Var>,
    explicit_inputs: Vec<Var>,
    eqns: Vec<Eqn>,
    outputs: Vec<Var>,
}

// A closed expression is an expression
// with captured_inputs bound.
// This is used for pretty-printing Expr
// using the closure vars for the pretty-printing.
pub struct ClosedExpr {
    expr: Expr,
    // The vars that were closed-over by the expr
    closure: Cow<'static, [Var]>
}

// Tools for converting trace -> expr

pub struct Capture {
    expr: Expr,
    // the tracers that were closed-over by the expr
    closure: Vec<Tracer<Generic>>
}

impl Capture {
    fn from_outputs<'a>(outputs: impl IntoIterator<Item=&'a Tracer<Generic>>) -> Self {
        todo!()
    }
}
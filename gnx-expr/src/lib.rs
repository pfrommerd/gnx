mod attr;
pub mod builtins;
mod dialects;
pub mod hexpr;
pub mod lexpr;
pub mod lower;
// pub mod expr_macro;
// pub mod pretty;
pub mod trace;
pub mod util;
pub mod value;

pub use attr::*;
pub use dialects::*;
pub use hexpr::*;
pub use lower::{InputId, InputSpec, LoweredExpr, LoweringError};
pub use trace::*;
pub use util::expr_macro;

use std::cmp::{Eq, PartialEq};
use std::cmp::{Ord, PartialOrd};
use std::fmt;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::util::pretty;

// The inventory crate is used to statically register all dialects.
#[doc(hidden)]
pub mod __private {
    pub use inventory;
}

/// The shared kind of a traced value or cell and its immutable value metadata.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Kind {
    Value(ValueInfo),
    Cell(ValueInfo),
}

/// Whether an immutable value is borrowed or ownership is transferred.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ownership {
    Borrow,
    Move,
}

pub trait Dialect: Sync {
    fn name(&self) -> &'static str;
    fn create(&self, name: &'static str) -> Option<&'static dyn Operation>;
}

pub trait Operation: Sync {
    fn name(&self) -> &'static str;
    fn dialect(&self) -> &'static dyn Dialect;
}

// An Op is an Operation combined with an attribute set.
#[derive(Clone)]
pub struct Op {
    impl_: &'static dyn Operation,
    attrs: AttrMap,
}

impl Debug for Op {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}

impl Hash for Op {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.impl_.dialect().name().hash(state);
        self.impl_.name().hash(state);
        self.attrs.hash(state);
    }
}

impl PartialEq for Op {
    fn eq(&self, other: &Self) -> bool {
        (self.impl_.dialect().name(), self.impl_.name(), &self.attrs)
            == (
                other.impl_.dialect().name(),
                other.impl_.name(),
                &other.attrs,
            )
    }
}

impl Eq for Op {}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.impl_.dialect().name(), self.impl_.name())?;
        if !self.attrs.is_empty() {
            write!(f, "<")?;
            for (i, (k, v)) in self.attrs.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{k}: {v}")?;
            }
            write!(f, ">")?;
        }
        Ok(())
    }
}

impl Op {
    pub fn new(impl_: &'static dyn Operation) -> Self {
        Op {
            impl_,
            attrs: AttrMap::default(),
        }
    }
    pub fn new_with_attrs(impl_: &'static dyn Operation, attrs: AttrMap) -> Self {
        Op { impl_, attrs }
    }
    pub fn attrs(&self) -> &AttrMap {
        &self.attrs
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct VarScope {
    next_id: usize,
}

impl VarScope {
    pub fn new() -> VarScope {
        VarScope { next_id: 0 }
    }

    pub fn create_var(&mut self) -> Var {
        let v = Var::bind(self.next_id);
        self.next_id += 1;
        v
    }

    pub fn create_var_with_hint(&mut self, name_hint: &'static str) -> Var {
        let v = Var::bind_with_hint(self.next_id, name_hint);
        self.next_id += 1;
        v
    }
}

/// A jaxpr-style variable: bound to an index or 0 for a hole.
#[derive(Clone, Copy, Debug)]
pub struct Var {
    id: usize,
    name_hint: Option<&'static str>,
}

impl Var {
    pub fn hole() -> Self {
        Var {
            id: 0,
            name_hint: None,
        }
    }

    pub fn bind(id: usize) -> Self {
        Var {
            id: id + 1,
            name_hint: None,
        } // 0 is reserved for holes
    }

    pub fn bind_with_hint(id: usize, name_hint: &'static str) -> Self {
        Var {
            id: id + 1,
            name_hint: Some(name_hint),
        }
    }

    pub fn id(&self) -> Option<usize> {
        match self.id {
            0 => None,
            i => Some(i - 1),
        }
    }

    pub fn name_hint(&self) -> Option<&str> {
        self.name_hint
    }
}

impl Hash for Var {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl PartialEq for Var {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Var {}

impl PartialOrd for Var {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Var {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.id.cmp(&other.id)
    }
}

/// An immutable computation operand. Cell actions belong to HExpr, while
/// borrow/move ownership is assigned when lowering to [`lexpr::Expr`].
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct Input(Var);

impl Input {
    pub fn new(var: Var) -> Self {
        Self(var)
    }
    pub fn var(&self) -> &Var {
        &self.0
    }
}

impl From<Var> for Input {
    fn from(value: Var) -> Self {
        Self(value)
    }
}

/// A subexpression argument to an equation, together with its closure variables.
#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct SubExpr {
    expr: Arc<Expr>,
    closure: Vec<Input>,
}

impl SubExpr {
    pub fn new(expr: Arc<Expr>, closure: Vec<Input>) -> Self {
        Self { expr, closure }
    }

    pub fn expr(&self) -> &Arc<Expr> {
        &self.expr
    }

    pub fn closure(&self) -> &[Input] {
        &self.closure
    }
}

impl From<Var> for Output {
    fn from(value: Var) -> Self {
        if value.id().is_none() {
            Output::Aliased(value)
        } else {
            Output::New(value)
        }
    }
}

impl Output {
    pub fn var(&self) -> &Var {
        match self {
            Output::New(var) | Output::Aliased(var) => var,
        }
    }

    pub fn is_new(&self) -> bool {
        matches!(self, Output::New(_))
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub enum Output {
    // This is a "fresh" variable that does
    // not exist in the scope currently.
    New(Var),
    // The output is an alias
    // of an input or closure variable.
    Aliased(Var),
}

/// One equation: `let outs... = op [attrs] closure... ins...` in ANF (see [jaxpr](https://docs.jax.dev/en/latest/jaxpr.html)).
#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Eqn {
    op: Op,
    subexprs: Vec<SubExpr>,
    inputs: Vec<Input>,
    outputs: Vec<Output>,
}

impl Eqn {
    pub fn new<O>(op: Op, inputs: Vec<Input>, outputs: Vec<O>) -> Self
    where
        O: Into<Output>,
    {
        Eqn {
            op,
            subexprs: Vec::new(),
            inputs,
            outputs: outputs.into_iter().map(Into::into).collect(),
        }
    }

    pub fn new_with_subexprs<O>(
        op: Op,
        subexprs: Vec<SubExpr>,
        inputs: Vec<Input>,
        outputs: Vec<O>,
    ) -> Self
    where
        O: Into<Output>,
    {
        Eqn {
            op,
            subexprs,
            inputs,
            outputs: outputs.into_iter().map(Into::into).collect(),
        }
    }

    pub fn op(&self) -> &Op {
        &self.op
    }

    pub fn subexprs(&self) -> &[SubExpr] {
        &self.subexprs
    }

    pub fn inputs(&self) -> &[Input] {
        &self.inputs
    }

    pub fn outputs(&self) -> &[Output] {
        &self.outputs
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ValidationError {
    errors: Vec<ValidationIssue>,
}

impl ValidationError {
    pub fn is_empty(&self) -> bool {
        self.errors.is_empty()
    }

    pub fn errors(&self) -> &[ValidationIssue] {
        &self.errors
    }

    fn push(&mut self, issue: ValidationIssue) {
        self.errors.push(issue);
    }
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{} validation error(s)", self.errors.len())?;
        for error in &self.errors {
            writeln!(f, "- {error:?}")?;
        }
        Ok(())
    }
}

impl std::error::Error for ValidationError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ValidationIssue {
    HoleInput {
        eqn: Option<usize>,
        input: usize,
    },
    DuplicateOutput {
        eqn: usize,
        var: Var,
    },
    UndefinedInput {
        eqn: usize,
        input: usize,
        var: Var,
    },
    UndefinedSubExprClosure {
        eqn: usize,
        subexpr: usize,
        input: usize,
        var: Var,
    },
    SubExprClosureArity {
        eqn: usize,
        subexpr: usize,
        expected: usize,
        actual: usize,
    },
    UndefinedOutput {
        output: usize,
        var: Var,
    },
}

impl fmt::Display for Eqn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        pretty::PrettyVarScope::new().fmt_eqn(self, f)
    }
}

/// Functional ANF expression: `lambda closure* ; explicit* . let eqns* in outputs*`.
#[derive(Clone, Debug)]
pub struct Expr {
    closure: Vec<Input>,
    inputs: Vec<Input>,
    eqns: Vec<Eqn>,
    outputs: Vec<Var>,
    var_scope: VarScope,
    name_hint: Option<String>,
}

impl Expr {
    pub fn new(
        closure: Vec<Input>,
        inputs: Vec<Input>,
        eqns: Vec<Eqn>,
        outputs: Vec<Var>,
        var_scope: VarScope,
    ) -> Self {
        Expr {
            closure,
            inputs,
            eqns,
            outputs,
            var_scope,
            name_hint: None,
        }
    }

    pub fn new_with_hint(
        closure: Vec<Input>,
        inputs: Vec<Input>,
        eqns: Vec<Eqn>,
        outputs: Vec<Var>,
        var_scope: VarScope,
        name_hint: impl Into<String>,
    ) -> Self {
        Expr {
            closure,
            inputs,
            eqns,
            outputs,
            var_scope,
            name_hint: Some(name_hint.into()),
        }
    }

    pub fn with_name_hint(mut self, name_hint: impl Into<String>) -> Self {
        self.name_hint = Some(name_hint.into());
        self
    }

    pub fn closure(&self) -> &[Input] {
        &self.closure
    }
    pub fn inputs(&self) -> &[Input] {
        &self.inputs
    }

    pub fn eqns(&self) -> &[Eqn] {
        &self.eqns
    }

    pub fn outputs(&self) -> &[Var] {
        &self.outputs
    }

    pub fn name_hint(&self) -> Option<&str> {
        self.name_hint.as_deref()
    }

    pub fn pretty(&self) -> pretty::PrettyExpr<'_> {
        pretty::PrettyExpr::new(self)
    }

    pub fn check(&self) -> Result<(), ValidationError> {
        let mut err = ValidationError::default();
        let mut bound = std::collections::BTreeSet::new();

        for (i, input) in self.closure.iter().chain(self.inputs.iter()).enumerate() {
            if input.var().id().is_none() {
                err.push(ValidationIssue::HoleInput {
                    eqn: None,
                    input: i,
                });
            } else {
                bound.insert(*input.var());
            }
        }

        for (eqn_idx, eqn) in self.eqns.iter().enumerate() {
            for (subexpr_idx, subexpr) in eqn.subexprs.iter().enumerate() {
                if subexpr.closure.len() != subexpr.expr.closure().len() {
                    err.push(ValidationIssue::SubExprClosureArity {
                        eqn: eqn_idx,
                        subexpr: subexpr_idx,
                        expected: subexpr.expr.closure().len(),
                        actual: subexpr.closure.len(),
                    });
                }
                for (input_idx, actual) in subexpr.closure.iter().enumerate() {
                    if actual.var().id().is_none() || !bound.contains(actual.var()) {
                        err.push(ValidationIssue::UndefinedSubExprClosure {
                            eqn: eqn_idx,
                            subexpr: subexpr_idx,
                            input: input_idx,
                            var: *actual.var(),
                        });
                    }
                }
            }
            for (input_idx, input) in eqn.inputs.iter().enumerate() {
                if input.var().id().is_none() || !bound.contains(input.var()) {
                    err.push(ValidationIssue::UndefinedInput {
                        eqn: eqn_idx,
                        input: input_idx,
                        var: *input.var(),
                    });
                }
            }
            for output in &eqn.outputs {
                if output.is_new() && !bound.insert(*output.var()) {
                    err.push(ValidationIssue::DuplicateOutput {
                        eqn: eqn_idx,
                        var: *output.var(),
                    });
                }
            }
        }

        for (output_idx, output) in self.outputs.iter().enumerate() {
            if output.id().is_none() || !bound.contains(output) {
                err.push(ValidationIssue::UndefinedOutput {
                    output: output_idx,
                    var: *output,
                });
            }
        }

        if err.is_empty() { Ok(()) } else { Err(err) }
    }
}

impl Hash for Expr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.closure.hash(state);
        self.inputs.hash(state);
        self.eqns.hash(state);
        self.outputs.hash(state);
        self.var_scope.hash(state);
    }
}

impl PartialEq for Expr {
    fn eq(&self, other: &Self) -> bool {
        self.closure == other.closure
            && self.inputs == other.inputs
            && self.eqns == other.eqns
            && self.outputs == other.outputs
            && self.var_scope == other.var_scope
    }
}

impl Eq for Expr {}

impl fmt::Display for pretty::PrettyExpr<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt(f)
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        pretty::PrettyVarScope::new().fmt_expr(self, f)
    }
}

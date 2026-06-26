mod attr;
pub mod builtins;
mod dialects;
// pub mod expr_macro;
// pub mod pretty;
pub mod trace;
pub mod util;
pub mod value;

pub use attr::*;
pub use dialects::*;
pub use trace::*;
pub use util::expr_macro;

use std::cmp::{Eq, PartialEq};
use std::cmp::{Ord, PartialOrd};
use std::collections::BTreeMap;
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

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum Effect {
    Unused,
    Read,
    Write,
    ReadWrite,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum InputKind {
    Unused,
    Read,
    // The following two fields are only for cells.
    //
    // noalias: whether we are guaranteed
    // that this input cell is not aliased to any other input cell.
    //
    Write { noalias: bool },
    ReadWrite { noalias: bool },
}

impl InputKind {
    pub fn effect(self) -> Effect {
        match self {
            InputKind::Unused => Effect::Unused,
            InputKind::Read => Effect::Read,
            InputKind::Write { .. } => Effect::Write,
            InputKind::ReadWrite { .. } => Effect::ReadWrite,
        }
    }

    fn restriction_rank(self) -> usize {
        match self {
            InputKind::Unused => 0,
            InputKind::Read => 1,
            InputKind::Write { noalias: false } => 2,
            InputKind::Write { noalias: true } => 3,
            InputKind::ReadWrite { noalias: false } => 4,
            InputKind::ReadWrite { noalias: true } => 5,
        }
    }

    pub fn can_bind_to(self, bound: InputKind) -> bool {
        bound == InputKind::Unused || self.restriction_rank() <= bound.restriction_rank()
    }
}

impl Effect {
    pub fn combine(self, other: Effect) -> Effect {
        match (self, other) {
            (Effect::Unused, _) => other,
            (_, Effect::Unused) => self,
            (Effect::Read, Effect::Write) => Effect::ReadWrite,
            (Effect::Write, Effect::Read) => Effect::ReadWrite,
            _ => self,
        }
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct Effects {
    effected: BTreeMap<Var, Effect>,
}

impl Effects {
    pub fn new() -> Self {
        Effects {
            effected: BTreeMap::new(),
        }
    }
    pub fn insert(&mut self, var: Var, effect: Effect) {
        self.effected.insert(var, effect);
    }
    pub fn remove(&mut self, var: &Var) -> Option<Effect> {
        self.effected.remove(var)
    }
    pub fn size(&self) -> usize {
        self.effected.len()
    }
    pub fn is_empty(&self) -> bool {
        self.effected.is_empty()
    }
}

impl std::ops::Index<&Var> for Effects {
    type Output = Effect;
    fn index(&self, var: &Var) -> &Self::Output {
        self.effected.get(var).unwrap_or(&Effect::Unused)
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum Input {
    Unused(Var),
    Read(Var),
    Write { src: Var, noalias: bool },
    ReadWrite { src: Var, noalias: bool },
}

impl Input {
    pub fn new(var: Var, effect: Effect) -> Self {
        match effect {
            Effect::Unused => Self::Unused(var),
            Effect::Read => Self::Read(var),
            Effect::Write => Self::Write {
                src: var,
                noalias: false,
            },
            Effect::ReadWrite => Self::ReadWrite {
                src: var,
                noalias: false,
            },
        }
    }
    pub fn unused(var: Var) -> Self {
        Self::Unused(var)
    }
    pub fn read(var: Var) -> Self {
        Self::Read(var)
    }
    pub fn write(var: Var, noalias: bool) -> Self {
        Self::Write { src: var, noalias }
    }
    pub fn read_write(var: Var, noalias: bool) -> Self {
        Self::ReadWrite { src: var, noalias }
    }
    pub fn var(&self) -> &Var {
        match self {
            Input::Unused(var)
            | Input::Read(var)
            | Input::Write { src: var, .. }
            | Input::ReadWrite { src: var, .. } => var,
        }
    }
    pub fn kind(&self) -> InputKind {
        match self {
            Input::Unused(_) => InputKind::Unused,
            Input::Read(_) => InputKind::Read,
            Input::Write { noalias, .. } => InputKind::Write { noalias: *noalias },
            Input::ReadWrite { noalias, .. } => InputKind::ReadWrite { noalias: *noalias },
        }
    }
    pub fn effect(&self) -> Effect {
        self.kind().effect()
    }

    pub fn can_bind_to(&self, bound: &Input) -> bool {
        self.kind().can_bind_to(bound.kind())
    }
}

impl From<Var> for Input {
    fn from(value: Var) -> Self {
        Input::Read(value)
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

    // Propagate effects *backwards* through an equation.
    pub fn propagate_effects(&self, effects: &mut Effects) {
        // By default add any input effects to the effects map.
        for input in self.inputs.iter().filter(|i| i.effect() != Effect::Unused) {
            effects.insert(input.var().clone(), input.effect());
        }
        for input in self.subexprs.iter().flat_map(|subexpr| subexpr.closure()) {
            effects.insert(input.var().clone(), input.effect());
        }
        let mut output_effects = Effects::new();
        let mut max_effect = Effect::Unused;
        for output in &self.outputs {
            if let Some(e) = effects.remove(output.var()) {
                output_effects.insert(*output.var(), e);
                max_effect = max_effect.combine(e);
            }
        }
        if output_effects.is_empty() {
            return;
        }
        // TODO: Lookup the operation effects implementation.
        // If there is no implementation, pessimistically assume
        // the worst possible input-output association.
        for input in &self.inputs {
            effects.insert(input.var().clone(), max_effect);
        }
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
    InvalidSubExprBinding {
        eqn: usize,
        subexpr: usize,
        input: usize,
        expected: InputKind,
        actual: InputKind,
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
                    if let Some(expected) = subexpr.expr.closure().get(input_idx) {
                        if !actual.can_bind_to(expected) {
                            err.push(ValidationIssue::InvalidSubExprBinding {
                                eqn: eqn_idx,
                                subexpr: subexpr_idx,
                                input: input_idx,
                                expected: expected.kind(),
                                actual: actual.kind(),
                            });
                        }
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

//! Immutable low-level IR. Cell actions and boundary mappings live in HExpr and
//! the lower module, respectively.

use std::borrow::Cow;
use std::collections::BTreeSet;

use crate::Symbol;
use crate::lower::LoweringError;
use crate::util::pretty;
use crate::{Op, Var, VarInfo};
use std::fmt;
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Eqn {
    pub(crate) op: Op,
    pub(crate) inputs: Vec<Var>,
    pub(crate) outputs: Vec<(Var, VarInfo)>,
    pub(crate) subexprs: Vec<SubExpr>,
}
impl Eqn {
    pub fn new(op: Op, inputs: Vec<Var>, outputs: Vec<(Var, VarInfo)>) -> Self {
        Self::new_with_subexprs(op, vec![], inputs, outputs)
    }
    pub fn new_with_subexprs(
        op: Op,
        subexprs: Vec<SubExpr>,
        inputs: Vec<Var>,
        outputs: Vec<(Var, VarInfo)>,
    ) -> Self {
        Self {
            op,
            subexprs,
            inputs,
            outputs,
        }
    }
    pub fn op(&self) -> &Op {
        &self.op
    }
    pub fn inputs(&self) -> &[Var] {
        &self.inputs
    }
    pub fn outputs(&self) -> &[(Var, VarInfo)] {
        &self.outputs
    }
    pub fn subexprs(&self) -> &[SubExpr] {
        &self.subexprs
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SubExpr {
    /// Parent symbols in the order of the child’s `symbol_closure`.
    pub symbol_captures: Vec<Symbol>,
    pub expr: Arc<Expr>,
    pub captures: Vec<Var>,
}

impl SubExpr {
    /// Bind the child’s free symbols to IDs in the enclosing region.
    pub fn with_symbol_captures(mut self, captures: Vec<Symbol>) -> Self {
        self.symbol_captures = captures;
        self
    }
    pub fn new(expr: Arc<Expr>, captures: Vec<Var>) -> Self {
        Self {
            expr,
            captures,
            symbol_captures: vec![],
        }
    }
    pub fn expr(&self) -> &Arc<Expr> {
        &self.expr
    }
    pub fn closure(&self) -> &[Var] {
        &self.captures
    }
}

/// Pure immutable SSA. Results refer to parameters or equation results by
/// variable. A value can be used or returned any number of times.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Expr {
    pub(crate) symbol_closure: Vec<Symbol>,
    pub(crate) symbol_inputs: Vec<Symbol>,
    pub(crate) closure: Vec<(Var, VarInfo)>,
    pub(crate) inputs: Vec<(Var, VarInfo)>,
    pub(crate) eqns: Vec<Eqn>,
    pub(crate) outputs: Vec<Var>,
    pub(crate) name_hint: Option<Cow<'static, str>>,
}
impl Expr {
    /// Declarations use IDs local to this region, not IDs in a caller.
    pub fn with_symbols(mut self, closure: Vec<Symbol>, inputs: Vec<Symbol>) -> Self {
        self.symbol_closure = closure;
        self.symbol_inputs = inputs;
        self
    }
    pub fn symbol_closure(&self) -> &[Symbol] {
        &self.symbol_closure
    }
    pub fn symbol_inputs(&self) -> &[Symbol] {
        &self.symbol_inputs
    }
    /// Whether this region declares `s` as a captured or input symbol.
    pub fn declares_symbol(&self, s: Symbol) -> bool {
        self.symbol_closure.contains(&s) || self.symbol_inputs.contains(&s)
    }
    pub fn new(
        closure: Vec<(Var, VarInfo)>,
        inputs: Vec<(Var, VarInfo)>,
        eqns: Vec<Eqn>,
        outputs: Vec<Var>,
    ) -> Self {
        Self {
            symbol_closure: vec![],
            symbol_inputs: vec![],
            closure,
            inputs,
            eqns,
            outputs,
            name_hint: None,
        }
    }
    pub fn new_with_hint(
        closure: Vec<(Var, VarInfo)>,
        inputs: Vec<(Var, VarInfo)>,
        eqns: Vec<Eqn>,
        outputs: Vec<Var>,
        name_hint: impl Into<Cow<'static, str>>,
    ) -> Self {
        Self::new(closure, inputs, eqns, outputs).with_name_hint(name_hint)
    }
    pub fn with_name_hint(mut self, hint: impl Into<Cow<'static, str>>) -> Self {
        self.name_hint = Some(hint.into());
        self
    }
    pub fn name_hint(&self) -> Option<&str> {
        self.name_hint.as_deref()
    }

    pub fn pretty(&self) -> pretty::PrettyExpr<'_> {
        pretty::PrettyExpr::new(self)
    }
    pub fn closure(&self) -> &[(Var, VarInfo)] {
        &self.closure
    }
    pub fn inputs(&self) -> &[(Var, VarInfo)] {
        &self.inputs
    }
    /// Find an immutable value's metadata at its definition site.
    pub fn value_info(&self, var: &Var) -> Option<&VarInfo> {
        self.closure
            .iter()
            .chain(&self.inputs)
            .find_map(|(defined, info)| (defined == var).then_some(info))
            .or_else(|| {
                self.eqns
                    .iter()
                    .rev()
                    .flat_map(Eqn::outputs)
                    .find_map(|(defined, info)| (defined == var).then_some(info))
            })
    }
    pub fn eqns(&self) -> &[Eqn] {
        &self.eqns
    }
    /// Results in order, including any inputs or repeated values.
    pub fn outputs(&self) -> &[Var] {
        &self.outputs
    }

    /// Verify definition order, nested capture bindings, and result sources.
    pub fn check(&self) -> Result<(), LoweringError> {
        crate::check_symbol_scope(
            &self.symbol_closure,
            &self.symbol_inputs,
            self.inputs.iter().flat_map(|v| v.1.symbols()),
            self.closure
                .iter()
                .chain(self.eqns.iter().flat_map(|e| &e.outputs))
                .flat_map(|v| v.1.symbols()),
        )?;
        let mut defined = BTreeSet::new();
        for &(var, _) in self.closure.iter().chain(&self.inputs) {
            if var.id().is_none() || !defined.insert(var) {
                return Err(LoweringError::InvalidComputation);
            }
        }
        for eqn in &self.eqns {
            for &input in eqn
                .inputs
                .iter()
                .chain(eqn.subexprs.iter().flat_map(|sub| &sub.captures))
            {
                if !defined.contains(&input) {
                    return Err(LoweringError::UndefinedVariable(input));
                }
            }
            for sub in &eqn.subexprs {
                if sub.captures.len() != sub.expr.closure.len() {
                    return Err(LoweringError::CaptureArity {
                        expected: sub.expr.closure.len(),
                        actual: sub.captures.len(),
                    });
                }
                crate::check_symbol_captures(
                    sub.expr.symbol_closure(),
                    &sub.symbol_captures,
                    |s| self.declares_symbol(s),
                )?;
                sub.expr.check()?;
            }
            eqn.op().check_expr(eqn)?;
            for &(output, _) in &eqn.outputs {
                if output.id().is_none() {
                    return Err(LoweringError::InvalidComputation);
                }
                if !defined.insert(output) {
                    return Err(LoweringError::DuplicateDefinition(output));
                }
            }
        }
        for &output in &self.outputs {
            if !defined.contains(&output) {
                return Err(LoweringError::UndefinedVariable(output));
            }
        }
        Ok(())
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        pretty::PrettyVarScope::new().fmt_expr(self, f)
    }
}
impl fmt::Display for Eqn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        pretty::PrettyVarScope::new().fmt_eqn(self, f)
    }
}
impl fmt::Display for pretty::PrettyExpr<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt(f)
    }
}

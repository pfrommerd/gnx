//! Immutable low-level IR. Cell actions and boundary mappings live in HExpr and
//! the lower module, respectively.

use std::borrow::Cow;
use std::collections::BTreeSet;

use crate::lower::LoweringError;
use crate::util::pretty;
use crate::{Op, ValueInfo, Var};
use std::fmt;
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Eqn {
    pub(crate) op: Op,
    pub(crate) inputs: Vec<Var>,
    pub(crate) outputs: Vec<(Var, ValueInfo)>,
    pub(crate) subexprs: Vec<SubExpr>,
}
impl Eqn {
    pub fn new(op: Op, inputs: Vec<Var>, outputs: Vec<(Var, ValueInfo)>) -> Self {
        Self::new_with_subexprs(op, vec![], inputs, outputs)
    }
    pub fn new_with_subexprs(
        op: Op,
        subexprs: Vec<SubExpr>,
        inputs: Vec<Var>,
        outputs: Vec<(Var, ValueInfo)>,
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
    pub fn outputs(&self) -> &[(Var, ValueInfo)] {
        &self.outputs
    }
    pub fn subexprs(&self) -> &[SubExpr] {
        &self.subexprs
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SubExpr {
    pub expr: Arc<Expr>,
    pub captures: Vec<Var>,
}

impl SubExpr {
    pub fn new(expr: Arc<Expr>, captures: Vec<Var>) -> Self {
        Self { expr, captures }
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
    pub(crate) closure: Vec<(Var, ValueInfo)>,
    pub(crate) inputs: Vec<(Var, ValueInfo)>,
    pub(crate) eqns: Vec<Eqn>,
    pub(crate) outputs: Vec<Var>,
    pub(crate) name_hint: Option<Cow<'static, str>>,
}
impl Expr {
    pub fn new(
        closure: Vec<(Var, ValueInfo)>,
        inputs: Vec<(Var, ValueInfo)>,
        eqns: Vec<Eqn>,
        outputs: Vec<Var>,
    ) -> Self {
        Self {
            closure,
            inputs,
            eqns,
            outputs,
            name_hint: None,
        }
    }
    pub fn new_with_hint(
        closure: Vec<(Var, ValueInfo)>,
        inputs: Vec<(Var, ValueInfo)>,
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
    pub fn closure(&self) -> &[(Var, ValueInfo)] {
        &self.closure
    }
    pub fn inputs(&self) -> &[(Var, ValueInfo)] {
        &self.inputs
    }
    /// Find an immutable value's metadata at its definition site.
    pub fn value_info(&self, var: &Var) -> Option<&ValueInfo> {
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

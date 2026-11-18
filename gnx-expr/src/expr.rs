//! Immutable low-level IR. Cell actions and boundary mappings live in HExpr and
//! the lower module, respectively.

use std::collections::{BTreeMap, BTreeSet};

use crate::lower::LoweringError;
use crate::util::pretty;
use crate::{Kind, Op, Ownership, ValueInfo, Var};
use std::fmt;
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Eqn {
    pub(crate) op: Op,
    pub(crate) inputs: Vec<(Var, Ownership)>,
    pub(crate) outputs: Vec<(Var, Kind)>,
    pub(crate) subexprs: Vec<SubExpr>,
}
impl Eqn {
    pub fn new(op: Op, inputs: Vec<(Var, Ownership)>, outputs: Vec<(Var, Kind)>) -> Self {
        Self::new_with_subexprs(op, vec![], inputs, outputs)
    }
    pub fn new_with_subexprs(
        op: Op,
        subexprs: Vec<SubExpr>,
        inputs: Vec<(Var, Ownership)>,
        outputs: Vec<(Var, Kind)>,
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
    pub fn inputs(&self) -> &[(Var, Ownership)] {
        &self.inputs
    }
    pub fn outputs(&self) -> &[(Var, Kind)] {
        &self.outputs
    }
    pub fn subexprs(&self) -> &[SubExpr] {
        &self.subexprs
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SubExpr {
    pub expr: Arc<Expr>,
    pub captures: Vec<(Var, Ownership)>,
}

impl SubExpr {
    pub fn new(expr: Arc<Expr>, captures: Vec<(Var, Ownership)>) -> Self {
        Self { expr, captures }
    }
    pub fn expr(&self) -> &Arc<Expr> {
        &self.expr
    }
    pub fn closure(&self) -> &[(Var, Ownership)] {
        &self.captures
    }
}

/// Pure immutable IR. Parameters carry ownership and metadata. Operands and
/// capture bindings carry ownership. Equations define owned values as `(Var, Kind)` pairs. Expression
/// returns are owned and refer to existing values by `Var`.
/// Borrowed input passthroughs are removed during lowering and represented in
/// boundary metadata, never in this expression's outputs. Unused moved inputs
/// are dropped by the callee. Moving does not imply unique storage.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Expr {
    pub(crate) closure: Vec<(Var, Ownership, ValueInfo)>,
    pub(crate) inputs: Vec<(Var, Ownership, ValueInfo)>,
    pub(crate) eqns: Vec<Eqn>,
    pub(crate) outputs: Vec<Var>,
    pub(crate) name_hint: Option<String>,
}
impl Expr {
    pub fn new(
        closure: Vec<(Var, Ownership, ValueInfo)>,
        inputs: Vec<(Var, Ownership, ValueInfo)>,
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
        closure: Vec<(Var, Ownership, ValueInfo)>,
        inputs: Vec<(Var, Ownership, ValueInfo)>,
        eqns: Vec<Eqn>,
        outputs: Vec<Var>,
        name_hint: impl Into<String>,
    ) -> Self {
        Self::new(closure, inputs, eqns, outputs).with_name_hint(name_hint)
    }
    pub fn with_name_hint(mut self, hint: impl Into<String>) -> Self {
        self.name_hint = Some(hint.into());
        self
    }
    pub fn name_hint(&self) -> Option<&str> {
        self.name_hint.as_deref()
    }
    pub fn pretty(&self) -> pretty::PrettyExpr<'_> {
        pretty::PrettyExpr::new(self)
    }
    pub fn closure(&self) -> &[(Var, Ownership, ValueInfo)] {
        &self.closure
    }
    pub fn inputs(&self) -> &[(Var, Ownership, ValueInfo)] {
        &self.inputs
    }
    /// Find an immutable value's metadata at its definition site.
    pub fn value_info(&self, var: &Var) -> Option<&ValueInfo> {
        self.closure
            .iter()
            .chain(&self.inputs)
            .find_map(|(defined, _, info)| (defined == var).then_some(info))
            .or_else(|| {
                self.eqns
                    .iter()
                    .rev()
                    .flat_map(Eqn::outputs)
                    .find_map(|(defined, kind)| match kind {
                        Kind::Value(info) if defined == var => Some(info),
                        _ => None,
                    })
            })
    }
    pub fn eqns(&self) -> &[Eqn] {
        &self.eqns
    }
    pub fn outputs(&self) -> &[Var] {
        &self.outputs
    }

    /// Verify definition order, ownership transfers, and owned-only returns.
    pub fn check(&self) -> Result<(), LoweringError> {
        let mut available = BTreeMap::new();
        for &(var, ownership, _) in self.closure.iter().chain(&self.inputs) {
            if var.id().is_none() || available.insert(var, ownership).is_some() {
                return Err(LoweringError::InvalidComputation);
            }
        }
        let mut defined: BTreeSet<_> = available.keys().copied().collect();
        for eqn in &self.eqns {
            for input in eqn
                .inputs
                .iter()
                .chain(eqn.subexprs.iter().flat_map(|s| &s.captures))
            {
                check_use(*input, &mut available)?;
            }
            for sub in &eqn.subexprs {
                if sub.captures.len() != sub.expr.closure.len() {
                    return Err(LoweringError::CaptureArity {
                        expected: sub.expr.closure.len(),
                        actual: sub.captures.len(),
                    });
                }
                for ((var, actual), (_, required, _)) in sub.captures.iter().zip(sub.expr.closure())
                {
                    if *required == Ownership::Move && *actual != Ownership::Move {
                        return Err(LoweringError::CaptureOwnership(*var));
                    }
                }
                sub.expr.check()?;
            }
            if eqn.op() == &crate::builtins::call_op() {
                if eqn.subexprs.len() != 1 {
                    return Err(LoweringError::CallArity);
                }
                let body = &eqn.subexprs[0].expr;
                if eqn.inputs.len() != body.inputs.len() || eqn.outputs.len() != body.outputs.len()
                {
                    return Err(LoweringError::CallArity);
                }
                for ((var, actual), (_, required, _)) in eqn.inputs.iter().zip(body.inputs()) {
                    if *required == Ownership::Move && *actual != Ownership::Move {
                        return Err(LoweringError::MoveBorrowed(*var));
                    }
                }
            }
            for (output, kind) in &eqn.outputs {
                let output = *output;
                if !matches!(kind, Kind::Value(_)) {
                    return Err(LoweringError::ExpectedValue(output));
                }
                if output.id().is_none() {
                    return Err(LoweringError::InvalidComputation);
                }
                if !defined.insert(output) {
                    return Err(LoweringError::DuplicateDefinition(output));
                }
                available.insert(output, Ownership::Move);
            }
        }
        for &output in &self.outputs {
            if available.get(&output) == Some(&Ownership::Borrow) {
                return Err(LoweringError::BorrowedOutput(output));
            }
            check_use((output, Ownership::Move), &mut available)?;
        }
        Ok(())
    }
}

fn check_use(
    (var, mode): (Var, Ownership),
    available: &mut BTreeMap<Var, Ownership>,
) -> Result<(), LoweringError> {
    let ownership = available
        .get(&var)
        .ok_or(LoweringError::UndefinedVariable(var))?;
    if mode == Ownership::Move {
        if *ownership != Ownership::Move {
            return Err(LoweringError::MoveBorrowed(var));
        }
        available.remove(&var);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::LoweringError;

    #[test]
    fn validator_rejects_borrowed_outputs() {
        let expr = crate::expr!(crate::ValueInfo::new(0usize); { lambda ; &value . in (value) });
        let var = expr.inputs()[0].0;
        assert_eq!(expr.check(), Err(LoweringError::BorrowedOutput(var)));
    }

    #[test]
    fn validator_rejects_use_after_move() {
        let expr =
            crate::expr!(crate::ValueInfo::new(0usize); { lambda ; value . in (value, value) });
        let var = expr.inputs()[0].0;
        assert_eq!(expr.check(), Err(LoweringError::UndefinedVariable(var)));
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

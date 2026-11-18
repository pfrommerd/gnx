//! Immutable low-level IR. Cell actions and boundary mappings live in HExpr and
//! the lower module, respectively.

use std::collections::{BTreeMap, BTreeSet};

use crate::lower::LoweringError;
use crate::{Op, Ownership, Var};

#[derive(Clone, Debug)]
pub struct Eqn {
    pub(crate) op: Op,
    pub(crate) inputs: Vec<(Var, Ownership)>,
    pub(crate) outputs: Vec<Var>,
    pub(crate) subexprs: Vec<SubExpr>,
}
impl Eqn {
    pub fn op(&self) -> &Op {
        &self.op
    }
    pub fn inputs(&self) -> &[(Var, Ownership)] {
        &self.inputs
    }
    pub fn outputs(&self) -> &[Var] {
        &self.outputs
    }
    pub fn subexprs(&self) -> &[SubExpr] {
        &self.subexprs
    }
}

#[derive(Clone, Debug)]
pub struct SubExpr {
    pub expr: Expr,
    pub captures: Vec<(Var, Ownership)>,
}

/// Pure immutable IR. Parameters, operands, and captures are (Var, Ownership)
/// pairs. Equation results and expression returns are always owned values.
/// Borrowed input passthroughs are removed during lowering and represented in
/// boundary metadata, never in this expression's outputs. Unused moved inputs
/// are dropped by the callee. Moving does not imply unique storage.
#[derive(Clone, Debug)]
pub struct Expr {
    pub(crate) closure: Vec<(Var, Ownership)>,
    pub(crate) inputs: Vec<(Var, Ownership)>,
    pub(crate) eqns: Vec<Eqn>,
    pub(crate) outputs: Vec<Var>,
}
impl Expr {
    pub fn closure(&self) -> &[(Var, Ownership)] {
        &self.closure
    }
    pub fn inputs(&self) -> &[(Var, Ownership)] {
        &self.inputs
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
        for &(var, ownership) in self.closure.iter().chain(&self.inputs) {
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
                    return Err(LoweringError::InvalidComputation);
                }
                sub.expr.check()?;
            }
            for &output in &eqn.outputs {
                if output.id().is_none() || !defined.insert(output) {
                    return Err(LoweringError::InvalidComputation);
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
    use super::*;
    use crate::VarScope;

    #[test]
    fn validator_rejects_borrowed_outputs() {
        let var = VarScope::new().create_var();
        let expr = Expr {
            closure: vec![],
            inputs: vec![(var, Ownership::Borrow)],
            eqns: vec![],
            outputs: vec![var],
        };
        assert_eq!(expr.check(), Err(LoweringError::BorrowedOutput(var)));
    }

    #[test]
    fn validator_rejects_use_after_move() {
        let var = VarScope::new().create_var();
        let expr = Expr {
            closure: vec![],
            inputs: vec![(var, Ownership::Move)],
            eqns: vec![],
            outputs: vec![var, var],
        };
        assert_eq!(expr.check(), Err(LoweringError::UndefinedVariable(var)));
    }
}

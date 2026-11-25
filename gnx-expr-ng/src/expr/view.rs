use std::ops::Deref;

use super::{EqnData, ExprData, RootExpr, SubExprData};
use crate::op::{OpData, Operation};
use crate::var::{Var, VarInfoData};

// Views into a root expression: each pairs a piece of the
// archived data with the root needed to resolve what it refers to
// (operation / info implementations, other expressions).

// View of an expression
#[derive(Clone, Copy)]
pub struct Expr<'a> {
    root: &'a RootExpr,
    expr: &'a ExprData,
}

impl<'a> Expr<'a> {
    // SAFETY: The root and exprdata must use the same typeinfo tables!
    pub(super) unsafe fn new(root: &'a RootExpr, expr: &'a ExprData) -> Self {
        Expr { root, expr }
    }
}

// View of an equation in an expression
#[derive(Clone, Copy)]
pub struct Eqn<'a> {
    root: &'a RootExpr,
    eqn: &'a EqnData,
}

impl<'a> Eqn<'a> {
    // SAFETY: The Root table and Eqn correspond
    pub(super) unsafe fn new(root: &'a RootExpr, eqn: &'a EqnData) -> Self {
        Eqn { root, eqn }
    }
}

// View of the operation of an equation
#[derive(Clone, Copy)]
pub struct Op<'a> {
    root: &'a RootExpr,
    op: &'a OpData,
}

impl<'a> Op<'a> {
    // SAFETY the root and op correspond!
    pub(super) unsafe fn new(root: &'a RootExpr, op: &'a OpData) -> Self {
        Op { root, op }
    }
}

impl<'a> Deref for Op<'a> {
    type Target = dyn Operation;

    fn deref(&self) -> &Self::Target {
        // SAFETY: The rootexpr and opdata correspond, so this is safe!
        unsafe { self.op.op.get(&self.root.op_impls) }.expect("Impl not found!")
    }
}

// View of a subexpression of an equation: a reference to
// another expression, plus the symbols and vars provided to it
#[derive(Clone, Copy)]
pub struct SubExpr<'a> {
    root: &'a RootExpr,
    subexpr: &'a SubExprData,
}

impl<'a> SubExpr<'a> {
    pub(super) unsafe fn new(root: &'a RootExpr, subexpr: &'a SubExprData) -> Self {
        SubExpr { root, subexpr }
    }
}

// View of a declared variable (an input or closure input
// of an expression): the variable and its info
#[derive(Clone, Copy)]
pub struct Param<'a> {
    root: &'a RootExpr,
    var: Var,
    info: &'a VarInfoData,
}

impl<'a> Param<'a> {
    pub(super) fn new(root: &'a RootExpr, var: Var, info: &'a VarInfoData) -> Self {
        Param { root, var, info }
    }
}

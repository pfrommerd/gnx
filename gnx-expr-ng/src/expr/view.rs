use rkyv::Archived;

use crate::expr::RootExpr;
use crate::expr::data::{EqnData, ExprData};

// Views of eqn
pub struct Eqn<'a> {
    root: &'a RootExpr,
    eqn: &'a Archived<EqnData>,
}

pub struct Expr<'a> {
    root: &'a RootExpr,
    expr: &'a Archived<ExprData>,
}

impl<'a> Expr<'a> {
    pub(super) fn new(root: &'a RootExpr, expr: &'a Archived<ExprData>) -> Self {
        Expr { root, expr }
    }
}

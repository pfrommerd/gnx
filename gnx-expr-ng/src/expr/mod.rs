use crate::util::Blake3Hash;
use crate::{op::OpTypeImpls, util::OwnedArchive, var::InfoTypeImpls};

use std::hash::{Hash, Hasher};
use std::sync::Arc;

mod builder;
pub use builder::*;

// Re-export the view types
mod view;
pub use view::*;

// A root expression
pub struct RootExpr {
    // An owned handle of the underlying root data
    data: OwnedArchive<RootData>,
    hash: Blake3Hash,
    // The associated operation lookup table, resolved
    op_impls: OpTypeImpls,
    // The associated info lookup table, resolved
    info_impls: InfoTypeImpls,
    // The external expressions, resolved at load time.
    external_exprs: ExternalExprs,
}

#[rustfmt::skip]
impl Hash for RootExpr {
    fn hash<H: Hasher>(&self, state: &mut H) { self.hash.hash(state); }
}
#[rustfmt::skip]
impl PartialEq for RootExpr {
    fn eq(&self, other: &Self) -> bool { self.hash == other.hash }
}
impl Eq for RootExpr {}
#[rustfmt::skip]
impl PartialOrd for RootExpr {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> { self.hash.partial_cmp(&other.hash) }
}
#[rustfmt::skip]
impl Ord for RootExpr {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering { self.hash.cmp(&other.hash) }
}

// A handle into a root expr, a combination
// of an index into the expr table + an arc to the underlying root expr
#[derive(Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct ExprHandle {
    // The underlying root expr
    root: Arc<RootExpr>,
    idx: ExprIdx,
}

impl ExprHandle {
    pub fn view(&self) -> Expr<'_> {
        // ExprHandles should always point to
        // internal expressions
        debug_assert!(self.idx.is_internal());
        let idx: usize = self.idx.into();
        let expr: &ExprData = &self.root.data.exprs.internal[idx];
        unsafe { Expr::new(self.root.as_ref(), expr) }
    }
}

impl RootExpr {
    pub fn builder() -> RootBuilder {
        RootBuilder::default()
    }
    pub fn view(&self) -> Expr<'_> {
        debug_assert!(self.data.root.is_internal());
        let idx: usize = self.data.root.into();
        let expr: &ExprData = &self.data.exprs.internal[idx];
        unsafe { Expr::new(self, expr) }
    }
    // Get a handle to the root expr in self
    pub fn handle(self: Arc<Self>) -> ExprHandle {
        let idx = self.data.root;
        ExprHandle { root: self, idx }
    }
}

pub struct ExternalExprs(Vec<ExprHandle>);

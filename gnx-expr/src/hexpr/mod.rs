//! Mutation-aware expressions.
//!
//! Variables are either values or cells. Cell creations, reads and writes
//! are structural instructions. Operations summarize their effects on
//! cells, including effects from subexpressions.

use crate::root::{Handle, Ref, Root};

mod builder;
pub use builder::*;

mod view;
pub use view::*;

/// A root high-level expression
pub type HRootExpr = Root<HExprBuilder>;

/// A handle to an expression of a root high-level expression
pub type HExprHandle = Handle<HExprBuilder>;

/// A reference to a high-level expression that is being built, or to one
/// of a root that is already built
pub type HExprRef = Ref<HExprBuilder>;

impl HRootExpr {
    pub fn builder() -> HExprBuilder {
        HExprBuilder::new()
    }
}

/// Summary of possible cell accesses. Values have no cell-access summary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CellAccess {
    Unused,
    Read,
    Write,
    ReadWrite,
}

impl CellAccess {
    pub fn combine(self, other: Self) -> Self {
        match (self, other) {
            (Self::Unused, _) => other,
            (_, Self::Unused) => self,
            (Self::ReadWrite, _) | (_, Self::ReadWrite) => Self::ReadWrite,
            (Self::Read, Self::Write) | (Self::Write, Self::Read) => Self::ReadWrite,
            _ => self,
        }
    }
}

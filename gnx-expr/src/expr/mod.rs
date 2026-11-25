//! Immutable low-level IR: pure SSA. Cells and their accesses live in
//! [`crate::hexpr`].

use crate::root::{Handle, Ref, Root};

mod builder;
pub use builder::*;

// Re-export the view types
mod view;
pub use view::*;

/// A root expression
pub type RootExpr = Root<ExprBuilder>;

/// A handle to an expression of a root expression
pub type ExprHandle = Handle<ExprBuilder>;

/// A reference to an expression that is being built, or to one of a
/// root expression that is already built
pub type ExprRef = Ref<ExprBuilder>;

impl RootExpr {
    pub fn builder() -> ExprBuilder {
        ExprBuilder::new()
    }
}

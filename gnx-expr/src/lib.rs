mod check;
pub mod eval;
mod expr;
pub mod hexpr;
mod interface;
pub mod lower;
mod op;
mod root;
mod symbol;
pub mod trace;
pub mod util;
pub mod value;
mod var;

pub use check::BuildError;
pub use eval::{EvalCommit, EvalPlanError, PreparedEval};
pub use expr::*;
pub use hexpr::*;
pub use interface::{Caster, InterfaceError, InterfaceImpl};
pub use lower::{CallError, InputId, InputSpec, LoweredExpr, LoweringError};
pub use op::*;
pub use root::{ExprIdx, ExprSource, Handle, Ref, Root, UnresolvedRoot};
pub use symbol::Symbol;
pub use trace::*;
pub use util::expr_macro;
pub use util::{
    ARCHIVE_ALIGNMENT, Blake3Hash, Dialect, DialectType, OwnedArchive, SharedBytes, TypeName, Typed,
};
pub use value::{ConcreteValue, Generic, Traceable, Value};
pub use var::*;

// Used by the exported macros
#[doc(hidden)]
pub mod __private {
    pub use inventory;
    pub use rkyv;
}

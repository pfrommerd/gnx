use std::fmt;

use crate::expr::Eqn;
use crate::hexpr::{CellAccess, HEqn};
use crate::util::{
    ArchivedDyn, ArchivedTypeTable, DynRef, OwnedDyn, TypeImpls, TypeTable, TypeTableBuilder,
    TypedIn, dyn_archive,
};
use crate::var::Var;

dyn_archive! {
    /// An operation and its immutable, operation-specific parameters.
    ///
    /// Implemented by archived types. A concrete operation is named in a
    /// dialect, and is made from a builder: a type that archives as it.
    pub trait Operation {
        /// Writes the qualified operation name and any parameters, such
        /// as `cf::call`.
        fn display(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.type_name())
        }

        /// Check the operation-specific signature of an equation, after the
        /// generic operand and subexpression checks.
        fn check_expr(&self, _eqn: Eqn<'_>) -> Result<(), crate::LoweringError> {
            Ok(())
        }

        /// Lower an application to immutable SSA. Operations with
        /// subexpressions must supply their own rule for passing operands
        /// and threading state.
        fn lower(
            &self,
            eqn: HEqn<'_>,
            ctx: &mut crate::lower::LoweringContext<'_>,
        ) -> Result<(), crate::LoweringError> {
            ctx.lower_value_apply(eqn)
        }

        /// Summarize every possible effect of this application on a cell of
        /// the enclosing expression. The default follows the variables
        /// provided to every subexpression. Overrides can reuse that
        /// traversal and must include all effects the operation can have.
        fn cell_access(&self, eqn: HEqn<'_>, cell: Var) -> CellAccess {
            eqn.captured_cell_access(cell)
        }
    }
}

impl dyn Operation {
    /// The name of the operation within its dialect.
    pub fn name(&self) -> &'static str {
        self.type_name().name
    }

    pub fn dialect(&self) -> &'static str {
        self.type_name().dialect
    }

    /// The operation, if it is of the (archived) type `A`.
    pub fn downcast_ref<A: TypedIn<dyn Operation>>(&self) -> Option<&A> {
        // SAFETY: a name identifies one type among the dialects of Operation
        (self.type_name() == A::NAME).then(|| unsafe { &*(self as *const Self).cast::<A>() })
    }
}

impl fmt::Display for dyn Operation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.display(f)
    }
}

/// An owned operation, made from a builder with `Op::new`.
pub type Op = OwnedDyn<dyn Operation>;

/// An operation in a root.
pub type OpRef<'a> = DynRef<'a, dyn Operation>;

/// The archived form of an operation in a root.
pub type OpData = ArchivedDyn<dyn Operation>;

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.get().display(f)
    }
}

impl fmt::Display for OpRef<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.get().display(f)
    }
}

// Serialization helpers: OpTypeTable and OpTypeImpls

// Used during serialization to build the OpTypeTable
pub type OpTypeCtx = TypeTableBuilder<dyn Operation>;

// The archived version of an optypetable
// contains the (dialect, opname) values.
// The unarchived version is a zero-sized type.
pub type OpTypeTable = TypeTable<dyn Operation>;
pub type OpTypeTableData = ArchivedTypeTable<dyn Operation>;

// The external table for operation ids, resolved
// from an OpTypeTableData at load time
pub type OpTypeImpls = TypeImpls<dyn Operation>;

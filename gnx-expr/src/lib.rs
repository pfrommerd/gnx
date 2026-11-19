mod attr;
mod dialects;
pub mod eval;
mod expr;
pub mod hexpr;
mod interface;
pub use expr::{Eqn, Expr, SubExpr};
pub use interface::{Caster, InterfaceError, InterfaceImpl};
pub mod lower;
pub mod pass;
// pub mod expr_macro;
// pub mod pretty;
pub mod trace;
pub mod util;
pub mod value;

pub use attr::*;
pub use dialects::*;
pub use eval::{EvalCommit, EvalPlanError, PreparedEval};
pub use hexpr::*;
pub use lower::{CallError, InputId, InputSpec, LoweredExpr, LoweringError};
pub use trace::*;
pub use util::expr_macro;

use std::cmp::{Eq, PartialEq};
use std::cmp::{Ord, PartialOrd};
use std::fmt;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};

// The inventory crate is used to statically register all dialects.
#[doc(hidden)]
pub mod __private {
    pub use inventory;
}

/// The shared kind of a traced value or cell and its immutable value metadata.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    Value(ValueInfo),
    Cell(ValueInfo),
}

pub trait Dialect: Sync {
    fn name(&self) -> &'static str;
    fn create(&self, name: &'static str) -> Option<&'static dyn Operation>;
}

pub trait Operation: std::any::Any + Sync {
    fn name(&self) -> &'static str;
    fn dialect(&self) -> &'static dyn Dialect;

    /// Lower an application to immutable SSA. Region operations must supply
    /// their own rule for passing operands and threading state.
    fn lower(&self, eqn: &HEqn, ctx: &mut lower::LoweringContext<'_>) -> Result<(), LoweringError> {
        ctx.lower_value_apply(eqn)
    }

    /// Check the operation-specific signature after generic operand and region checks.
    fn check_expr(&self, _eqn: &Eqn) -> Result<(), LoweringError> {
        Ok(())
    }

    /// Parent binding of a region's explicit parameter, when it corresponds
    /// to a parent variable. Used to trace structural cell accesses.
    fn region_operand(&self, _eqn: &HEqn, _region: usize, _param: usize) -> Option<Var> {
        None
    }
}

// An Op is an Operation combined with an attribute set.
#[derive(Clone)]
pub struct Op {
    impl_: &'static dyn Operation,
    attrs: AttrMap,
}

impl Debug for Op {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}

impl Hash for Op {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.impl_.dialect().name().hash(state);
        self.impl_.name().hash(state);
        self.attrs.hash(state);
    }
}

impl PartialEq for Op {
    fn eq(&self, other: &Self) -> bool {
        (self.impl_.dialect().name(), self.impl_.name(), &self.attrs)
            == (
                other.impl_.dialect().name(),
                other.impl_.name(),
                &other.attrs,
            )
    }
}

impl Eq for Op {}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.impl_.dialect().name(), self.impl_.name())?;
        if !self.attrs.is_empty() {
            write!(f, "<")?;
            for (i, (k, v)) in self.attrs.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{k}: {v}")?;
            }
            write!(f, ">")?;
        }
        Ok(())
    }
}

impl Op {
    pub fn operation(&self) -> &'static dyn Operation {
        self.impl_
    }

    pub fn new(impl_: &'static dyn Operation) -> Self {
        Op {
            impl_,
            attrs: AttrMap::default(),
        }
    }
    pub fn new_with_attrs(impl_: &'static dyn Operation, attrs: AttrMap) -> Self {
        Op { impl_, attrs }
    }
    pub fn attrs(&self) -> &AttrMap {
        &self.attrs
    }
}

#[derive(Clone, Default, Hash, PartialEq, Eq, Debug)]
pub struct VarScope {
    next_id: usize,
}

impl VarScope {
    pub fn new() -> VarScope {
        VarScope { next_id: 0 }
    }

    pub fn create_var(&mut self) -> Var {
        let v = Var::bind(self.next_id);
        self.next_id += 1;
        v
    }

    pub fn create_var_with_hint(&mut self, name_hint: &'static str) -> Var {
        let v = Var::bind_with_hint(self.next_id, name_hint);
        self.next_id += 1;
        v
    }
}

/// A jaxpr-style variable: bound to an index or 0 for a hole.
#[derive(Clone, Copy, Debug)]
pub struct Var {
    id: usize,
    name_hint: Option<&'static str>,
}

impl Var {
    pub fn hole() -> Self {
        Var {
            id: 0,
            name_hint: None,
        }
    }

    pub fn bind(id: usize) -> Self {
        Var {
            id: id + 1,
            name_hint: None,
        } // 0 is reserved for holes
    }

    pub fn bind_with_hint(id: usize, name_hint: &'static str) -> Self {
        Var {
            id: id + 1,
            name_hint: Some(name_hint),
        }
    }

    pub fn id(&self) -> Option<usize> {
        match self.id {
            0 => None,
            i => Some(i - 1),
        }
    }

    pub fn name_hint(&self) -> Option<&str> {
        self.name_hint
    }
}

impl Hash for Var {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl PartialEq for Var {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Var {}

impl PartialOrd for Var {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Var {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.id.cmp(&other.id)
    }
}

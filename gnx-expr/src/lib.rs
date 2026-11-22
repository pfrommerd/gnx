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

pub use eval::{EvalCommit, EvalPlanError, PreparedEval};
pub use hexpr::*;
pub use lower::{CallError, InputId, InputSpec, LoweredExpr, LoweringError};
pub use trace::*;
pub use util::expr_macro;

use std::any::Any;
use std::cmp::{Eq, PartialEq};
use std::cmp::{Ord, PartialOrd};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::ops::Deref;

// Re-exported for link-time registration of operation interfaces.
#[doc(hidden)]
pub mod __private {
    pub use inventory;
}

/// Access kind of a high-level definition; its VarInfo is stored separately.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    Value,
    Cell,
}

pub trait Dialect: Sync {
    fn name(&self) -> &'static str;
}

/// Object-safe access to a concrete value's hash implementation.
pub trait DynHash {
    fn dyn_hash(&self, state: &mut dyn Hasher);
}

impl<T: Hash> DynHash for T {
    fn dyn_hash(&self, mut state: &mut dyn Hasher) {
        self.hash(&mut state);
    }
}

/// Object-safe equality with a value of the same concrete type.
pub trait DynEq {
    fn dyn_eq(&self, other: &dyn Any) -> bool;
}

impl<T: Any + Eq> DynEq for T {
    fn dyn_eq(&self, other: &dyn Any) -> bool {
        other.downcast_ref::<T>() == Some(self)
    }
}

/// Object-safe cloning of an operation.
pub trait DynClone {
    /// Return an owned clone with the same concrete type and parameters.
    fn dyn_clone(&self) -> Box<dyn Operation>;
}

impl<T: Operation + Clone> DynClone for T {
    fn dyn_clone(&self) -> Box<dyn Operation> {
        Box::new(self.clone())
    }
}

/// An operation and its immutable, operation-specific parameters.
///
/// Concrete operations implement `Clone`, `Eq`, and `Hash` so expressions
/// can copy and compare them through the trait object. `Display` writes the
/// qualified operation name and any parameters, such as `cf::call`.
pub trait Operation: Any + Send + Sync + fmt::Display + DynHash + DynEq + DynClone {
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

    /// Summarize every possible effect of this application on a parent cell.
    /// The default follows captured bindings in every region. Overrides can
    /// reuse that traversal and must include all effects the operation can have.
    fn cell_access(&self, eqn: &HEqn, cell: Var) -> CellAccess {
        eqn.captured_cell_access(cell)
    }
}

impl dyn Operation {
    pub fn downcast_ref<T: Operation>(&self) -> Option<&T> {
        (self as &dyn Any).downcast_ref()
    }
}

/// An owned operation in the expression IR.
pub struct Op(Box<dyn Operation>);

impl Deref for Op {
    type Target = dyn Operation;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

impl From<Op> for Box<dyn Operation> {
    fn from(op: Op) -> Self {
        op.0
    }
}

impl Clone for Op {
    fn clone(&self) -> Self {
        Self(self.0.dyn_clone())
    }
}

impl Debug for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl Hash for Op {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.dialect().name().hash(state);
        self.name().hash(state);
        self.deref().dyn_hash(state);
    }
}

impl PartialEq for Op {
    fn eq(&self, other: &Self) -> bool {
        let op = self.deref();
        let other = other.deref();
        op.type_id() == other.type_id()
            && op.dialect().name() == other.dialect().name()
            && op.name() == other.name()
            && op.dyn_eq(other)
    }
}

impl Eq for Op {}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.deref(), f)
    }
}

impl Op {
    pub fn new(operation: impl Operation) -> Self {
        Self(Box::new(operation))
    }
}

#[derive(Clone, Default, Hash, PartialEq, Eq, Debug)]
pub struct VarScope {
    next_id: usize,
    next_symbol: usize,
}

impl VarScope {
    pub fn new() -> VarScope {
        Self::default()
    }

    /// Allocate a symbol independently of the SSA variable counter.
    pub fn create_symbol(&mut self) -> Symbol {
        let s = Symbol(self.next_symbol);
        self.next_symbol += 1;
        s
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

/// A region-local symbol in metadata, allocated by `VarScope::create_symbol`.
/// A symbol is not an SSA operand and carries no metadata of its own. The
/// dialect owns the meaning of the [`Info`] slots that refer to it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Symbol(pub(crate) usize);

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "s{}", self.0)
    }
}

pub(crate) fn check_symbol_scope<'a>(
    closure: &[Symbol],
    inputs: &[Symbol],
    input_symbols: impl Iterator<Item = &'a Symbol>,
    mut body_symbols: impl Iterator<Item = &'a Symbol>,
) -> Result<(), LoweringError> {
    let mut declared = BTreeSet::new();
    for s in closure.iter().chain(inputs) {
        if !declared.insert(*s) {
            return Err(LoweringError::RegionSignature(
                "duplicate symbol declaration",
            ));
        }
    }
    let mut bound = BTreeSet::new();
    for s in input_symbols {
        if !declared.contains(s) {
            return Err(LoweringError::RegionSignature("undeclared input symbol"));
        }
        bound.insert(*s);
    }
    if inputs.iter().any(|s| !bound.contains(s)) {
        return Err(LoweringError::RegionSignature(
            "symbol input is not bound by input metadata",
        ));
    }
    if body_symbols.any(|s| !declared.contains(s)) {
        return Err(LoweringError::RegionSignature("undeclared metadata symbol"));
    }
    Ok(())
}
pub(crate) fn check_symbol_captures(
    decls: &[Symbol],
    args: &[Symbol],
    defined: impl Fn(Symbol) -> bool,
) -> Result<(), LoweringError> {
    if decls.len() != args.len() {
        return Err(LoweringError::RegionSignature("symbol capture arity"));
    }
    if args.iter().any(|s| !defined(*s)) {
        return Err(LoweringError::RegionSignature("undefined symbol capture"));
    }
    Ok(())
}

/// Infer input symbol substitutions from metadata slots, checking repeated
/// occurrences and explicit captures rather than equating unrelated local IDs.
pub(crate) fn symbol_substitution<'a>(
    closure: &[Symbol],
    captures: &[Symbol],
    inputs: &[Symbol],
    bindings: impl Iterator<Item = (&'a VarInfo, &'a VarInfo)>,
    defined: impl Fn(Symbol) -> bool,
) -> Result<BTreeMap<Symbol, Symbol>, LoweringError> {
    check_symbol_captures(closure, captures, &defined)?;
    let mut map: BTreeMap<_, _> = closure
        .iter()
        .copied()
        .zip(captures.iter().copied())
        .collect();
    for (formal, actual) in bindings {
        let (formal, actual) = (formal.symbols(), actual.symbols());
        if formal.len() != actual.len() {
            return Err(LoweringError::RegionSignature("symbol slot arity mismatch"));
        }
        for (&s, &a) in formal.iter().zip(actual) {
            if !closure.contains(&s) && !inputs.contains(&s) {
                return Err(LoweringError::RegionSignature("undeclared input symbol"));
            }
            if !defined(a) || map.insert(s, a).is_some_and(|prior| prior != a) {
                return Err(LoweringError::RegionSignature(
                    "inconsistent symbol binding",
                ));
            }
        }
    }
    if inputs.iter().any(|s| !map.contains_key(s)) {
        return Err(LoweringError::RegionSignature("unbound symbol input"));
    }
    Ok(map)
}

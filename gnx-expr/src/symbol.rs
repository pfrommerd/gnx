use std::fmt;

use rkyv::{
    Archive, Portable, Serialize,
    bytecheck::{self, CheckBytes},
    rend::u32_le,
};

/// A symbol of an expression, used in metadata. A symbol is not a variable
/// and carries no metadata of its own. The dialect owns the meaning of the
/// [`Info`](crate::Info) slots that refer to it. Symbols are local to
/// their expression.
#[derive(
    Archive,
    Serialize,
    Portable,
    CheckBytes,
    Copy,
    Clone,
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
)]
#[rkyv(as = Self)]
#[bytecheck(crate = bytecheck)]
#[repr(transparent)]
pub struct Symbol(u32_le);
impl Symbol {
    pub fn new(id: u32) -> Self {
        Self(id.into())
    }
    pub fn id(self) -> u32 {
        self.0.to_native()
    }
}

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "s{}", self.id())
    }
}

// Binding the symbols of an expression where it is used.
use std::collections::BTreeMap;

use crate::lower::LoweringError;

// Checks the symbols provided for the closure symbols `decls`.
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
///
/// `closure` and `inputs` are the closure symbols of an expression and the
/// symbols its inputs bind. `captures` are the symbols provided for the
/// closure symbols. Each binding pairs the symbols of the info of a
/// parameter of the expression with those of the info of what it is bound
/// to.
pub(crate) fn symbol_substitution<'a>(
    closure: &[Symbol],
    captures: &[Symbol],
    inputs: &[Symbol],
    bindings: impl Iterator<Item = (&'a [Symbol], &'a [Symbol])>,
    defined: impl Fn(Symbol) -> bool,
) -> Result<BTreeMap<Symbol, Symbol>, LoweringError> {
    check_symbol_captures(closure, captures, &defined)?;
    let mut map: BTreeMap<_, _> = closure
        .iter()
        .copied()
        .zip(captures.iter().copied())
        .collect();
    for (formal, actual) in bindings {
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

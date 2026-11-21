//! Region-local symbols in metadata. A symbol is not an SSA operand. The
//! dialect owns the meaning of its sort and of the ordered metadata slots.
use crate::value::AnyInfo;
use crate::{LoweringError, ValueInfo};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// Allocated by `VarScope::create_symbol`; meaningful only in its region.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Symbol(pub(crate) usize);

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "s{}", self.0)
    }
}

/// Closed, dialect-defined sort of a symbol. Equal sorts do not imply equal symbols.
#[derive(Clone)]
pub struct SymbolInfo(Arc<dyn AnyInfo>);
impl SymbolInfo {
    pub fn new<T: AnyInfo>(sort: T) -> Self {
        Self(Arc::new(sort))
    }
    pub fn downcast<T: AnyInfo>(&self) -> Option<&T> {
        (self.0.as_ref() as &dyn std::any::Any).downcast_ref()
    }
}
impl PartialEq for SymbolInfo {
    fn eq(&self, other: &Self) -> bool {
        self.0.dyn_eq(other.0.as_ref())
    }
}
impl Eq for SymbolInfo {}
impl Hash for SymbolInfo {
    fn hash<H: Hasher>(&self, h: &mut H) {
        self.0.type_id().hash(h);
        self.0.dyn_hash(h);
    }
}
impl fmt::Debug for SymbolInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&*self.0, f)
    }
}
impl fmt::Display for SymbolInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&*self.0, f)
    }
}

pub(crate) fn check_scope<'a>(
    closure: &[(Symbol, SymbolInfo)],
    inputs: &[(Symbol, SymbolInfo)],
    input_symbols: impl Iterator<Item = &'a Symbol>,
    mut body_symbols: impl Iterator<Item = &'a Symbol>,
) -> Result<(), LoweringError> {
    let mut declared = BTreeSet::new();
    for (s, _) in closure.iter().chain(inputs) {
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
    if inputs.iter().any(|(s, _)| !bound.contains(s)) {
        return Err(LoweringError::RegionSignature(
            "symbol input is not bound by input metadata",
        ));
    }
    if body_symbols.any(|s| !declared.contains(s)) {
        return Err(LoweringError::RegionSignature("undeclared metadata symbol"));
    }
    Ok(())
}
pub(crate) fn check_captures<'a>(
    decls: &[(Symbol, SymbolInfo)],
    args: &[Symbol],
    lookup: impl Fn(Symbol) -> Option<&'a SymbolInfo>,
) -> Result<(), LoweringError> {
    if decls.len() != args.len() {
        return Err(LoweringError::RegionSignature("symbol capture arity"));
    }
    for ((_, sort), s) in decls.iter().zip(args) {
        if lookup(*s) != Some(sort) {
            return Err(LoweringError::RegionSignature(
                "undefined symbol capture or incompatible sort",
            ));
        }
    }
    Ok(())
}

/// Infer input symbol substitutions from metadata slots, checking repeated
/// occurrences and explicit captures rather than equating unrelated local IDs.
pub(crate) fn substitution<'a>(
    closure: &[(Symbol, SymbolInfo)],
    captures: &[Symbol],
    inputs: &[(Symbol, SymbolInfo)],
    bindings: impl Iterator<Item = (&'a ValueInfo, &'a ValueInfo)>,
    lookup: impl Fn(Symbol) -> Option<SymbolInfo>,
) -> Result<BTreeMap<Symbol, Symbol>, LoweringError> {
    if closure.len() != captures.len() {
        return Err(LoweringError::RegionSignature("symbol capture arity"));
    }
    for ((_, sort), s) in closure.iter().zip(captures) {
        if lookup(*s).as_ref() != Some(sort) {
            return Err(LoweringError::RegionSignature(
                "incompatible symbol capture",
            ));
        }
    }
    let mut map: BTreeMap<_, _> = closure
        .iter()
        .map(|(s, _)| *s)
        .zip(captures.iter().copied())
        .collect();
    for (formal, actual) in bindings {
        if formal.symbols().len() != actual.symbols().len() {
            return Err(LoweringError::RegionSignature("symbol slot arity mismatch"));
        }
        for (&s, &a) in formal.symbols().iter().zip(actual.symbols()) {
            let sort = closure
                .iter()
                .chain(inputs)
                .find(|(id, _)| *id == s)
                .ok_or(LoweringError::RegionSignature("undeclared input symbol"))?;
            if lookup(a).as_ref() != Some(&sort.1)
                || map.insert(s, a).is_some_and(|prior| prior != a)
            {
                return Err(LoweringError::RegionSignature(
                    "inconsistent symbol binding",
                ));
            }
        }
    }
    if inputs.iter().any(|(s, _)| !map.contains_key(s)) {
        return Err(LoweringError::RegionSignature("unbound symbol input"));
    }
    Ok(map)
}

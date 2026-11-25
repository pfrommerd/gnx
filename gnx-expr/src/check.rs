// The checks made when a root is built, shared by Expr and HExpr.

use std::collections::{HashMap, HashSet};
use std::fmt;

use rkyv::rancor;

use crate::symbol::Symbol;
use crate::var::{Kind, Var, VarInfo};

/// Why an expression could not be built.
#[derive(Debug)]
#[non_exhaustive]
pub enum BuildError {
    /// A variable is used before it is defined in its expression.
    UndefinedVar(Var),
    /// A hole is used as an operand or a result, or stands for a parameter
    /// or a new cell.
    Hole,
    /// A variable is defined more than once in its expression.
    DuplicateVar(Var),
    /// A symbol is neither a closure symbol nor bound by an input.
    UndefinedSymbol(Symbol),
    /// A value is used where a cell is expected, or the other way around.
    Kind {
        var: Var,
        expected: Kind,
    },
    /// A subexpression is not provided one symbol per closure symbol.
    SymbolArity {
        expected: usize,
        actual: usize,
    },
    /// A subexpression is not provided one variable per closure input.
    VarArity {
        expected: usize,
        actual: usize,
    },
    /// An operation or info is of a type that no dialect registers.
    UnknownType {
        dialect: String,
        name: String,
    },
    Serialize(rancor::Error),
}

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UndefinedVar(var) => write!(f, "variable {} is not defined", var.id()),
            Self::Hole => write!(f, "a hole can only discard a value"),
            Self::DuplicateVar(var) => write!(f, "variable {} is defined twice", var.id()),
            Self::UndefinedSymbol(symbol) => write!(f, "symbol {} is not in scope", symbol.id()),
            Self::Kind { var, expected } => {
                write!(f, "variable {} is not a {expected}", var.id())
            }
            Self::SymbolArity { expected, actual } => write!(
                f,
                "subexpression has {expected} closure symbols, but {actual} were provided"
            ),
            Self::VarArity { expected, actual } => write!(
                f,
                "subexpression has {expected} closure inputs, but {actual} were provided"
            ),
            Self::UnknownType { dialect, name } => {
                write!(f, "no dialect registers the type {dialect}.{name}")
            }
            Self::Serialize(err) => write!(f, "serialization failed: {err}"),
        }
    }
}

impl std::error::Error for BuildError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Serialize(err) => Some(err),
            _ => None,
        }
    }
}

// What is in scope at a point of an expression.
pub(crate) struct Scope {
    symbols: HashSet<Symbol>,
    vars: HashMap<Var, Kind>,
}

impl Scope {
    // The scope at the start of an expression: its closure symbols, and
    // the symbols bound by its inputs. Variables are defined in order.
    pub fn new<'a>(closure_symbols: &[Symbol], inputs: impl Iterator<Item = &'a VarInfo>) -> Self {
        let mut symbols: HashSet<Symbol> = closure_symbols.iter().copied().collect();
        for info in inputs {
            symbols.extend(info.symbols());
        }
        Self {
            symbols,
            vars: HashMap::new(),
        }
    }

    // Defines an input, whose info may bind symbols.
    pub fn define_input(&mut self, var: Var, kind: Kind) -> Result<(), BuildError> {
        if var.is_hole() {
            return Err(BuildError::Hole);
        }
        match self.vars.insert(var, kind) {
            Some(_) => Err(BuildError::DuplicateVar(var)),
            None => Ok(()),
        }
    }

    // Defines a closure input or a new cell, whose info must use symbols
    // in scope.
    pub fn define_param(&mut self, var: Var, kind: Kind, info: &VarInfo) -> Result<(), BuildError> {
        self.symbols(info.symbols())?;
        self.define_input(var, kind)
    }

    // Defines the result of an instruction, whose info must use symbols in
    // scope. A hole discards the result.
    pub fn define(&mut self, var: Var, kind: Kind, info: &VarInfo) -> Result<(), BuildError> {
        self.symbols(info.symbols())?;
        if var.is_hole() {
            return Ok(());
        }
        self.define_input(var, kind)
    }

    pub fn symbols(&self, used: &[Symbol]) -> Result<(), BuildError> {
        match used.iter().find(|s| !self.symbols.contains(s)) {
            Some(&symbol) => Err(BuildError::UndefinedSymbol(symbol)),
            None => Ok(()),
        }
    }

    pub fn vars(&self, used: &[Var]) -> Result<(), BuildError> {
        used.iter().try_for_each(|&var| self.kind(var).map(drop))
    }

    pub fn kind(&self, var: Var) -> Result<Kind, BuildError> {
        if var.is_hole() {
            return Err(BuildError::Hole);
        }
        (self.vars.get(&var).copied()).ok_or(BuildError::UndefinedVar(var))
    }

    pub fn expect(&self, var: Var, expected: Kind) -> Result<(), BuildError> {
        if self.kind(var)? == expected {
            Ok(())
        } else {
            Err(BuildError::Kind { var, expected })
        }
    }
}

// Checks that a subexpression is provided as much as its closure declares.
pub(crate) fn check_arity(
    (provided_symbols, closure_symbols): (usize, usize),
    (provided_vars, closure_inputs): (usize, usize),
) -> Result<(), BuildError> {
    if provided_symbols != closure_symbols {
        return Err(BuildError::SymbolArity {
            expected: closure_symbols,
            actual: provided_symbols,
        });
    }
    if provided_vars != closure_inputs {
        return Err(BuildError::VarArity {
            expected: closure_inputs,
            actual: provided_vars,
        });
    }
    Ok(())
}

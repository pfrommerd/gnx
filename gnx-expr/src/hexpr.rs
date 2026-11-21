//! Mutation-aware expressions captured from a trace DAG.
//!
//! Cell reads and writes are structural instructions. Operations summarize their
//! effects on those cells, including effects from nested regions. Lowering requires
//! explicit alias bindings and produces an immutable [`crate::Expr`].

use crate::{Symbol, SymbolInfo};
use std::borrow::Cow;
use std::fmt;
use std::sync::Arc;

use crate::{InputSpec, Kind, LoweredExpr, LoweringError, Op, ValueInfo, Var};

#[derive(Clone, Debug)]
pub struct HSubExpr {
    /// Parent symbols in the order of the child’s `symbol_closure`.
    pub symbol_captures: Vec<Symbol>,
    pub expr: Arc<HExpr>,
    pub captures: Vec<Var>,
}
impl HSubExpr {
    /// Bind the child’s free symbols to IDs in the enclosing region.
    pub fn with_symbol_captures(mut self, captures: Vec<Symbol>) -> Self {
        self.symbol_captures = captures;
        self
    }
    pub fn new(expr: Arc<HExpr>, captures: Vec<Var>) -> Self {
        Self {
            expr,
            captures,
            symbol_captures: vec![],
        }
    }
    pub fn expr(&self) -> &Arc<HExpr> {
        &self.expr
    }
    pub fn closure(&self) -> &[Var] {
        &self.captures
    }

    /// Summarize the region's effects on a parent cell through its captures.
    pub fn captured_cell_access(&self, cell: Var) -> CellAccess {
        self.bound_cell_access(self.expr().closure(), self.closure(), cell)
    }

    /// Summarize the region's effects through explicit inputs in binding order.
    pub fn input_cell_access(&self, operands: &[Var], cell: Var) -> CellAccess {
        self.bound_cell_access(self.expr().inputs(), operands, cell)
    }

    fn bound_cell_access(
        &self,
        parameters: &[(Var, Kind, ValueInfo)],
        operands: &[Var],
        cell: Var,
    ) -> CellAccess {
        parameters
            .iter()
            .zip(operands)
            .filter(|(_, operand)| **operand == cell)
            .filter_map(|((param, ..), _)| self.expr().cell_access(param))
            .fold(CellAccess::Unused, CellAccess::combine)
    }
}

/// A high-level equation with value or cell operands.
#[derive(Clone, Debug)]
pub struct HEqn {
    op: Op,
    subexprs: Vec<HSubExpr>,
    inputs: Vec<Var>,
    outputs: Vec<(Var, Kind, ValueInfo)>,
}
impl HEqn {
    pub fn new(
        op: Op,
        subexprs: Vec<HSubExpr>,
        inputs: Vec<Var>,
        outputs: Vec<(Var, Kind, ValueInfo)>,
    ) -> Self {
        Self {
            op,
            subexprs,
            inputs,
            outputs,
        }
    }
    pub fn op(&self) -> &Op {
        &self.op
    }
    pub fn subexprs(&self) -> &[HSubExpr] {
        &self.subexprs
    }
    pub fn inputs(&self) -> &[Var] {
        &self.inputs
    }
    pub fn outputs(&self) -> &[(Var, Kind, ValueInfo)] {
        &self.outputs
    }

    /// Combine effects through captured bindings in every region.
    pub fn captured_cell_access(&self, cell: Var) -> CellAccess {
        self.subexprs()
            .iter()
            .map(|region| region.captured_cell_access(cell))
            .fold(CellAccess::Unused, CellAccess::combine)
    }
}

/// Summary of possible cell accesses. Immutable value parameters
/// have no cell-access summary.
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

#[derive(Clone, Debug)]
pub enum HInstr {
    Apply(HEqn),
    /// Defines `cell` with the contents metadata of `initial`.
    CreateCell {
        cell: Var,
        initial: Var,
        info: ValueInfo,
    },
    /// Defines `output` with the contents metadata of `cell`.
    ReadCell {
        cell: Var,
        output: Var,
        info: ValueInfo,
    },
    WriteCell {
        cell: Var,
        value: Var,
    },
}
impl HInstr {
    pub fn as_apply(&self) -> Option<&HEqn> {
        match self {
            Self::Apply(eqn) => Some(eqn),
            _ => None,
        }
    }
}

/// Captured high-level expression. Variables distinguish values from cells.
///
/// ```
/// use gnx_expr::{Capture, Generic, InputId, InputSpec, Kind,
///                CellTrace, TraceContext, ObjectTracer, Tracer, ValueInfo};
/// let guard = TraceContext::enter();
/// let info = ValueInfo::new(0usize);
/// let x = Tracer::<Generic>::placeholder(info.clone());
/// let cell = CellTrace::placeholder(info.clone());
/// cell.set(x.trace_ref().clone());
/// let result = cell.get();
/// let captured = Capture::from_context(
///     guard.context(),
///     &[ObjectTracer::from(cell.clone()), ObjectTracer::from(x)],
///     &[ObjectTracer::from(result)],
/// ).unwrap();
/// let spec = InputSpec {
///     closure: vec![],
///     inputs: vec![
///         (InputId::Input(0), Kind::Cell, info.clone()),
///         (InputId::Input(1), Kind::Value, info),
///     ],
/// };
/// let lowered = captured.expr().lower(&spec).unwrap();
/// assert!(lowered.expr.eqns().is_empty()); // Cell actions became value flow.
/// ```
#[derive(Clone, Debug)]
pub struct HExpr {
    pub(crate) symbol_closure: Vec<(Symbol, SymbolInfo)>,
    pub(crate) symbol_inputs: Vec<(Symbol, SymbolInfo)>,
    pub(crate) closure: Vec<(Var, Kind, ValueInfo)>,
    pub(crate) inputs: Vec<(Var, Kind, ValueInfo)>,
    pub(crate) instructions: Vec<HInstr>,
    pub(crate) outputs: Vec<Var>,
    pub(crate) name_hint: Option<Cow<'static, str>>,
}
impl HExpr {
    /// Build a region in its own variable and symbol namespace. Attach any
    /// symbol declarations with `with_symbols` before checking or lowering.
    pub fn new(
        closure: Vec<(Var, Kind, ValueInfo)>,
        inputs: Vec<(Var, Kind, ValueInfo)>,
        instructions: Vec<HInstr>,
        outputs: Vec<Var>,
    ) -> Self {
        Self {
            symbol_closure: vec![],
            symbol_inputs: vec![],
            closure,
            inputs,
            instructions,
            outputs,
            name_hint: None,
        }
    }

    /// Declarations use IDs local to this region, not IDs in a caller.
    pub fn with_symbols(
        mut self,
        closure: Vec<(Symbol, SymbolInfo)>,
        inputs: Vec<(Symbol, SymbolInfo)>,
    ) -> Self {
        self.symbol_closure = closure;
        self.symbol_inputs = inputs;
        self
    }
    pub fn symbol_closure(&self) -> &[(Symbol, SymbolInfo)] {
        &self.symbol_closure
    }
    pub fn symbol_inputs(&self) -> &[(Symbol, SymbolInfo)] {
        &self.symbol_inputs
    }
    pub fn symbol_info(&self, s: Symbol) -> Option<&SymbolInfo> {
        self.symbol_closure
            .iter()
            .chain(&self.symbol_inputs)
            .find_map(|(id, sort)| (*id == s).then_some(sort))
    }
    /// Match symbolic slots across the whole boundary. Input symbols are
    /// inferred from argument metadata, never passed as separate operands.
    pub fn bind_symbols<'a>(
        &'a self,
        symbol_captures: &[Symbol],
        actual_closure: &'a [ValueInfo],
        actual_inputs: &'a [ValueInfo],
        lookup: impl Fn(Symbol) -> Option<SymbolInfo>,
    ) -> Result<std::collections::BTreeMap<Symbol, Symbol>, LoweringError> {
        if self.closure.len() != actual_closure.len() || self.inputs.len() != actual_inputs.len() {
            return Err(LoweringError::CallArity);
        }
        crate::symbol::substitution(
            &self.symbol_closure,
            symbol_captures,
            &self.symbol_inputs,
            self.inputs
                .iter()
                .map(|v| &v.2)
                .zip(actual_inputs)
                .chain(self.closure.iter().map(|v| &v.2).zip(actual_closure)),
            lookup,
        )
    }

    /// Instantiate a child in the namespace of the caller. Keep nested
    /// definitions local; only their attachments belong to this namespace.
    pub(crate) fn substitute_symbols(
        &self,
        bindings: &std::collections::BTreeMap<Symbol, Symbol>,
    ) -> Self {
        let mut body = self.clone();
        let remap = |s| bindings[&s];
        for (_, _, info) in body.closure.iter_mut().chain(&mut body.inputs) {
            *info = info.remap_symbols(remap);
        }
        for instr in &mut body.instructions {
            match instr {
                HInstr::Apply(e) => {
                    for (_, _, info) in &mut e.outputs {
                        *info = info.remap_symbols(remap);
                    }
                    for sub in &mut e.subexprs {
                        for s in &mut sub.symbol_captures {
                            *s = remap(*s);
                        }
                    }
                }
                HInstr::CreateCell { info, .. } | HInstr::ReadCell { info, .. } => {
                    *info = info.remap_symbols(remap)
                }
                HInstr::WriteCell { .. } => {}
            }
        }
        let mut seen = std::collections::BTreeSet::new();
        for list in [&mut body.symbol_closure, &mut body.symbol_inputs] {
            list.retain_mut(|(s, _)| {
                *s = remap(*s);
                seen.insert(*s)
            });
        }
        body
    }
    /// Move the flagged explicit inputs, in order, to the end of the closure.
    /// Transformations use this to turn arguments they don't map or
    /// differentiate into captures of the body.
    ///
    /// # Panics
    /// If `captured` doesn't have one flag per input.
    pub fn capture_inputs(&self, captured: &[bool]) -> HExpr {
        assert_eq!(captured.len(), self.inputs.len(), "one flag per input");
        let mut closure = self.closure.clone();
        let mut inputs = Vec::new();
        for (param, &capture) in self.inputs.iter().zip(captured) {
            if capture {
                closure.push(param.clone());
            } else {
                inputs.push(param.clone());
            }
        }
        let (moved, kept): (Vec<_>, Vec<_>) = self
            .symbol_inputs
            .iter()
            .cloned()
            .partition(|(s, _)| !inputs.iter().any(|v| v.2.symbols().contains(s)));
        HExpr {
            symbol_closure: self.symbol_closure.iter().cloned().chain(moved).collect(),
            symbol_inputs: kept,
            closure,
            inputs,
            instructions: self.instructions.clone(),
            outputs: self.outputs.clone(),
            name_hint: self.name_hint.clone(),
        }
    }

    pub fn with_name_hint(mut self, hint: impl Into<Cow<'static, str>>) -> Self {
        self.name_hint = Some(hint.into());
        self
    }

    pub fn name_hint(&self) -> Option<&str> {
        self.name_hint.as_deref()
    }

    pub fn pretty(&self) -> crate::util::pretty::PrettyHExpr<'_> {
        crate::util::pretty::PrettyHExpr::new(self)
    }

    pub fn closure(&self) -> &[(Var, Kind, ValueInfo)] {
        &self.closure
    }
    pub fn inputs(&self) -> &[(Var, Kind, ValueInfo)] {
        &self.inputs
    }
    pub fn instructions(&self) -> &[HInstr] {
        &self.instructions
    }
    pub fn outputs(&self) -> &[Var] {
        &self.outputs
    }
    /// Symbols used by closure metadata, instructions and nested attachments.
    pub(crate) fn body_symbols(&self) -> impl Iterator<Item = &Symbol> {
        self.closure
            .iter()
            .flat_map(|v| v.2.symbols())
            .chain(self.instructions.iter().flat_map(|i| {
                let cell = match i {
                    HInstr::ReadCell { info, .. } | HInstr::CreateCell { info, .. } => Some(info),
                    _ => None,
                };
                let eqn = i.as_apply();
                cell.into_iter()
                    .flat_map(ValueInfo::symbols)
                    .chain(
                        eqn.into_iter()
                            .flat_map(HEqn::outputs)
                            .flat_map(|v| v.2.symbols()),
                    )
                    .chain(
                        eqn.into_iter()
                            .flat_map(HEqn::subexprs)
                            .flat_map(|r| &r.symbol_captures),
                    )
            }))
    }
    /// Verify declarations and lexical symbol bindings throughout the tree.
    pub fn check_symbols(&self) -> Result<(), LoweringError> {
        crate::symbol::check_scope(
            &self.symbol_closure,
            &self.symbol_inputs,
            self.inputs.iter().flat_map(|v| v.2.symbols()),
            self.body_symbols(),
        )?;
        for sub in self
            .instructions
            .iter()
            .filter_map(HInstr::as_apply)
            .flat_map(HEqn::subexprs)
        {
            crate::symbol::check_captures(sub.expr.symbol_closure(), &sub.symbol_captures, |s| {
                self.symbol_info(s)
            })?;
            sub.expr.check_symbols()?;
        }
        Ok(())
    }
    /// Find a variable's kind at its definition site.
    pub fn kind(&self, var: &Var) -> Option<Kind> {
        self.definition(var).map(|(kind, _)| kind)
    }
    pub fn definition(&self, var: &Var) -> Option<(Kind, &ValueInfo)> {
        self.closure
            .iter()
            .chain(&self.inputs)
            .find_map(|(defined, kind, info)| (defined == var).then_some((*kind, info)))
            .or_else(|| {
                self.instructions
                    .iter()
                    .rev()
                    .find_map(|instr| match instr {
                        HInstr::Apply(eqn) => {
                            eqn.outputs().iter().find_map(|(defined, kind, info)| {
                                (defined == var).then_some((*kind, info))
                            })
                        }
                        HInstr::CreateCell { cell, info, .. } if cell == var => {
                            Some((Kind::Cell, info))
                        }
                        HInstr::ReadCell { output, info, .. } if output == var => {
                            Some((Kind::Value, info))
                        }
                        _ => None,
                    })
            })
    }

    /// Derive access to a cell from structural instructions and operation effects.
    /// Returning a handle alone has no effect. Values return `None`.
    pub fn cell_access(&self, var: &Var) -> Option<CellAccess> {
        if !matches!(self.kind(var), Some(Kind::Cell)) {
            return None;
        }
        Some(
            self.instructions
                .iter()
                .fold(CellAccess::Unused, |access, instr| {
                    let next = match instr {
                        HInstr::ReadCell { cell, .. } if cell == var => CellAccess::Read,
                        HInstr::WriteCell { cell, .. } if cell == var => CellAccess::Write,
                        HInstr::Apply(eqn) => eqn.op().cell_access(eqn, *var),
                        _ => CellAccess::Unused,
                    };
                    access.combine(next)
                }),
        )
    }

    /// Resolve cell aliases and replace cell actions with immutable SSA flow.
    pub fn lower(&self, spec: &InputSpec) -> Result<LoweredExpr, LoweringError> {
        crate::lower::lower(self, spec)
    }
}

/// Formats the high-level IR, hoisting large nested bodies. Metadata is displayed
/// as a value, not serialized back to Rust source.
impl fmt::Display for HExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        crate::util::pretty::PrettyVarScope::new().fmt_hexpr(self, f)
    }
}

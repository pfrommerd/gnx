use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::ops::Deref;

use rkyv::tuple::ArchivedTuple3;

use super::{
    CellAccess, HEqnData, HEqnSource, HExprBuilder, HExprData, HExprHandle, HExprRef, HInstrData,
    HInstrSource, HParamSource, HRootExpr, HSubExprBuilder, HSubExprData,
};
use crate::lower::LoweringError;
use crate::op::OpRef;
use crate::symbol::Symbol;
use crate::symbol::symbol_substitution;
use crate::util::{DebugList, TypeName};
use crate::var::{Info, Kind, Var, VarInfo, VarInfoData, VarInfoRef};

// Views into a root high-level expression. As for Expr, views are Copy,
// and their accessors return data that lives as long as the root.

type HParamData = ArchivedTuple3<Var, Kind, VarInfoData>;

fn param<'a>(root: &'a HRootExpr, var: Var, kind: Kind, info: &'a VarInfoData) -> HParam<'a> {
    HParam {
        var,
        kind,
        // SAFETY: callers pass an info of the data of root
        info: unsafe { root.info(info) },
    }
}

fn params<'a>(
    root: &'a HRootExpr,
    params: &'a [HParamData],
) -> impl ExactSizeIterator<Item = HParam<'a>> + DoubleEndedIterator + Clone {
    (params.iter()).map(move |data| param(root, data.0, data.1, &data.2))
}

type Copied = HashMap<*const HExprData, HExprRef>;

/// View of a high-level expression. Variables distinguish values from cells.
#[derive(Clone, Copy)]
pub struct HExpr<'a> {
    root: &'a HRootExpr,
    expr: &'a HExprData,
}

impl<'a> HExpr<'a> {
    // SAFETY: expr must be part of the data of root
    pub(super) unsafe fn new(root: &'a HRootExpr, expr: &'a HExprData) -> Self {
        HExpr { root, expr }
    }

    /// The root whose data this expression is stored in.
    pub fn root(self) -> &'a HRootExpr {
        self.root
    }

    /// The name the expression was given, for when it is shown.
    pub fn name_hint(self) -> Option<&'a str> {
        self.expr.name_hint.as_ref().map(|name| name.as_str())
    }

    // Identifies the stored expression
    pub(crate) fn data_ptr(self) -> *const () {
        std::ptr::from_ref(self.expr).cast()
    }

    /// The symbols provided by the enclosing expression.
    pub fn closure_symbols(self) -> &'a [Symbol] {
        &self.expr.closure_symbols
    }

    /// The variables provided by the enclosing expression.
    pub fn closure_inputs(
        self,
    ) -> impl ExactSizeIterator<Item = HParam<'a>> + DoubleEndedIterator + Clone {
        params(self.root, &self.expr.closure_inputs)
    }

    pub fn inputs(self) -> impl ExactSizeIterator<Item = HParam<'a>> + DoubleEndedIterator + Clone {
        params(self.root, &self.expr.inputs)
    }

    /// The instructions, in order.
    pub fn instrs(self) -> impl ExactSizeIterator<Item = HInstr<'a>> + DoubleEndedIterator + Clone {
        let root = self.root;
        (self.expr.instrs.iter()).map(move |instr| HInstr::new(root, instr))
    }

    /// The results in order, including any inputs or repeated variables.
    pub fn outputs(self) -> &'a [Var] {
        &self.expr.outputs
    }

    /// The symbols bound by the inputs: those in their infos that are
    /// not closure symbols, in order of first appearance.
    pub fn input_symbols(self) -> Vec<Symbol> {
        let mut symbols = Vec::new();
        for &symbol in self.inputs().flat_map(|param| param.info().symbols()) {
            if !self.closure_symbols().contains(&symbol) && !symbols.contains(&symbol) {
                symbols.push(symbol);
            }
        }
        symbols
    }

    /// Whether `symbol` is in scope: a closure symbol, or bound by an input.
    pub fn declares_symbol(self, symbol: Symbol) -> bool {
        self.closure_symbols().contains(&symbol)
            || (self.inputs()).any(|param| param.info().symbols().contains(&symbol))
    }

    /// Find a variable at its definition site.
    pub fn definition(self, var: Var) -> Option<HParam<'a>> {
        if var.is_hole() {
            return None;
        }
        (self.closure_inputs().chain(self.inputs()))
            .find(|param| param.var == var)
            .or_else(|| {
                (self.instrs().rev())
                    .flat_map(|instr| instr.defines())
                    .find(|param| param.var == var)
            })
    }

    /// Find a variable's kind at its definition site.
    pub fn kind(self, var: Var) -> Option<Kind> {
        self.definition(var).map(|param| param.kind)
    }

    /// Derive access to a cell from structural instructions and operation
    /// effects. Returning a cell alone has no effect. Values return `None`.
    pub fn cell_access(self, var: Var) -> Option<CellAccess> {
        if self.kind(var) != Some(Kind::Cell) {
            return None;
        }
        Some(self.instrs().fold(CellAccess::Unused, |access, instr| {
            let next = match instr {
                HInstr::ReadCell { cell, .. } if cell == var => CellAccess::Read,
                HInstr::WriteCell { cell, .. } if cell == var => CellAccess::Write,
                HInstr::Apply(eqn) => eqn.op().get().cell_access(eqn, var),
                _ => CellAccess::Unused,
            };
            access.combine(next)
        }))
    }

    /// Whether the expression may write a cell it is given: one of its
    /// inputs or closure inputs.
    pub fn writes_parameters(self) -> bool {
        (self.closure_inputs().chain(self.inputs()))
            .filter_map(|param| self.cell_access(param.var()))
            .any(|access| matches!(access, CellAccess::Write | CellAccess::ReadWrite))
    }

    /// Copies the expression out of its root, as an expression under
    /// construction. Variables and symbols keep their identity.
    pub fn to_builder(self) -> HExprBuilder {
        self.copy(&mut HashMap::new())
    }

    // `copied` holds the internal subexpressions copied so far, so that
    // one used several times is still stored once.
    fn copy(self, copied: &mut Copied) -> HExprBuilder {
        let mut copy = HExprBuilder::from_parts(
            self.closure_symbols().to_vec(),
            self.closure_inputs().map(HParam::to_source).collect(),
            self.inputs().map(HParam::to_source).collect(),
            self.instrs().map(|instr| instr.copy(copied)).collect(),
            self.outputs().to_vec(),
        );
        copy.name_hint = self.name_hint().map(str::to_owned);
        copy
    }
}

/// View of an instruction of a high-level expression
#[derive(Clone, Copy, Debug)]
pub enum HInstr<'a> {
    Apply(HEqn<'a>),
    /// Defines `cell`, holding the value `initial`. `info` is the info
    /// of its contents.
    CreateCell {
        cell: Var,
        initial: Var,
        info: VarInfoRef<'a>,
    },
    /// Defines `output` as the value `cell` holds, with `info`.
    ReadCell {
        cell: Var,
        output: Var,
        info: VarInfoRef<'a>,
    },
    WriteCell {
        cell: Var,
        value: Var,
    },
}

impl<'a> HInstr<'a> {
    fn new(root: &'a HRootExpr, instr: &'a HInstrData) -> Self {
        // SAFETY: instr is part of the data of root
        let info = |info| unsafe { root.info(info) };
        match instr {
            HInstrData::Apply(eqn) => Self::Apply(HEqn { root, eqn }),
            HInstrData::CreateCell {
                cell,
                initial,
                info: i,
            } => Self::CreateCell {
                cell: *cell,
                initial: *initial,
                info: info(i),
            },
            HInstrData::ReadCell {
                cell,
                output,
                info: i,
            } => Self::ReadCell {
                cell: *cell,
                output: *output,
                info: info(i),
            },
            HInstrData::WriteCell { cell, value } => Self::WriteCell {
                cell: *cell,
                value: *value,
            },
        }
    }

    pub fn as_apply(self) -> Option<HEqn<'a>> {
        match self {
            Self::Apply(eqn) => Some(eqn),
            _ => None,
        }
    }

    /// The variables this instruction defines.
    pub fn defines(self) -> impl Iterator<Item = HParam<'a>> {
        let (eqn, other) = match self {
            Self::Apply(eqn) => (Some(eqn), None),
            Self::CreateCell { cell, info, .. } => (None, Some((cell, Kind::Cell, info))),
            Self::ReadCell { output, info, .. } => (None, Some((output, Kind::Value, info))),
            Self::WriteCell { .. } => (None, None),
        };
        (eqn.into_iter().flat_map(HEqn::outputs)).chain(other.map(|(var, kind, info)| HParam {
            var,
            kind,
            info,
        }))
    }

    /// Copies the instruction out of its root.
    pub fn to_source(self) -> HInstrSource {
        self.copy(&mut HashMap::new())
    }

    fn copy(self, copied: &mut Copied) -> HInstrSource {
        match self {
            Self::Apply(eqn) => HInstrSource::Apply(eqn.copy(copied)),
            Self::CreateCell {
                cell,
                initial,
                info,
            } => HInstrSource::CreateCell {
                cell,
                initial,
                info: info.to_owned(),
            },
            Self::ReadCell { cell, output, info } => HInstrSource::ReadCell {
                cell,
                output,
                info: info.to_owned(),
            },
            Self::WriteCell { cell, value } => HInstrSource::WriteCell { cell, value },
        }
    }
}

/// View of an equation of a high-level expression, whose operands are
/// values or cells
#[derive(Clone, Copy)]
pub struct HEqn<'a> {
    root: &'a HRootExpr,
    eqn: &'a HEqnData,
}

impl<'a> HEqn<'a> {
    pub fn op(self) -> OpRef<'a> {
        // SAFETY: the equation is part of the data of root
        unsafe { self.root.op(&self.eqn.op) }
    }

    pub fn inputs(self) -> &'a [Var] {
        &self.eqn.inputs
    }

    /// The variables this equation defines.
    pub fn outputs(
        self,
    ) -> impl ExactSizeIterator<Item = HParam<'a>> + DoubleEndedIterator + Clone {
        params(self.root, &self.eqn.outputs)
    }

    pub fn subexprs(
        self,
    ) -> impl ExactSizeIterator<Item = HSubExpr<'a>> + DoubleEndedIterator + Clone {
        let root = self.root;
        (self.eqn.subexprs.iter()).map(move |subexpr| HSubExpr { root, subexpr })
    }

    /// Combine the effects on `cell` of every subexpression it is
    /// provided to.
    pub fn captured_cell_access(self, cell: Var) -> CellAccess {
        self.subexprs()
            .map(|sub| sub.captured_cell_access(cell))
            .fold(CellAccess::Unused, CellAccess::combine)
    }

    /// Copies the equation out of its root.
    pub fn to_source(self) -> HEqnSource {
        self.copy(&mut HashMap::new())
    }

    fn copy(self, copied: &mut Copied) -> HEqnSource {
        HEqnSource::new(
            self.op().to_owned(),
            self.subexprs().map(|sub| sub.copy(copied)).collect(),
            self.inputs().to_vec(),
            self.outputs().map(HParam::to_source).collect(),
        )
    }
}

/// View of a subexpression of an equation: a reference to another
/// high-level expression, plus the symbols and vars provided to it
#[derive(Clone, Copy)]
pub struct HSubExpr<'a> {
    root: &'a HRootExpr,
    subexpr: &'a HSubExprData,
}

impl<'a> HSubExpr<'a> {
    pub fn expr(self) -> HExpr<'a> {
        self.root.expr_at(self.subexpr.expr)
    }

    /// A handle to the expression, if it is stored in another root.
    pub fn external(self) -> Option<&'a HExprHandle> {
        self.root.external_at(self.subexpr.expr)
    }

    /// The symbols provided for the closure symbols of the expression.
    pub fn provided_symbols(self) -> &'a [Symbol] {
        &self.subexpr.provided_symbols
    }

    /// The variables provided for the closure inputs of the expression.
    pub fn provided_vars(self) -> &'a [Var] {
        &self.subexpr.provided_vars
    }

    /// Summarize the effects of the expression on a cell of the enclosing
    /// expression, through the variables provided for its closure.
    pub fn captured_cell_access(self, cell: Var) -> CellAccess {
        self.bound_cell_access(self.expr().closure_inputs(), self.provided_vars(), cell)
    }

    /// Summarize the effects of the expression on a cell of the enclosing
    /// expression, when `operands` are passed as its inputs, in order.
    pub fn input_cell_access(self, operands: &[Var], cell: Var) -> CellAccess {
        self.bound_cell_access(self.expr().inputs(), operands, cell)
    }

    fn bound_cell_access(
        self,
        params: impl Iterator<Item = HParam<'a>>,
        operands: &[Var],
        cell: Var,
    ) -> CellAccess {
        let expr = self.expr();
        params
            .zip(operands)
            .filter(|(_, operand)| **operand == cell)
            .filter_map(|(param, _)| expr.cell_access(param.var))
            .fold(CellAccess::Unused, CellAccess::combine)
    }

    /// A reference to the expression for use in an expression under
    /// construction: a handle if it is stored in another root, and
    /// otherwise a copy.
    pub fn to_ref(self) -> HExprRef {
        self.copy_ref(&mut HashMap::new())
    }

    fn copy_ref(self, copied: &mut Copied) -> HExprRef {
        if let Some(handle) = self.external() {
            return handle.clone().into();
        }
        let expr = self.expr();
        if let Some(copy) = copied.get(&std::ptr::from_ref(expr.expr)) {
            return copy.clone();
        }
        let copy = expr.copy(copied).finish();
        copied.insert(expr.expr, copy.clone());
        copy
    }

    fn copy(self, copied: &mut Copied) -> HSubExprBuilder {
        HSubExprBuilder::new(self.copy_ref(copied))
            .with_symbols(self.provided_symbols().iter().copied())
            .with_vars(self.provided_vars().iter().copied())
    }
}

/// View of a variable at its definition: the variable, whether it is a
/// value or a cell, and its info. The info of a cell is that of its contents.
#[derive(Clone, Copy)]
pub struct HParam<'a> {
    var: Var,
    kind: Kind,
    info: VarInfoRef<'a>,
}

impl<'a> HParam<'a> {
    pub fn var(self) -> Var {
        self.var
    }

    pub fn kind(self) -> Kind {
        self.kind
    }

    /// The info. Unlike `Deref`, the result outlives this view.
    pub fn info(self) -> &'a dyn Info {
        self.info.get()
    }

    /// The info, as a reference that can be compared and copied out.
    pub fn info_ref(self) -> VarInfoRef<'a> {
        self.info
    }

    /// The dialect and name the type of the info is registered as.
    pub fn type_name(self) -> TypeName {
        self.info.type_name()
    }

    /// Copies the definition out of its root.
    pub fn to_source(self) -> HParamSource {
        (self.var, self.kind, self.info.to_owned())
    }
}

impl<'a> Deref for HParam<'a> {
    type Target = dyn Info;

    fn deref(&self) -> &Self::Target {
        self.info.get()
    }
}

impl fmt::Debug for HExpr<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HExpr")
            .field("closure_symbols", &self.closure_symbols())
            .field("closure_inputs", &DebugList(self.closure_inputs()))
            .field("inputs", &DebugList(self.inputs()))
            .field("instrs", &DebugList(self.instrs()))
            .field("outputs", &self.outputs())
            .finish()
    }
}

impl fmt::Debug for HEqn<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HEqn")
            .field("op", &self.op())
            .field("inputs", &self.inputs())
            .field("outputs", &DebugList(self.outputs()))
            .field("subexprs", &DebugList(self.subexprs()))
            .finish()
    }
}

impl fmt::Debug for HSubExpr<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HSubExpr")
            .field("expr", &self.expr())
            .field("provided_symbols", &self.provided_symbols())
            .field("provided_vars", &self.provided_vars())
            .finish()
    }
}

impl fmt::Debug for HParam<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {} {:?}", self.var, self.kind, self.info)
    }
}

// Binding and moving the symbols of an expression.
impl<'a> HExpr<'a> {
    /// Match symbolic slots across the whole boundary. Input symbols are
    /// inferred from argument metadata, never passed as separate operands.
    pub fn bind_symbols(
        self,
        symbol_captures: &[Symbol],
        actual_closure: &[VarInfo],
        actual_inputs: &[VarInfo],
        defined: impl Fn(Symbol) -> bool,
    ) -> Result<BTreeMap<Symbol, Symbol>, LoweringError> {
        if self.closure_inputs().len() != actual_closure.len()
            || self.inputs().len() != actual_inputs.len()
        {
            return Err(LoweringError::CallArity);
        }
        symbol_substitution(
            self.closure_symbols(),
            symbol_captures,
            &self.input_symbols(),
            (self.inputs().chain(self.closure_inputs()))
                .map(|param| param.info().symbols())
                .zip((actual_inputs.iter().chain(actual_closure)).map(|info| info.symbols())),
            defined,
        )
    }

    /// Instantiate the expression in the namespace of a caller. Nested
    /// definitions stay local; only what is provided to them belongs to
    /// this namespace.
    pub(crate) fn substitute_symbols(self, bindings: &BTreeMap<Symbol, Symbol>) -> HExprBuilder {
        let mut body = self.to_builder();
        let remap = |s| bindings[&s];
        for (_, _, info) in body.closure_inputs.iter_mut().chain(&mut body.inputs) {
            *info = info.remap_symbols(remap);
        }
        for instr in &mut body.instrs {
            match instr {
                HInstrSource::Apply(e) => {
                    for (_, _, info) in &mut e.outputs {
                        *info = info.remap_symbols(remap);
                    }
                    for sub in &mut e.subexprs {
                        for s in &mut sub.provided_symbols {
                            *s = remap(*s);
                        }
                    }
                }
                HInstrSource::CreateCell { info, .. } | HInstrSource::ReadCell { info, .. } => {
                    *info = info.remap_symbols(remap)
                }
                HInstrSource::WriteCell { .. } => {}
            }
        }
        let mut seen = std::collections::BTreeSet::new();
        body.closure_symbols.retain_mut(|s| {
            *s = remap(*s);
            seen.insert(*s)
        });
        body.reserve();
        body
    }

    /// Move the flagged explicit inputs, in order, to the end of the closure.
    /// Transformations use this to turn arguments they don't map or
    /// differentiate into captures of the body. Symbols that the remaining
    /// inputs no longer bind become closure symbols.
    ///
    /// # Panics
    /// If `captured` doesn't have one flag per input.
    pub fn capture_inputs(self, captured: &[bool]) -> HExprBuilder {
        assert_eq!(captured.len(), self.inputs().len(), "one flag per input");
        let input_symbols = self.input_symbols();
        let mut body = self.to_builder();
        let mut inputs = Vec::new();
        for (param, &capture) in std::mem::take(&mut body.inputs).into_iter().zip(captured) {
            if capture {
                body.closure_inputs.push(param);
            } else {
                inputs.push(param);
            }
        }
        let moved = (input_symbols.into_iter())
            .filter(|s| !inputs.iter().any(|(_, _, info)| info.symbols().contains(s)));
        body.closure_symbols.extend(moved);
        body.inputs = inputs;
        body
    }
}

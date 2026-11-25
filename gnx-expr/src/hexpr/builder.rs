use std::collections::HashSet;
use std::rc::Rc;
use std::sync::Arc;

use rkyv::{Archive, Serialize, with::Skip};

use super::{HExpr, HExprRef, HRootExpr};
use crate::check::{BuildError, Scope, check_arity};
use crate::op::Op;
use crate::root::{self, ExprSource, Ref};
use crate::symbol::Symbol;
use crate::var::{Kind, Var, VarInfo, VarScope};

// As for Expr, the builders double as the unarchived form of the data.

/// A variable at its definition: whether it is a value or a cell, and its
/// info. The info of a cell is that of its contents.
pub type HParamSource = (Var, Kind, VarInfo);

/// A high-level expression that is being constructed: the owned, mutable
/// form of a high-level expression. Variables and symbols are local to the
/// expression: those of an enclosing expression are provided explicitly,
/// where the expression is used as a subexpression.
///
/// ```ignore
/// let mut b = HRootExpr::builder();
/// let x = b.input(Kind::Value, info.clone());
/// let cell = b.create_cell(x, info.clone());
/// let [y] = b.eqn(op).input(cell).output(Kind::Value, info.clone()).done();
/// b.write_cell(cell, y);
/// let z = b.read_cell(cell, info);
/// b.output(z);
/// let root = b.build()?;
/// ```
#[derive(Archive, Serialize, Clone, Debug, Default)]
#[rkyv(archived = HExprData)]
pub struct HExprBuilder {
    pub(crate) closure_symbols: Vec<Symbol>,
    pub(crate) closure_inputs: Vec<HParamSource>,
    pub(crate) inputs: Vec<HParamSource>,
    pub(crate) instrs: Vec<HInstrSource>,
    pub(crate) outputs: Vec<Var>,
    pub(crate) name_hint: Option<String>,
    #[rkyv(with = Skip)]
    pub(crate) scope: VarScope,
}

/// An instruction that is being constructed.
#[derive(Archive, Serialize, Clone, Debug)]
#[rkyv(archived = HInstrData)]
pub enum HInstrSource {
    Apply(HEqnSource),
    /// Defines `cell`, holding the value `initial`. `info` is the info
    /// of its contents.
    CreateCell {
        cell: Var,
        initial: Var,
        info: VarInfo,
    },
    /// Defines `output` as the value `cell` holds, with `info`.
    ReadCell {
        cell: Var,
        output: Var,
        info: VarInfo,
    },
    WriteCell {
        cell: Var,
        value: Var,
    },
}

impl HInstrSource {
    pub fn as_apply(&self) -> Option<&HEqnSource> {
        match self {
            Self::Apply(eqn) => Some(eqn),
            _ => None,
        }
    }

    // The variables this instruction defines
    fn defines(&self) -> impl Iterator<Item = (Var, Kind, &VarInfo)> {
        let (eqn, other) = match self {
            Self::Apply(eqn) => (Some(eqn), None),
            Self::CreateCell { cell, info, .. } => (None, Some((*cell, Kind::Cell, info))),
            Self::ReadCell { output, info, .. } => (None, Some((*output, Kind::Value, info))),
            Self::WriteCell { .. } => (None, None),
        };
        (eqn.into_iter().flat_map(|eqn| &eqn.outputs))
            .map(|(var, kind, info)| (*var, *kind, info))
            .chain(other)
    }
}

impl HExprBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Assembles an expression from definitions that use variables and
    /// symbols allocated elsewhere, such as in a `VarScope`.
    pub fn from_parts(
        closure_symbols: Vec<Symbol>,
        closure_inputs: Vec<HParamSource>,
        inputs: Vec<HParamSource>,
        instrs: Vec<HInstrSource>,
        outputs: Vec<Var>,
    ) -> Self {
        let mut expr = Self {
            closure_symbols,
            closure_inputs,
            inputs,
            instrs,
            outputs,
            name_hint: None,
            scope: VarScope::new(),
        };
        expr.reserve();
        expr
    }

    // The variables defined in this expression
    fn defs(&self) -> impl Iterator<Item = (Var, Kind, &VarInfo)> {
        (self.closure_inputs.iter().chain(&self.inputs))
            .map(|(var, kind, info)| (*var, *kind, info))
            .chain(self.instrs.iter().flat_map(|instr| instr.defines()))
    }

    // Makes the variables and symbols allocated from here on distinct from
    // every one in use.
    pub(crate) fn reserve(&mut self) {
        let vars: Vec<_> = self.defs().map(|(var, ..)| var).collect();
        let symbols: Vec<_> = (self.closure_symbols.iter().copied())
            .chain(
                self.defs()
                    .flat_map(|(.., info)| info.symbols().iter().copied()),
            )
            .collect();
        self.scope.reserve(vars, symbols);
    }

    /// Allocates a variable without defining it.
    pub fn fresh_var(&mut self) -> Var {
        self.scope.create_var()
    }

    /// Names the expression, for when it is shown.
    pub fn with_name_hint(mut self, name: impl Into<String>) -> Self {
        self.name_hint = Some(name.into());
        self
    }

    pub(crate) fn with_name_hint_opt(mut self, name: Option<String>) -> Self {
        self.name_hint = name;
        self
    }

    pub fn set_name_hint(&mut self, name: Option<String>) {
        self.name_hint = name;
    }

    pub fn name_hint(&self) -> Option<&str> {
        self.name_hint.as_deref()
    }

    /// Allocates a symbol without declaring it. It comes into scope once
    /// it appears in the info of an input, which binds it.
    pub fn symbol(&mut self) -> Symbol {
        self.scope.create_symbol()
    }

    /// Declares a symbol provided by the enclosing expression.
    pub fn closure_symbol(&mut self) -> Symbol {
        let symbol = self.symbol();
        self.closure_symbols.push(symbol);
        symbol
    }

    /// Declares a variable provided by the enclosing expression.
    pub fn closure_input(&mut self, kind: Kind, info: VarInfo) -> Var {
        let var = self.fresh_var();
        self.closure_inputs.push((var, kind, info));
        var
    }

    /// Declares an input. Symbols in `info` that are not closure symbols
    /// are bound by this input.
    pub fn input(&mut self, kind: Kind, info: VarInfo) -> Var {
        let var = self.fresh_var();
        self.inputs.push((var, kind, info));
        var
    }

    /// Appends an equation that applies `op`, to be filled in through
    /// the returned builder.
    pub fn eqn(&mut self, op: Op) -> HEqnBuilder<'_> {
        self.instrs.push(HInstrSource::Apply(HEqnSource::new(
            op,
            vec![],
            vec![],
            vec![],
        )));
        HEqnBuilder { expr: self }
    }

    /// Appends an instruction whose definitions were allocated with
    /// [`HExprBuilder::fresh_var`].
    pub fn push(&mut self, instr: HInstrSource) -> &mut Self {
        self.instrs.push(instr);
        self
    }

    /// Appends the creation of a cell that holds the value `initial`,
    /// and returns the cell. `info` is the info of its contents.
    pub fn create_cell(&mut self, initial: Var, info: VarInfo) -> Var {
        let cell = self.fresh_var();
        self.instrs.push(HInstrSource::CreateCell {
            cell,
            initial,
            info,
        });
        cell
    }

    /// Appends a read of `cell`, and returns the value it holds at that
    /// point. `info` is the info of that value.
    pub fn read_cell(&mut self, cell: Var, info: VarInfo) -> Var {
        let output = self.fresh_var();
        self.instrs
            .push(HInstrSource::ReadCell { cell, output, info });
        output
    }

    /// Appends a write of the value `value` to `cell`.
    pub fn write_cell(&mut self, cell: Var, value: Var) -> &mut Self {
        self.instrs.push(HInstrSource::WriteCell { cell, value });
        self
    }

    /// Appends a result of the expression. A variable can be returned
    /// any number of times.
    pub fn output(&mut self, var: Var) -> &mut Self {
        self.outputs.push(var);
        self
    }

    pub fn outputs(&mut self, vars: impl IntoIterator<Item = Var>) -> &mut Self {
        self.outputs.extend(vars);
        self
    }

    /// Whether `symbol` is in scope: a closure symbol, or bound by an input.
    pub fn declares_symbol(&self, symbol: Symbol) -> bool {
        self.closure_symbols.contains(&symbol)
            || (self.inputs.iter()).any(|(_, _, info)| info.symbols().contains(&symbol))
    }

    /// Find a variable's kind and info at its definition.
    pub fn definition(&self, var: Var) -> Option<(Kind, &VarInfo)> {
        (self.defs().collect::<Vec<_>>().into_iter().rev()).find_map(|(defined, kind, info)| {
            (defined == var && !var.is_hole()).then_some((kind, info))
        })
    }

    /// Freezes the expression, for use as a subexpression. Clones of the
    /// result refer to the same expression, which is then stored once.
    #[must_use]
    pub fn finish(self) -> HExprRef {
        self.into()
    }

    /// Checks the expression and everything it refers to, and builds a
    /// root of it.
    pub fn build(self) -> Result<Arc<HRootExpr>, BuildError> {
        root::build(self)
    }
}

/// An equation that is being constructed: an operation applied to values
/// or cells of the expression, defining a variable for each output.
#[derive(Archive, Serialize, Clone, Debug)]
#[rkyv(archived = HEqnData)]
pub struct HEqnSource {
    pub(crate) op: Op,
    pub(crate) inputs: Vec<Var>,
    pub(crate) outputs: Vec<HParamSource>,
    pub(crate) subexprs: Vec<HSubExprBuilder>,
}

impl HEqnSource {
    pub fn new(
        op: Op,
        subexprs: Vec<HSubExprBuilder>,
        inputs: Vec<Var>,
        outputs: Vec<HParamSource>,
    ) -> Self {
        Self {
            op,
            inputs,
            outputs,
            subexprs,
        }
    }
    pub fn op(&self) -> &Op {
        &self.op
    }
    pub fn inputs(&self) -> &[Var] {
        &self.inputs
    }
    pub fn outputs(&self) -> &[HParamSource] {
        &self.outputs
    }
    pub fn subexprs(&self) -> &[HSubExprBuilder] {
        &self.subexprs
    }
}

/// Fills in the equation appended by [`HExprBuilder::eqn`].
pub struct HEqnBuilder<'a> {
    expr: &'a mut HExprBuilder,
}

impl HEqnBuilder<'_> {
    // The expression is borrowed for as long as this builder exists,
    // so the equation stays the last instruction.
    fn eqn(&mut self) -> &mut HEqnSource {
        match self.expr.instrs.last_mut() {
            Some(HInstrSource::Apply(eqn)) => eqn,
            _ => unreachable!("eqn appended an equation"),
        }
    }

    pub fn input(&mut self, var: Var) -> &mut Self {
        self.eqn().inputs.push(var);
        self
    }

    pub fn inputs(&mut self, vars: impl IntoIterator<Item = Var>) -> &mut Self {
        self.eqn().inputs.extend(vars);
        self
    }

    /// Adds an output: a new variable, which [`HEqnBuilder::done`] returns.
    pub fn output(&mut self, kind: Kind, info: VarInfo) -> &mut Self {
        let var = self.expr.fresh_var();
        self.eqn().outputs.push((var, kind, info));
        self
    }

    pub fn outputs(&mut self, outputs: impl IntoIterator<Item = (Kind, VarInfo)>) -> &mut Self {
        for (kind, info) in outputs {
            self.output(kind, info);
        }
        self
    }

    /// Adds a use of `child` as a subexpression. What its closure
    /// declares is provided through the returned builder.
    pub fn subexpr(&mut self, child: impl Into<HExprRef>) -> &mut HSubExprBuilder {
        let subexprs = &mut self.eqn().subexprs;
        subexprs.push(HSubExprBuilder::new(child));
        subexprs.last_mut().expect("just pushed")
    }

    /// The variables the equation defines, one per output, in order.
    ///
    /// # Panics
    /// If the equation does not have exactly `N` outputs.
    #[track_caller]
    pub fn done<const N: usize>(&mut self) -> [Var; N] {
        let outputs = &self.eqn().outputs;
        assert_eq!(
            outputs.len(),
            N,
            "the equation has {} outputs",
            outputs.len()
        );
        std::array::from_fn(|i| outputs[i].0)
    }
}

/// A use of a high-level expression within an equation, with the symbols
/// and variables of the enclosing expression that are provided for its
/// closure.
#[derive(Archive, Serialize, Clone, Debug)]
#[rkyv(archived = HSubExprData)]
pub struct HSubExprBuilder {
    pub(crate) expr: HExprRef,
    // In the order of the closure symbols of expr
    pub(crate) provided_symbols: Vec<Symbol>,
    // In the order of the closure inputs of expr
    pub(crate) provided_vars: Vec<Var>,
}

impl HSubExprBuilder {
    pub fn new(expr: impl Into<HExprRef>) -> Self {
        Self {
            expr: expr.into(),
            provided_symbols: Vec::new(),
            provided_vars: Vec::new(),
        }
    }

    /// Provides the closure symbols of the expression, in order.
    pub fn symbols(&mut self, symbols: impl IntoIterator<Item = Symbol>) -> &mut Self {
        self.provided_symbols.extend(symbols);
        self
    }

    /// Provides the closure inputs of the expression, in order. A cell
    /// is provided for a cell, and a value for a value.
    pub fn vars(&mut self, vars: impl IntoIterator<Item = Var>) -> &mut Self {
        self.provided_vars.extend(vars);
        self
    }

    /// Like [`HSubExprBuilder::symbols`], by value.
    pub fn with_symbols(mut self, symbols: impl IntoIterator<Item = Symbol>) -> Self {
        self.provided_symbols.extend(symbols);
        self
    }

    /// Like [`HSubExprBuilder::vars`], by value.
    pub fn with_vars(mut self, vars: impl IntoIterator<Item = Var>) -> Self {
        self.provided_vars.extend(vars);
        self
    }

    pub fn expr(&self) -> &HExprRef {
        &self.expr
    }
    pub fn provided_symbols(&self) -> &[Symbol] {
        &self.provided_symbols
    }
    pub fn provided_vars(&self) -> &[Var] {
        &self.provided_vars
    }
}

impl ExprSource for HExprBuilder {
    type Data = HExprData;
    type View<'a> = HExpr<'a>;

    unsafe fn view<'a>(root: &'a HRootExpr, data: &'a HExprData) -> HExpr<'a> {
        // SAFETY: guaranteed by the caller
        unsafe { HExpr::new(root, data) }
    }

    fn check(expr: &Rc<Self>, checked: &mut HashSet<*const Self>) -> Result<(), BuildError> {
        if !checked.insert(Rc::as_ptr(expr)) {
            return Ok(());
        }

        let mut scope = Scope::new(
            &expr.closure_symbols,
            expr.inputs.iter().map(|(_, _, info)| info),
        );
        for (var, kind, info) in &expr.closure_inputs {
            scope.define_param(*var, *kind, info)?;
        }
        for (var, kind, _) in &expr.inputs {
            scope.define_input(*var, *kind)?;
        }

        for instr in &expr.instrs {
            match instr {
                HInstrSource::Apply(eqn) => {
                    scope.vars(&eqn.inputs)?;
                    for sub in &eqn.subexprs {
                        Self::check_subexpr(sub, &scope, checked)?;
                    }
                    for (var, kind, info) in &eqn.outputs {
                        scope.define(*var, *kind, info)?;
                    }
                }
                HInstrSource::CreateCell {
                    cell,
                    initial,
                    info,
                } => {
                    scope.expect(*initial, Kind::Value)?;
                    scope.define_param(*cell, Kind::Cell, info)?;
                }
                HInstrSource::ReadCell { cell, output, info } => {
                    scope.expect(*cell, Kind::Cell)?;
                    scope.define(*output, Kind::Value, info)?;
                }
                HInstrSource::WriteCell { cell, value } => {
                    scope.expect(*cell, Kind::Cell)?;
                    scope.expect(*value, Kind::Value)?;
                }
            }
        }
        scope.vars(&expr.outputs)
    }
}

impl HExprBuilder {
    fn check_subexpr(
        sub: &HSubExprBuilder,
        scope: &Scope,
        checked: &mut HashSet<*const Self>,
    ) -> Result<(), BuildError> {
        scope.symbols(&sub.provided_symbols)?;
        scope.vars(&sub.provided_vars)?;
        let (closure_symbols, closure_kinds): (usize, Vec<Kind>) = match &sub.expr {
            Ref::Internal(child) => {
                Self::check(child, checked)?;
                (
                    child.closure_symbols.len(),
                    child.closure_inputs.iter().map(|p| p.1).collect(),
                )
            }
            Ref::External(handle) => {
                let child = handle.view();
                (
                    child.closure_symbols().len(),
                    child.closure_inputs().map(|p| p.kind()).collect(),
                )
            }
        };
        check_arity(
            (sub.provided_symbols.len(), closure_symbols),
            (sub.provided_vars.len(), closure_kinds.len()),
        )?;
        (sub.provided_vars.iter().zip(closure_kinds))
            .try_for_each(|(&var, kind)| scope.expect(var, kind))
    }
}

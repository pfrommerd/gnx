use std::collections::HashSet;
use std::rc::Rc;
use std::sync::Arc;

use rkyv::{Archive, Serialize, with::Skip};

use super::{Expr, ExprRef, RootExpr};
use crate::check::{BuildError, Scope, check_arity};
use crate::op::Op;
use crate::root::{self, ExprSource, Ref};
use crate::symbol::Symbol;
use crate::var::{Kind, Var, VarInfo, VarScope};

// The builders double as the unarchived form of the data: an ExprBuilder
// archives as an ExprData, a SubExprBuilder as a SubExprData, and so on.

/// An expression that is being constructed: the owned, mutable form of an
/// expression. Variables and symbols are local to the expression: those of
/// an enclosing expression are provided explicitly, where the expression
/// is used as a subexpression.
///
/// ```ignore
/// let mut b = RootExpr::builder();
/// let x = b.input(info.clone());
/// let [y] = b.eqn(op).input(x).output(info).done();
/// b.output(y);
/// let root = b.build()?;
/// ```
#[derive(Archive, Serialize, Clone, Debug, Default)]
#[rkyv(archived = ExprData)]
pub struct ExprBuilder {
    pub(crate) closure_symbols: Vec<Symbol>,
    pub(crate) closure_inputs: Vec<(Var, VarInfo)>,
    pub(crate) inputs: Vec<(Var, VarInfo)>,
    pub(crate) eqns: Vec<EqnSource>,
    pub(crate) outputs: Vec<Var>,
    pub(crate) name_hint: Option<String>,
    #[rkyv(with = Skip)]
    pub(crate) scope: VarScope,
}

impl ExprBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Assembles an expression from definitions that use variables and
    /// symbols allocated elsewhere, such as in a `VarScope`.
    pub fn from_parts(
        closure_symbols: Vec<Symbol>,
        closure_inputs: Vec<(Var, VarInfo)>,
        inputs: Vec<(Var, VarInfo)>,
        eqns: Vec<EqnSource>,
        outputs: Vec<Var>,
    ) -> Self {
        let mut expr = Self {
            closure_symbols,
            closure_inputs,
            inputs,
            eqns,
            outputs,
            name_hint: None,
            scope: VarScope::new(),
        };
        expr.reserve();
        expr
    }

    // Makes the variables and symbols allocated from here on distinct from
    // every one in use.
    pub(crate) fn reserve(&mut self) {
        let params = self.closure_inputs.iter().chain(&self.inputs);
        let defs = params.chain(self.eqns.iter().flat_map(|eqn| &eqn.outputs));
        let symbols = (self.closure_symbols.iter().copied()).chain(
            defs.clone()
                .flat_map(|(_, info)| info.symbols().iter().copied()),
        );
        self.scope.reserve(defs.map(|(var, _)| *var), symbols);
    }

    /// A new, empty expression whose variables and symbols are distinct
    /// from every one allocated here so far. An expression nested in this
    /// one can then refer to the symbols it captures by the same names.
    pub fn nested(&self) -> Self {
        Self {
            scope: self.scope.clone(),
            ..Self::default()
        }
    }

    /// The symbols provided by the enclosing expression.
    pub fn closure_symbols(&self) -> &[Symbol] {
        &self.closure_symbols
    }

    /// Declares `symbol`, allocated elsewhere, as provided by the enclosing
    /// expression.
    pub fn declare_closure_symbol(&mut self, symbol: Symbol) {
        self.closure_symbols.push(symbol);
        self.scope.reserve([], [symbol]);
    }

    /// Keeps the symbols allocated from here on distinct from `symbols`.
    pub fn reserve_symbols(&mut self, symbols: impl IntoIterator<Item = Symbol>) {
        self.scope.reserve([], symbols);
    }

    /// The variables provided by the enclosing expression.
    pub fn closure_inputs(&self) -> &[(Var, VarInfo)] {
        &self.closure_inputs
    }

    /// Replaces the variables provided by the enclosing expression.
    pub fn set_closure_inputs(&mut self, closure_inputs: Vec<(Var, VarInfo)>) {
        self.closure_inputs = closure_inputs;
    }

    pub fn inputs(&self) -> &[(Var, VarInfo)] {
        &self.inputs
    }

    pub fn eqns(&self) -> &[EqnSource] {
        &self.eqns
    }

    /// The results of the expression.
    pub fn output_vars(&self) -> &[Var] {
        &self.outputs
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
    pub fn closure_input(&mut self, info: VarInfo) -> Var {
        let var = self.fresh_var();
        self.closure_inputs.push((var, info));
        var
    }

    /// Declares an input. Symbols in `info` that are not closure symbols
    /// are bound by this input.
    pub fn input(&mut self, info: VarInfo) -> Var {
        let var = self.fresh_var();
        self.inputs.push((var, info));
        var
    }

    /// Appends an equation that applies `op`, to be filled in through
    /// the returned builder.
    pub fn eqn(&mut self, op: Op) -> EqnBuilder<'_> {
        self.eqns.push(EqnSource::new(op, vec![], vec![], vec![]));
        EqnBuilder { expr: self }
    }

    /// Appends an equation whose outputs were allocated with
    /// [`ExprBuilder::fresh_var`], and returns those outputs.
    pub fn push(&mut self, eqn: EqnSource) -> Vec<Var> {
        let outputs = eqn.outputs.iter().map(|(var, _)| *var).collect();
        self.eqns.push(eqn);
        outputs
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
            || (self.inputs.iter()).any(|(_, info)| info.symbols().contains(&symbol))
    }

    /// Find a variable's info at its definition.
    pub fn value_info(&self, var: Var) -> Option<&VarInfo> {
        (self.closure_inputs.iter().chain(&self.inputs))
            .chain(self.eqns.iter().rev().flat_map(|eqn| &eqn.outputs))
            .find_map(|(defined, info)| (*defined == var && !var.is_hole()).then_some(info))
    }

    /// Freezes the expression, for use as a subexpression. Clones of the
    /// result refer to the same expression, which is then stored once.
    #[must_use]
    pub fn finish(self) -> ExprRef {
        self.into()
    }

    /// Checks the expression and everything it refers to, and builds a
    /// root expression of it.
    pub fn build(self) -> Result<Arc<RootExpr>, BuildError> {
        root::build(self)
    }
}

/// An equation that is being constructed: an operation applied to
/// variables of the expression, defining a variable for each output.
#[derive(Archive, Serialize, Clone, Debug)]
#[rkyv(archived = EqnData)]
pub struct EqnSource {
    pub(crate) op: Op,
    pub(crate) inputs: Vec<Var>,
    pub(crate) outputs: Vec<(Var, VarInfo)>,
    pub(crate) subexprs: Vec<SubExprBuilder>,
}

impl EqnSource {
    pub fn new(
        op: Op,
        subexprs: Vec<SubExprBuilder>,
        inputs: Vec<Var>,
        outputs: Vec<(Var, VarInfo)>,
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
    pub fn outputs(&self) -> &[(Var, VarInfo)] {
        &self.outputs
    }
    pub fn subexprs(&self) -> &[SubExprBuilder] {
        &self.subexprs
    }
}

/// Fills in the equation appended by [`ExprBuilder::eqn`].
///
/// ```ignore
/// let mut e = b.eqn(op);
/// e.inputs([x, y]);
/// e.subexpr(child).symbols([s]).vars([x]);
/// e.output(info);
/// let [z] = e.done();
/// ```
pub struct EqnBuilder<'a> {
    expr: &'a mut ExprBuilder,
}

impl EqnBuilder<'_> {
    // The expression is borrowed for as long as this builder exists,
    // so the equation stays the last one.
    fn eqn(&mut self) -> &mut EqnSource {
        self.expr.eqns.last_mut().expect("eqn appended an equation")
    }

    pub fn input(&mut self, var: Var) -> &mut Self {
        self.eqn().inputs.push(var);
        self
    }

    pub fn inputs(&mut self, vars: impl IntoIterator<Item = Var>) -> &mut Self {
        self.eqn().inputs.extend(vars);
        self
    }

    /// Adds an output: a new variable, which [`EqnBuilder::done`] returns.
    pub fn output(&mut self, info: VarInfo) -> &mut Self {
        let var = self.expr.fresh_var();
        self.eqn().outputs.push((var, info));
        self
    }

    pub fn outputs(&mut self, infos: impl IntoIterator<Item = VarInfo>) -> &mut Self {
        for info in infos {
            self.output(info);
        }
        self
    }

    /// Adds a use of `child` as a subexpression. What its closure
    /// declares is provided through the returned builder.
    pub fn subexpr(&mut self, child: impl Into<ExprRef>) -> &mut SubExprBuilder {
        let subexprs = &mut self.eqn().subexprs;
        subexprs.push(SubExprBuilder::new(child));
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

/// A use of an expression within an equation, with the symbols and
/// variables of the enclosing expression that are provided for its closure.
#[derive(Archive, Serialize, Clone, Debug)]
#[rkyv(archived = SubExprData)]
pub struct SubExprBuilder {
    pub(crate) expr: ExprRef,
    // In the order of the closure symbols of expr
    pub(crate) provided_symbols: Vec<Symbol>,
    // In the order of the closure inputs of expr
    pub(crate) provided_vars: Vec<Var>,
}

impl SubExprBuilder {
    pub fn new(expr: impl Into<ExprRef>) -> Self {
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

    /// Provides the closure inputs of the expression, in order.
    pub fn vars(&mut self, vars: impl IntoIterator<Item = Var>) -> &mut Self {
        self.provided_vars.extend(vars);
        self
    }

    /// Like [`SubExprBuilder::symbols`], by value.
    pub fn with_symbols(mut self, symbols: impl IntoIterator<Item = Symbol>) -> Self {
        self.provided_symbols.extend(symbols);
        self
    }

    /// Like [`SubExprBuilder::vars`], by value.
    pub fn with_vars(mut self, vars: impl IntoIterator<Item = Var>) -> Self {
        self.provided_vars.extend(vars);
        self
    }

    pub fn expr(&self) -> &ExprRef {
        &self.expr
    }
    pub fn provided_symbols(&self) -> &[Symbol] {
        &self.provided_symbols
    }
    pub fn provided_vars(&self) -> &[Var] {
        &self.provided_vars
    }
}

impl ExprSource for ExprBuilder {
    type Data = ExprData;
    type View<'a> = Expr<'a>;

    unsafe fn view<'a>(root: &'a RootExpr, data: &'a ExprData) -> Expr<'a> {
        // SAFETY: guaranteed by the caller
        unsafe { Expr::new(root, data) }
    }

    fn check(expr: &Rc<Self>, checked: &mut HashSet<*const Self>) -> Result<(), BuildError> {
        if !checked.insert(Rc::as_ptr(expr)) {
            return Ok(());
        }

        // Every variable of an Expr is a value.
        let mut scope = Scope::new(
            &expr.closure_symbols,
            expr.inputs.iter().map(|(_, info)| info),
        );
        for (var, info) in &expr.closure_inputs {
            scope.define_param(*var, Kind::Value, info)?;
        }
        for (var, _) in &expr.inputs {
            scope.define_input(*var, Kind::Value)?;
        }

        for eqn in &expr.eqns {
            scope.vars(&eqn.inputs)?;
            for sub in &eqn.subexprs {
                scope.symbols(&sub.provided_symbols)?;
                scope.vars(&sub.provided_vars)?;
                let (closure_symbols, closure_inputs) = match &sub.expr {
                    Ref::Internal(child) => {
                        Self::check(child, checked)?;
                        (child.closure_symbols.len(), child.closure_inputs.len())
                    }
                    Ref::External(handle) => {
                        let child = handle.view();
                        (child.closure_symbols().len(), child.closure_inputs().len())
                    }
                };
                check_arity(
                    (sub.provided_symbols.len(), closure_symbols),
                    (sub.provided_vars.len(), closure_inputs),
                )?;
            }
            for (var, info) in &eqn.outputs {
                scope.define(*var, Kind::Value, info)?;
            }
        }
        scope.vars(&expr.outputs)
    }
}

use std::collections::HashMap;
use std::fmt;
use std::ops::Deref;

use rkyv::tuple::ArchivedTuple2;

use super::{
    EqnData, EqnSource, ExprBuilder, ExprData, ExprHandle, ExprRef, RootExpr, SubExprBuilder,
    SubExprData,
};
use crate::lower::LoweringError;
use crate::op::OpRef;
use crate::symbol::Symbol;
use crate::util::{DebugList, TypeName};
use crate::var::{Info, Var, VarInfo, VarInfoData, VarInfoRef};

// Views into a root expression: each pairs a piece of the
// archived data with the root needed to resolve what it refers to
// (operation / info implementations, other expressions).
//
// Views are Copy, and their accessors return data that lives as long as
// the root, not as long as the view.

type ParamData = ArchivedTuple2<Var, VarInfoData>;

fn params<'a>(
    root: &'a RootExpr,
    params: &'a [ParamData],
) -> impl ExactSizeIterator<Item = Param<'a>> + DoubleEndedIterator + Clone {
    params
        .iter()
        .map(move |ParamData { 0: var, 1: info }| Param {
            var: *var,
            // SAFETY: the params are part of the data of root
            info: unsafe { root.info(info) },
        })
}

/// View of an expression: pure, immutable SSA. Results refer to inputs
/// or equation outputs by variable. A value can be used or returned any
/// number of times.
#[derive(Clone, Copy)]
pub struct Expr<'a> {
    root: &'a RootExpr,
    expr: &'a ExprData,
}

impl<'a> Expr<'a> {
    // SAFETY: expr must be part of the data of root
    pub(super) unsafe fn new(root: &'a RootExpr, expr: &'a ExprData) -> Self {
        Expr { root, expr }
    }

    /// The root whose data this expression is stored in.
    pub fn root(self) -> &'a RootExpr {
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
    ) -> impl ExactSizeIterator<Item = Param<'a>> + DoubleEndedIterator + Clone {
        params(self.root, &self.expr.closure_inputs)
    }

    pub fn inputs(self) -> impl ExactSizeIterator<Item = Param<'a>> + DoubleEndedIterator + Clone {
        params(self.root, &self.expr.inputs)
    }

    /// The equations, in definition order.
    pub fn eqns(self) -> impl ExactSizeIterator<Item = Eqn<'a>> + DoubleEndedIterator + Clone {
        let root = self.root;
        self.expr.eqns.iter().map(move |eqn| Eqn { root, eqn })
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

    /// Find a variable at its definition.
    pub fn definition(self, var: Var) -> Option<Param<'a>> {
        if var.is_hole() {
            return None;
        }
        (self.closure_inputs().chain(self.inputs()))
            .find(|param| param.var == var)
            .or_else(|| {
                (self.eqns().rev())
                    .flat_map(Eqn::outputs)
                    .find(|param| param.var == var)
            })
    }

    /// Find a variable's info at its definition.
    pub fn value_info(self, var: Var) -> Option<VarInfoRef<'a>> {
        self.definition(var).map(|param| param.info)
    }

    /// Runs the operation-specific check of every equation, here and in
    /// subexpressions. Building the root has already verified definition
    /// order and what subexpressions are provided.
    pub fn check(self) -> Result<(), LoweringError> {
        for eqn in self.eqns() {
            for sub in eqn.subexprs() {
                sub.expr().check()?;
            }
            eqn.op().get().check_expr(eqn)?;
        }
        Ok(())
    }

    /// Copies the expression out of its root, as an expression under
    /// construction. Variables and symbols keep their identity.
    pub fn to_builder(self) -> ExprBuilder {
        self.copy(&mut HashMap::new())
    }

    // `copied` holds the internal subexpressions copied so far, so that
    // one used several times is still stored once.
    fn copy(self, copied: &mut HashMap<*const ExprData, ExprRef>) -> ExprBuilder {
        let params = |params: &mut dyn Iterator<Item = Param<'a>>| -> Vec<(Var, VarInfo)> {
            params
                .map(|param| (param.var, param.info.to_owned()))
                .collect()
        };
        let mut copy = ExprBuilder::from_parts(
            self.closure_symbols().to_vec(),
            params(&mut self.closure_inputs()),
            params(&mut self.inputs()),
            self.eqns().map(|eqn| eqn.copy(copied)).collect(),
            self.outputs().to_vec(),
        );
        copy.name_hint = self.name_hint().map(str::to_owned);
        copy
    }
}

/// View of an equation in an expression
#[derive(Clone, Copy)]
pub struct Eqn<'a> {
    root: &'a RootExpr,
    eqn: &'a EqnData,
}

impl<'a> Eqn<'a> {
    pub fn op(self) -> OpRef<'a> {
        // SAFETY: the equation is part of the data of root
        unsafe { self.root.op(&self.eqn.op) }
    }

    pub fn inputs(self) -> &'a [Var] {
        &self.eqn.inputs
    }

    /// The variables this equation defines.
    pub fn outputs(self) -> impl ExactSizeIterator<Item = Param<'a>> + DoubleEndedIterator + Clone {
        params(self.root, &self.eqn.outputs)
    }

    pub fn subexprs(
        self,
    ) -> impl ExactSizeIterator<Item = SubExpr<'a>> + DoubleEndedIterator + Clone {
        let root = self.root;
        (self.eqn.subexprs.iter()).map(move |subexpr| SubExpr { root, subexpr })
    }

    /// Copies the equation out of its root.
    pub fn to_source(self) -> EqnSource {
        self.copy(&mut HashMap::new())
    }

    fn copy(self, copied: &mut HashMap<*const ExprData, ExprRef>) -> EqnSource {
        EqnSource::new(
            self.op().to_owned(),
            self.subexprs().map(|sub| sub.copy(copied)).collect(),
            self.inputs().to_vec(),
            (self.outputs())
                .map(|param| (param.var, param.info.to_owned()))
                .collect(),
        )
    }
}

/// View of a subexpression of an equation: a reference to
/// another expression, plus the symbols and vars provided to it
#[derive(Clone, Copy)]
pub struct SubExpr<'a> {
    root: &'a RootExpr,
    subexpr: &'a SubExprData,
}

impl<'a> SubExpr<'a> {
    pub fn expr(self) -> Expr<'a> {
        self.root.expr_at(self.subexpr.expr)
    }

    /// A handle to the expression, if it is stored in another root.
    pub fn external(self) -> Option<&'a ExprHandle> {
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

    /// A reference to the expression for use in an expression under
    /// construction: a handle if it is stored in another root, and
    /// otherwise a copy.
    pub fn to_ref(self) -> ExprRef {
        self.copy_ref(&mut HashMap::new())
    }

    fn copy_ref(self, copied: &mut HashMap<*const ExprData, ExprRef>) -> ExprRef {
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

    fn copy(self, copied: &mut HashMap<*const ExprData, ExprRef>) -> SubExprBuilder {
        SubExprBuilder::new(self.copy_ref(copied))
            .with_symbols(self.provided_symbols().iter().copied())
            .with_vars(self.provided_vars().iter().copied())
    }
}

/// View of a variable at its definition (an input or closure input
/// of an expression, or an output of an equation): the variable and its info
#[derive(Clone, Copy)]
pub struct Param<'a> {
    var: Var,
    info: VarInfoRef<'a>,
}

impl<'a> Param<'a> {
    pub fn var(self) -> Var {
        self.var
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
}

impl<'a> Deref for Param<'a> {
    type Target = dyn Info;

    fn deref(&self) -> &Self::Target {
        self.info.get()
    }
}

impl fmt::Debug for Expr<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Expr")
            .field("closure_symbols", &self.closure_symbols())
            .field("closure_inputs", &DebugList(self.closure_inputs()))
            .field("inputs", &DebugList(self.inputs()))
            .field("eqns", &DebugList(self.eqns()))
            .field("outputs", &self.outputs())
            .finish()
    }
}

impl fmt::Debug for Eqn<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Eqn")
            .field("op", &self.op())
            .field("inputs", &self.inputs())
            .field("outputs", &DebugList(self.outputs()))
            .field("subexprs", &DebugList(self.subexprs()))
            .finish()
    }
}

impl fmt::Debug for SubExpr<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SubExpr")
            .field("expr", &self.expr())
            .field("provided_symbols", &self.provided_symbols())
            .field("provided_vars", &self.provided_vars())
            .finish()
    }
}

impl fmt::Debug for Param<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {:?}", self.var, self.info)
    }
}

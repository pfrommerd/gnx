use std::collections::{BTreeSet, HashMap};

use gnx_expr::{
    EqnSource, Expr, ExprBuilder, LoweringError, Op, SubExprBuilder, Symbol, Var, VarInfo,
};

/// Incrementally constructs one closed expression in its own variable
/// namespace, keeping track of the info of every variable.
///
/// Variables from another expression (including a parent) are never valid
/// here; pass parent values into nested regions explicitly as captures (see
/// [`Builder::region`]).
#[derive(Default)]
pub struct Builder {
    expr: ExprBuilder,
    infos: HashMap<Var, VarInfo>,
}

impl Builder {
    pub fn new() -> Self {
        Self::default()
    }

    // Declares the closure symbols of expr, and keeps clear of the
    // symbols it uses.
    pub(super) fn inherit_symbols(&mut self, expr: Expr<'_>) {
        for &symbol in expr.closure_symbols() {
            self.expr.declare_closure_symbol(symbol);
        }
        self.expr.reserve_symbols(expr.input_symbols());
    }

    pub fn set_name_hint(&mut self, hint: Option<String>) {
        self.expr.set_name_hint(hint);
    }

    /// Allocate an equation result variable for use with [`Builder::push`].
    pub fn fresh(&mut self, info: VarInfo) -> Var {
        let var = self.expr.fresh_var();
        self.infos.insert(var, info);
        var
    }

    /// Declares a symbol provided by the enclosing expression.
    pub fn symbol_closure(&mut self) -> Symbol {
        self.expr.closure_symbol()
    }

    /// Allocates a symbol to be bound by an input: it comes into scope
    /// once it appears in the info of an input.
    pub fn symbol_input(&mut self) -> Symbol {
        self.expr.symbol()
    }

    fn declares_symbol(&self, s: Symbol) -> bool {
        self.expr.declares_symbol(s)
    }

    pub fn closure(&mut self, info: VarInfo) -> Var {
        let var = self.expr.closure_input(info.clone());
        self.infos.insert(var, info);
        var
    }

    pub fn input(&mut self, info: VarInfo) -> Var {
        let var = self.expr.input(info.clone());
        self.infos.insert(var, info);
        var
    }

    /// Metadata of a variable defined in this builder.
    ///
    /// # Panics
    /// If `var` was not allocated by this builder.
    pub fn info(&self, var: Var) -> &VarInfo {
        self.infos
            .get(&var)
            .unwrap_or_else(|| panic!("variable {var:?} is not defined in this builder"))
    }

    pub fn infos(&self, vars: &[Var]) -> Vec<VarInfo> {
        vars.iter().map(|&var| self.info(var).clone()).collect()
    }

    pub fn closure_vars(&self) -> Vec<Var> {
        (self.expr.closure_inputs().iter())
            .map(|(var, _)| *var)
            .collect()
    }

    pub fn input_vars(&self) -> Vec<Var> {
        self.expr.inputs().iter().map(|(var, _)| *var).collect()
    }

    /// Append an equation with fresh result variables.
    pub fn emit(
        &mut self,
        op: Op,
        inputs: &[Var],
        outputs: impl IntoIterator<Item = VarInfo>,
        subexprs: Vec<SubExprBuilder>,
    ) -> Vec<Var> {
        let outputs = outputs
            .into_iter()
            .map(|info| (self.fresh(info.clone()), info))
            .collect();
        self.push(EqnSource::new(op, subexprs, inputs.to_vec(), outputs))
    }

    /// Append a single-result equation without regions.
    pub fn emit1(&mut self, op: Op, inputs: &[Var], output: VarInfo) -> Var {
        self.emit(op, inputs, [output], vec![])[0]
    }

    /// Append an equation whose results were allocated with [`Builder::fresh`].
    pub fn push(&mut self, eqn: EqnSource) -> Vec<Var> {
        for (var, info) in eqn.outputs() {
            if !var.is_hole() {
                self.infos.insert(*var, info.clone());
            }
        }
        self.expr.push(eqn)
    }

    /// Copy a region's equations into this builder and return its result variables.
    /// Bind captured symbols explicitly and infer input symbols from operand metadata.
    pub fn inline(
        &mut self,
        expr: Expr<'_>,
        symbol_captures: &[Symbol],
        captures: &[Var],
        inputs: &[Var],
    ) -> Result<Vec<Var>, LoweringError> {
        let formal: Vec<_> = expr.closure_inputs().chain(expr.inputs()).collect();
        let symbols = expr.bind_symbols(
            symbol_captures,
            &self.infos(captures),
            &self.infos(inputs),
            |s| self.declares_symbol(s),
        )?;
        expr.check()?;
        let mut map: HashMap<Var, Var> = (formal.iter().map(|param| param.var()))
            .zip(captures.iter().chain(inputs).copied())
            .collect();
        for eqn in expr.eqns() {
            let outputs = eqn
                .outputs()
                .map(|param| {
                    let info = param.info_ref().to_owned().remap_symbols(|s| symbols[&s]);
                    if param.var().is_hole() {
                        return (Var::hole(), info);
                    }
                    let new = self.fresh(info.clone());
                    map.insert(param.var(), new);
                    (new, info)
                })
                .collect();
            let subexprs = eqn
                .subexprs()
                .map(|sub| {
                    SubExprBuilder::new(sub.to_ref())
                        .with_vars(rename(&map, sub.provided_vars()))
                        .with_symbols(sub.provided_symbols().iter().map(|s| symbols[s]))
                })
                .collect();
            self.push(EqnSource::new(
                eqn.op().to_owned(),
                subexprs,
                rename(&map, eqn.inputs()),
                outputs,
            ));
        }
        Ok(rename(&map, expr.outputs()))
    }

    /// Build a closed child region. `build` receives the child's closure
    /// variables (bound to `captures` in this builder) and its inputs.
    pub fn region<E>(
        &self,
        captures: &[Var],
        inputs: &[VarInfo],
        build: impl FnOnce(&mut Builder, &[Var], &[Var]) -> Result<Vec<Var>, E>,
    ) -> Result<SubExprBuilder, E> {
        // The child refers to symbols of this builder by the same names.
        let mut child = Builder {
            expr: self.expr.nested(),
            infos: HashMap::new(),
        };
        let mut symbol_captures = Vec::new();
        let mut bound = BTreeSet::new();
        // The inputs of the child bind the symbols in their infos.
        for &s in inputs.iter().flat_map(|info| info.symbols()) {
            if bound.insert(s) {
                assert!(self.declares_symbol(s), "undeclared input symbol");
            }
        }
        for &s in captures.iter().flat_map(|v| self.info(*v).symbols()) {
            if bound.insert(s) {
                assert!(self.declares_symbol(s), "undeclared capture symbol");
                child.expr.declare_closure_symbol(s);
                symbol_captures.push(s);
            }
        }
        let closure: Vec<_> = captures
            .iter()
            .map(|&var| child.closure(self.info(var).clone()))
            .collect();
        let params: Vec<_> = inputs
            .iter()
            .map(|info| child.input(info.clone()))
            .collect();
        let outputs = build(&mut child, &closure, &params)?;
        let mut expr = child.finish(outputs);
        // A result type (or nested attachment) can capture a symbol even when
        // the region has no captured value with that symbol in its metadata.
        let free: Vec<_> = (expr.eqns().iter())
            .flat_map(|e| {
                (e.outputs().iter().flat_map(|(_, info)| info.symbols()))
                    .chain(e.subexprs().iter().flat_map(|r| r.provided_symbols()))
            })
            .copied()
            .collect();
        for s in free {
            if bound.insert(s) {
                assert!(self.declares_symbol(s), "undeclared free symbol");
                expr.declare_closure_symbol(s);
                symbol_captures.push(s);
            }
        }
        Ok(SubExprBuilder::new(expr)
            .with_vars(captures.iter().copied())
            .with_symbols(symbol_captures))
    }

    pub fn finish(self, outputs: Vec<Var>) -> ExprBuilder {
        let mut expr = self.expr;
        expr.outputs(outputs);
        expr
    }
}

fn rename(map: &HashMap<Var, Var>, vars: &[Var]) -> Vec<Var> {
    vars.iter()
        .map(|var| {
            *map.get(var)
                .unwrap_or_else(|| panic!("undefined variable {var:?} in inlined expression"))
        })
        .collect()
}

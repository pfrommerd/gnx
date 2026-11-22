use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

use crate::{Eqn, Expr, Op, SubExpr, Var, VarInfo, VarScope};

/// Incrementally constructs one closed [`Expr`] in its own variable namespace.
///
/// Variables from another expression (including a parent) are never valid
/// here; pass parent values into nested regions explicitly as captures (see
/// [`Builder::region`]).
#[derive(Default)]
pub struct Builder {
    symbol_closure: Vec<crate::Symbol>,
    symbol_inputs: Vec<crate::Symbol>,
    scope: VarScope,
    closure: Vec<(Var, VarInfo)>,
    inputs: Vec<(Var, VarInfo)>,
    eqns: Vec<Eqn>,
    infos: HashMap<Var, VarInfo>,
    name_hint: Option<Cow<'static, str>>,
}

impl Builder {
    pub fn new() -> Self {
        Self::default()
    }

    pub(crate) fn inherit_symbols(&mut self, expr: &Expr) {
        self.symbol_closure = expr.symbol_closure.clone();
        self.symbol_inputs = expr.symbol_inputs.clone();
        self.scope.next_symbol = self
            .symbol_closure
            .iter()
            .chain(&self.symbol_inputs)
            .map(|s| s.0 + 1)
            .max()
            .unwrap_or(0);
    }
    pub fn set_name_hint(&mut self, hint: Option<Cow<'static, str>>) {
        self.name_hint = hint;
    }

    /// Allocate a variable bound to `info`, reusing the name hint of `like`.
    pub(crate) fn fresh_like(&mut self, like: Option<Var>, info: VarInfo) -> Var {
        let var = match like.and_then(|var| var.name_hint) {
            Some(hint) => self.scope.create_var_with_hint(hint),
            None => self.scope.create_var(),
        };
        self.infos.insert(var, info);
        var
    }

    /// Allocate an equation result variable for use with [`Builder::push`].
    pub fn fresh(&mut self, info: VarInfo) -> Var {
        self.fresh_like(None, info)
    }

    pub fn symbol_closure(&mut self) -> crate::Symbol {
        let s = self.scope.create_symbol();
        self.symbol_closure.push(s);
        s
    }
    pub fn symbol_input(&mut self) -> crate::Symbol {
        let s = self.scope.create_symbol();
        self.symbol_inputs.push(s);
        s
    }
    fn declares_symbol(&self, s: crate::Symbol) -> bool {
        self.symbol_closure.contains(&s) || self.symbol_inputs.contains(&s)
    }
    pub fn closure(&mut self, info: VarInfo) -> Var {
        let var = self.fresh(info.clone());
        self.closure.push((var, info));
        var
    }

    pub fn input(&mut self, info: VarInfo) -> Var {
        let var = self.fresh(info.clone());
        self.inputs.push((var, info));
        var
    }

    pub(crate) fn closure_like(&mut self, like: Var, info: VarInfo) -> Var {
        let var = self.fresh_like(Some(like), info.clone());
        self.closure.push((var, info));
        var
    }

    pub(crate) fn input_like(&mut self, like: Var, info: VarInfo) -> Var {
        let var = self.fresh_like(Some(like), info.clone());
        self.inputs.push((var, info));
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
        self.closure.iter().map(|(var, _)| *var).collect()
    }

    pub fn input_vars(&self) -> Vec<Var> {
        self.inputs.iter().map(|(var, _)| *var).collect()
    }

    /// Append an equation with fresh result variables.
    pub fn emit(
        &mut self,
        op: Op,
        inputs: &[Var],
        outputs: impl IntoIterator<Item = VarInfo>,
        subexprs: Vec<SubExpr>,
    ) -> Vec<Var> {
        let outputs = outputs
            .into_iter()
            .map(|info| (self.fresh(info.clone()), info))
            .collect();
        self.push(Eqn::new_with_subexprs(
            op,
            subexprs,
            inputs.to_vec(),
            outputs,
        ))
    }

    /// Append a single-result equation without regions.
    pub fn emit1(&mut self, op: Op, inputs: &[Var], output: VarInfo) -> Var {
        self.emit(op, inputs, [output], vec![])[0]
    }

    /// Append an equation whose results were allocated with [`Builder::fresh`].
    pub fn push(&mut self, eqn: Eqn) -> Vec<Var> {
        let outputs = eqn
            .outputs
            .iter()
            .map(|(var, info)| {
                self.infos.insert(*var, info.clone());
                *var
            })
            .collect();
        self.eqns.push(eqn);
        outputs
    }

    /// Copy a region's equations into this builder and return its result variables.
    /// Bind captured symbols explicitly and infer input symbols from operand metadata.
    pub fn inline(
        &mut self,
        expr: &Expr,
        symbol_captures: &[crate::Symbol],
        captures: &[Var],
        inputs: &[Var],
    ) -> Result<Vec<Var>, crate::LoweringError> {
        let symbols = crate::symbol_substitution(
            &expr.symbol_closure,
            symbol_captures,
            &expr.symbol_inputs,
            expr.closure
                .iter()
                .chain(&expr.inputs)
                .map(|v| &v.1)
                .zip(captures.iter().chain(inputs).map(|v| self.info(*v))),
            |s| self.declares_symbol(s),
        )?;
        if captures.len() != expr.closure.len() || inputs.len() != expr.inputs.len() {
            return Err(crate::LoweringError::CallArity);
        }
        expr.check()?;
        let mut map: HashMap<Var, Var> = expr
            .closure
            .iter()
            .chain(&expr.inputs)
            .map(|(var, _)| *var)
            .zip(captures.iter().chain(inputs).copied())
            .collect();
        for eqn in &expr.eqns {
            let outputs = eqn
                .outputs
                .iter()
                .map(|(var, info)| {
                    let info = info.remap_symbols(|s| symbols[&s]);
                    let new = self.fresh_like(Some(*var), info.clone());
                    map.insert(*var, new);
                    (new, info)
                })
                .collect();
            let eqn = Eqn::new_with_subexprs(
                eqn.op.clone(),
                eqn.subexprs
                    .iter()
                    .map(|sub| {
                        SubExpr::new(sub.expr.clone(), rename(&map, &sub.captures))
                            .with_symbol_captures(
                                sub.symbol_captures.iter().map(|s| symbols[s]).collect(),
                            )
                    })
                    .collect(),
                rename(&map, &eqn.inputs),
                outputs,
            );
            self.push(eqn);
        }
        Ok(rename(&map, &expr.outputs))
    }

    /// Build a closed child region. `build` receives the child's closure
    /// variables (bound to `captures` in this builder) and its inputs.
    pub fn region<E>(
        &self,
        captures: &[Var],
        inputs: &[VarInfo],
        build: impl FnOnce(&mut Builder, &[Var], &[Var]) -> Result<Vec<Var>, E>,
    ) -> Result<SubExpr, E> {
        let mut child = Builder::new();
        let mut symbol_captures = Vec::new();
        let mut bound = std::collections::BTreeSet::new();
        for &s in inputs.iter().flat_map(VarInfo::symbols) {
            if bound.insert(s) {
                assert!(self.declares_symbol(s), "undeclared input symbol");
                child.symbol_inputs.push(s);
            }
        }
        for &s in captures.iter().flat_map(|v| self.info(*v).symbols()) {
            if bound.insert(s) {
                assert!(self.declares_symbol(s), "undeclared capture symbol");
                child.symbol_closure.push(s);
                symbol_captures.push(s);
            }
        }
        child.scope.next_symbol = self.scope.next_symbol;
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
        bound.extend(
            expr.symbol_inputs
                .iter()
                .chain(&expr.symbol_closure)
                .copied(),
        );
        let free: Vec<_> = expr
            .eqns
            .iter()
            .flat_map(|e| {
                e.outputs
                    .iter()
                    .flat_map(|v| v.1.symbols())
                    .chain(e.subexprs.iter().flat_map(|r| &r.symbol_captures))
            })
            .copied()
            .collect();
        for s in free {
            if bound.insert(s) {
                assert!(self.declares_symbol(s), "undeclared free symbol");
                expr.symbol_closure.push(s);
                symbol_captures.push(s);
            }
        }
        Ok(SubExpr::new(Arc::new(expr), captures.to_vec()).with_symbol_captures(symbol_captures))
    }

    pub fn finish(self, outputs: Vec<Var>) -> Expr {
        Expr {
            symbol_inputs: self.symbol_inputs,
            symbol_closure: self.symbol_closure,
            closure: self.closure,
            inputs: self.inputs,
            eqns: self.eqns,
            outputs,
            name_hint: self.name_hint,
        }
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

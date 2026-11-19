use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

use crate::{Eqn, Expr, Op, SubExpr, ValueInfo, Var, VarScope};

/// Incrementally constructs one closed [`Expr`] in its own variable namespace.
///
/// Variables from another expression (including a parent) are never valid
/// here; pass parent values into nested regions explicitly as captures (see
/// [`Builder::region`]).
#[derive(Default)]
pub struct Builder {
    scope: VarScope,
    closure: Vec<(Var, ValueInfo)>,
    inputs: Vec<(Var, ValueInfo)>,
    eqns: Vec<Eqn>,
    infos: HashMap<Var, ValueInfo>,
    name_hint: Option<Cow<'static, str>>,
}

impl Builder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_name_hint(&mut self, hint: Option<Cow<'static, str>>) {
        self.name_hint = hint;
    }

    /// Allocate a variable bound to `info`, reusing the name hint of `like`.
    pub(crate) fn fresh_like(&mut self, like: Option<Var>, info: ValueInfo) -> Var {
        let var = match like.and_then(|var| var.name_hint) {
            Some(hint) => self.scope.create_var_with_hint(hint),
            None => self.scope.create_var(),
        };
        self.infos.insert(var, info);
        var
    }

    /// Allocate an equation result variable for use with [`Builder::push`].
    pub fn fresh(&mut self, info: ValueInfo) -> Var {
        self.fresh_like(None, info)
    }

    pub fn closure(&mut self, info: ValueInfo) -> Var {
        let var = self.fresh(info.clone());
        self.closure.push((var, info));
        var
    }

    pub fn input(&mut self, info: ValueInfo) -> Var {
        let var = self.fresh(info.clone());
        self.inputs.push((var, info));
        var
    }

    pub(crate) fn closure_like(&mut self, like: Var, info: ValueInfo) -> Var {
        let var = self.fresh_like(Some(like), info.clone());
        self.closure.push((var, info));
        var
    }

    pub(crate) fn input_like(&mut self, like: Var, info: ValueInfo) -> Var {
        let var = self.fresh_like(Some(like), info.clone());
        self.inputs.push((var, info));
        var
    }

    /// Metadata of a variable defined in this builder.
    ///
    /// # Panics
    /// If `var` was not allocated by this builder.
    pub fn info(&self, var: Var) -> &ValueInfo {
        self.infos
            .get(&var)
            .unwrap_or_else(|| panic!("variable {var:?} is not defined in this builder"))
    }

    pub fn infos(&self, vars: &[Var]) -> Vec<ValueInfo> {
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
        outputs: impl IntoIterator<Item = ValueInfo>,
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
    pub fn emit1(&mut self, op: Op, inputs: &[Var], output: ValueInfo) -> Var {
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

    /// Copy a closed expression's equations, binding its closure to `captures`
    /// and its inputs to `inputs`. Returns its results.
    pub fn inline(&mut self, expr: &Expr, captures: &[Var], inputs: &[Var]) -> Vec<Var> {
        assert_eq!(captures.len(), expr.closure.len(), "inline capture arity");
        assert_eq!(inputs.len(), expr.inputs.len(), "inline input arity");
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
                    let new = self.fresh_like(Some(*var), info.clone());
                    map.insert(*var, new);
                    (new, info.clone())
                })
                .collect();
            let eqn = Eqn::new_with_subexprs(
                eqn.op.clone(),
                eqn.subexprs
                    .iter()
                    .map(|sub| SubExpr::new(sub.expr.clone(), rename(&map, &sub.captures)))
                    .collect(),
                rename(&map, &eqn.inputs),
                outputs,
            );
            self.push(eqn);
        }
        rename(&map, &expr.outputs)
    }

    /// Build a closed child region. `build` receives the child's closure
    /// variables (bound to `captures` in this builder) and its inputs.
    pub fn region<E>(
        &self,
        captures: &[Var],
        inputs: &[ValueInfo],
        build: impl FnOnce(&mut Builder, &[Var], &[Var]) -> Result<Vec<Var>, E>,
    ) -> Result<SubExpr, E> {
        let mut child = Builder::new();
        let closure: Vec<_> = captures
            .iter()
            .map(|&var| child.closure(self.info(var).clone()))
            .collect();
        let params: Vec<_> = inputs
            .iter()
            .map(|info| child.input(info.clone()))
            .collect();
        let outputs = build(&mut child, &closure, &params)?;
        Ok(SubExpr::new(
            Arc::new(child.finish(outputs)),
            captures.to_vec(),
        ))
    }

    pub fn finish(self, outputs: Vec<Var>) -> Expr {
        Expr {
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

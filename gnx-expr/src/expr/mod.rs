pub mod value;
pub mod trace;
mod attr;
mod capture;

pub use attr::*;
pub use capture::*;

use std::collections::BTreeMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::fmt::Debug;
use std::cmp::{Eq, PartialEq};
use std::cmp::{Ord, PartialOrd};

pub trait Operation: Sync {
    fn name(&self) -> &'static str;
    fn dialect(&self) -> &'static dyn Dialect;
}

pub trait Dialect: Sync {
    fn name(&self) -> &'static str;
    fn create(&self, name: &'static str) -> Option<&'static dyn Operation>;
}

// An Op is an Operation combined with an attribute set.
#[derive(Clone)]
pub struct Op {
    impl_: &'static dyn Operation,
    attrs: AttrMap,
}

impl Debug for Op {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}

impl Hash for Op {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.impl_.dialect().name().hash(state);
        self.impl_.name().hash(state);
        self.attrs.hash(state);
    }
}

impl PartialEq for Op {
    fn eq(&self, other: &Self) -> bool {
        (self.impl_.dialect().name(), self.impl_.name(), &self.attrs) == (other.impl_.dialect().name(), other.impl_.name(), &other.attrs)
    }
}

impl Eq for Op {}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.impl_.dialect().name(), self.impl_.name())?;
        if !self.attrs.is_empty() {
            write!(f, "<")?;
            for (i, (k, v)) in self.attrs.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{k}: {v}")?;
            }
            write!(f, ">")?;
        }
        Ok(())
    }
}

impl Op {
    pub fn new(impl_: &'static dyn Operation, attrs: AttrMap) -> Self {
        Op { impl_, attrs }
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct VarScope {
    next_id: usize,
}

impl VarScope {
    pub fn new() -> VarScope {
        VarScope { next_id: 0 }
    }

    pub fn create_var(&mut self) -> Var {
        let v = Var::bind(self.next_id);
        self.next_id += 1;
        v
    }
}


/// A jaxpr-style variable: bound to an index or 0 for a hole.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Var {
    id: usize,
}

impl Var {
    pub fn hole() -> Self {
        Var { id: 0 }
    }

    pub fn bind(id: usize) -> Self {
        Var { id: id + 1 } // 0 is reserved for holes
    }

    pub fn id(&self) -> Option<usize> {
        match self.id {
            0 => None,
            i => Some(i - 1),
        }
    }
}

impl fmt::Display for Var {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.id() {
            Some(i) => write!(f, "v{}", i),
            None => write!(f, "?"),
        }
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum Effect {
    Unused, Read, Write, ReadWrite
}

impl Effect {
    pub fn combine(self, other: Effect) -> Effect {
        match (self, other) {
            (Effect::Unused, _) => other,
            (_, Effect::Unused) => self,
            (Effect::Read, Effect::Write) => Effect::ReadWrite,
            (Effect::Write, Effect::Read) => Effect::ReadWrite,
            _ => self,
        }
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct Effects {
    effected: BTreeMap<Var, Effect>
}

impl Effects {
    pub fn new() -> Self {
        Effects { effected: BTreeMap::new() }
    }
    pub fn insert(&mut self, var: Var, effect: Effect) {
        self.effected.insert(var, effect);
    }
    pub fn remove(&mut self, var: &Var) -> Option<Effect> {
        self.effected.remove(var)
    }
    pub fn size(&self) -> usize {
        self.effected.len()
    }
    pub fn is_empty(&self) -> bool {
        self.effected.is_empty()
    }
}

impl std::ops::Index<&Var> for Effects {
    type Output = Effect;
    fn index(&self, var: &Var) -> &Self::Output {
        self.effected.get(var).unwrap_or(&Effect::Unused)
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct Input {
    var: Var,
    effect: Effect,
}

impl Input {
    pub fn new(var: Var, effect: Effect) -> Self {
        Self { var, effect }
    }
    pub fn var(&self) -> &Var {
        &self.var
    }
    pub fn effect(&self) -> Effect {
        self.effect
    }
}

impl fmt::Display for Input {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.var)?;
        match self.effect {
            Effect::Unused | Effect::Read => (),
            Effect::Write | Effect::ReadWrite => write!(f, "[:]")?,
        }
        Ok(())
    }
}

/// One equation: `let outs... = op [attrs] closure... ins...` in ANF (see [jaxpr](https://docs.jax.dev/en/latest/jaxpr.html)).
#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Eqn {
    op: Op,
    closure: Vec<Input>,
    inputs: Vec<Input>,
    outputs: Vec<Var>,
}

impl Eqn {
    pub fn new(op: Op, closure: Vec<Input>, inputs: Vec<Input>, outputs: Vec<Var>) -> Self {
        Eqn {
            op,
            closure,
            inputs,
            outputs,
        }
    }

    pub fn op(&self) -> &Op {
        &self.op
    }

    pub fn closure(&self) -> &[Input] {
        &self.closure
    }

    pub fn inputs(&self) -> &[Input] {
        &self.inputs
    }

    pub fn outputs(&self) -> &[Var] {
        &self.outputs
    }
    
    // Propagate effects *backwards* through an equation.
    pub fn propagate_effects(&self, effects: &mut Effects) {
        // By default add any input effects to the effects map.
        for input in self.inputs.iter().filter(|i| i.effect() != Effect::Unused) {
            effects.insert(input.var().clone(), input.effect());
        }
        for closure in self.closure.iter().filter(|c| c.effect() != Effect::Unused) {
            effects.insert(closure.var().clone(), closure.effect());
        }
        let mut output_effects = Effects::new();
        let mut max_effect = Effect::Unused;
        for output in &self.outputs {
            if let Some(e) = effects.remove(output) {
                output_effects.insert(output.clone(), e);
                max_effect = max_effect.combine(e);
            }
        }
        if output_effects.is_empty() { return; }
        // TODO: Lookup the operation effects implementation.
        // If there is no implementation, pessimistically assume
        // the worst possible input-output association.
        for input in &self.inputs {
            effects.insert(input.var().clone(), max_effect);
        }
    }
}

impl fmt::Display for Eqn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, o) in self.outputs.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{o}")?;
        }
        if !self.outputs.is_empty() {
            write!(f, " = ")?;
        }
        write!(f, "{}", self.op)?;
        if !self.closure.is_empty() {
            write!(f, " [")?;
            for (i, c) in self.closure.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{c}")?;
            }
            write!(f, "]")?;
        }
        write!(f, " ")?;
        for (i, inp) in self.inputs.iter().enumerate() {
            if i > 0 {
                write!(f, " ")?;
            }
            write!(f, "{inp}")?;
        }
        Ok(())
    }
}



/// Functional ANF expression: `lambda closure* ; explicit* . let eqns* in outputs*`.
#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Expr {
    closure: Vec<Input>,
    inputs: Vec<Input>,
    eqns: Vec<Eqn>,
    outputs: Vec<Var>,
    var_scope: VarScope,
}


impl Expr {
    pub fn new(
        closure: Vec<Input>,
        inputs: Vec<Input>,
        eqns: Vec<Eqn>,
        outputs: Vec<Var>,
        var_scope: VarScope,
    ) -> Self {
        Expr {
            closure,
            inputs,
            eqns,
            outputs,
            var_scope,
        }
    }

    pub fn closure(&self) -> &[Input] { &self.closure }
    pub fn inputs(&self) -> &[Input] { &self.inputs }

    pub fn eqns(&self) -> &[Eqn] {
        &self.eqns
    }

    pub fn outputs(&self) -> &[Var] {
        &self.outputs
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{ lambda ")?;
        for (i, c) in self.closure.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{c}")?;
        }
        write!(f, " ; ")?;
        for (i, x) in self.inputs.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{x}")?;
        }
        writeln!(f, " .")?;
        for eq in &self.eqns {
            writeln!(f, "    {eq}")?;
        }
        write!(f, "  in (")?;
        for (i, o) in self.outputs.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{o}")?;
        }
        write!(f, ") }}")
    }
}
pub mod value;
pub mod trace;
mod attr;

pub use attr::*;

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt;
use std::ops::Deref;
use std::sync::Arc;

use self::trace::{Generic, Invocation, Tracer, TraceRef};

#[derive(Clone, Hash, PartialEq, Eq)]
pub enum OpString {
    Static(&'static str),
    Shared(Arc<String>),
}

impl fmt::Debug for OpString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OpString::Static(s) => write!(f, "\"{}\"", s),
            OpString::Shared(s) => write!(f, "\"{}\"", s.as_str()),
        }
    }
}

impl Deref for OpString {
    type Target = str;
    fn deref(&self) -> &Self::Target {
        match self {
            OpString::Static(s) => s,
            OpString::Shared(s) => s.as_str(),
        }
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Op {
    pub dialect: OpString,
    pub name: OpString,
    pub attrs: AttrMap,
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.dialect.deref(), self.name.deref())
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


/// A jaxpr-style variable: either bound to an index or a hole (unification / rewrite).
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Var {
    id: Option<usize>,
}

impl Var {
    pub fn hole() -> Self {
        Var { id: None }
    }

    pub fn bind(id: usize) -> Self {
        Var { id: Some(id) }
    }

    pub fn id(&self) -> Option<usize> {
        self.id
    }
}

impl fmt::Display for Var {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.id {
            Some(i) => write!(f, "v{}", i),
            None => write!(f, "?"),
        }
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum Effect {
    Unused, Read, Write, ReadWrite
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

/// A closed expression is an expression with captured_inputs bound.
/// This is used for pretty-printing Expr using the closure vars for the pretty-printing.
pub struct ClosedExpr {
    expr: Expr,
    closure: Cow<'static, [Var]>,
}

impl ClosedExpr {
    pub fn expr(&self) -> &Expr {
        &self.expr
    }

    pub fn closure(&self) -> &[Var] {
        &self.closure
    }
}

/// Snapshot of a trace subgraph in jaxpr-like [`Expr`] form, plus hoisted closure [`TraceRef`]s
/// (analogous to `consts` in [`ClosedJaxpr`](https://docs.jax.dev/en/latest/jaxpr.html)).
pub struct Capture {
    expr: Expr,
    closure: Vec<TraceRef>,
}

impl Capture {
    /// Materialize the trace subgraph needed for `outputs` into an [`Expr`].
    ///
    /// `inputs` is the ordered formal parameter list (jaxpr `invars` / `constvars` split).
    /// A boundary trace is classified as **closure** (and listed in [`Capture::closure`]) when
    /// it is not among `inputs`, or when it holds a concrete [`crate::expr::value::Value`]. Otherwise
    /// it is an **explicit** input slot.
    pub fn from_trace_refs<'a, I, O>(inputs: I, outputs: O) -> Result<Self, ()>
    where
        I: IntoIterator<Item = &'a TraceRef>,
        O: IntoIterator<Item = &'a TraceRef>,
    {
        let inputs: Vec<&TraceRef> = inputs.into_iter().collect();
        let outputs: Vec<&TraceRef> = outputs.into_iter().collect();
        let input_keys: HashSet<TracerKey> = inputs.iter().map(|t| TracerKey::from(*t)).collect();

        let mut var_scope = VarScope::new();
        let mut trace_to_var: HashMap<TracerKey, Var> = HashMap::new();
        let mut input_vars = Vec::new();
        // The lifted closure variables and the corresponding trace refs.
        let mut closure_vars = Vec::new();
        let mut closure_refs = Vec::new();
        let (invocations, output_vars) = collect_invocations(&outputs);
        // Sort the invocations topologically and allocate space for the equations.
        let ordered_invs = topo_sort_invocations(invocations);
        let mut eqns = Vec::with_capacity(ordered_invs.len());
        // Create any variables for the explicit inputs.
        for k in input_keys {
            let v = var_scope.create_var();
            trace_to_var.insert(k, v.clone());
            input_vars.push(v);
        }
        // Create an equation for each invocation.
        for inv in &ordered_invs {
            let closure : Vec<Input> = inv.closure().iter().map(|inp| {
                match trace_to_var.get(&TracerKey::from(&inp.trace)) {
                    Some(var) => Input { var: var.clone(), effect: inp.effect},
                    None => {
                        let v = var_scope.create_var();
                        trace_to_var.insert(TracerKey::from(&inp.trace), v.clone());
                        closure_vars.push(v.clone());
                        closure_refs.push(inp.trace.clone());
                        Input { var: v, effect: inp.effect }
                    }
                }
            }).collect();
            let inputs : Vec<Input> = inv.inputs().iter().map(|inp| {
                match trace_to_var.get(&TracerKey::from(&inp.trace)) {
                    Some(var) => Input { var: var.clone(), effect: inp.effect},
                    None => {
                        // Capture the trace as a closure variable.
                        let v = var_scope.create_var();
                        trace_to_var.insert(TracerKey::from(&inp.trace), v.clone());
                        closure_vars.push(v.clone());
                        closure_refs.push(inp.trace.clone());
                        Input { var: v, effect: inp.effect }
                    }
                }
            }).collect();
            // Add variables for the outputs.
            let outputs = &output_vars[&InvKey::from(inv)];
            let mut out_vars = Vec::with_capacity(inv.outputs());
            // The number of outputs is the maximum output index plus one.
            let n_out = inv.outputs();
            for out_idx in 0..n_out {
                match outputs.get(&out_idx) {
                    Some(out_tracer) => {
                        let v = var_scope.create_var();
                        trace_to_var.insert(TracerKey::from(out_tracer), v.clone());
                        out_vars.push(v);
                    },
                    None => out_vars.push(Var::hole()),
                }
            }
            eqns.push(Eqn::new(
                inv.op().clone(),
                closure,
                inputs,
                out_vars,
            ));
        }

        for trace in &outputs {
            let k = TracerKey::from(*trace);
            if !trace_to_var.contains_key(&k) {
                return Err(());
            }
        }
        let output_vars: Vec<Var> = outputs.iter()
            .map(|t| trace_to_var[&TracerKey::from(*t)].clone())
            .collect();
        // TODO: Propagate the effects through the equations and into the closure/input variables.
        let closure: Vec<Input> = closure_vars.into_iter().map(|v| Input { var: v, effect: Effect::Read }).collect();
        let inputs: Vec<Input> = input_vars.into_iter().map(|v| Input { var: v, effect: Effect::Read }).collect();
        Ok(Capture {
            expr: Expr::new(
                closure,
                inputs,
                eqns,
                output_vars,
                var_scope,
            ),
            closure: closure_refs,
        })
    }

    pub fn expr(&self) -> &Expr {
        &self.expr
    }

    pub fn closure(&self) -> &[TraceRef] {
        &self.closure
    }

    pub fn into_parts(self) -> (Expr, Vec<TraceRef>) {
        (self.expr, self.closure)
    }
}

/// Identity key for a [`TraceRef`] / trace node (stable for the lifetime of that `Arc<Trace>`).
#[derive(Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct TracerKey(usize);

impl From<&TraceRef> for TracerKey {
    fn from(t: &TraceRef) -> Self {
        TracerKey(Arc::as_ptr(t) as usize)
    }
}

impl From<&Tracer<Generic>> for TracerKey {
    fn from(t: &Tracer<Generic>) -> Self {
        TracerKey::from(&t.trace_ref())
    }
}

/// Identity key for an [`Invocation`] (stable for the lifetime of that `Arc<Invocation>`).
#[derive(Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct InvKey(usize);

impl From<&Arc<Invocation>> for InvKey {
    fn from(inv: &Arc<Invocation>) -> Self {
        InvKey(Arc::as_ptr(inv) as usize)
    }
}

// Collect the invocations and the output vars for each invocation.
fn collect_invocations(outputs: &[&TraceRef]) -> (HashMap<InvKey, Arc<Invocation>>, HashMap<InvKey, BTreeMap<usize, TraceRef>>) {
    let mut invs: HashMap<InvKey, Arc<Invocation>> = HashMap::new();
    let mut output_vars: HashMap<InvKey, BTreeMap<usize, TraceRef>> = HashMap::new();
    let mut pending: Vec<Arc<Invocation>> = Vec::new();

    for out in outputs {
        if let Some((inv, ret_idx)) = out.producer() {
            let key = InvKey::from(&inv);
            pending.push(inv);
            output_vars.entry(key).or_default().insert(ret_idx, (*out).clone());
        }
    }

    while let Some(inv) = pending.pop() {
        let k = InvKey::from(&inv);
        if invs.contains_key(&k) {
            continue;
        }
        invs.insert(k, inv.clone());
        for inp in inv.inputs() {
            if let Some((pred, out_idx)) = inp.trace.producer() {
                let key = InvKey::from(&pred);
                pending.push(pred);
                output_vars.entry(key).or_default().insert(out_idx, inp.trace.clone());
            }
        }
    }

    (invs, output_vars)
}

fn topo_sort_invocations(invs: HashMap<InvKey, Arc<Invocation>>) -> Vec<Arc<Invocation>> {
    if invs.is_empty() {
        return vec![];
    }

    let mut indeg: HashMap<InvKey, usize> = invs.keys().map(|&k| (k, 0)).collect();
    let mut succ: HashMap<InvKey, Vec<InvKey>> = HashMap::new();

    for inv in invs.values() {
        let kid = InvKey::from(inv);
        for inp in inv.inputs() {
            if let Some((pred, _)) = inp.trace.producer() {
                let pid = InvKey::from(&pred);
                if pid != kid && invs.contains_key(&pid) {
                    succ.entry(pid).or_default().push(kid);
                    *indeg.get_mut(&kid).unwrap() += 1;
                }
            }
        }
    }

    let mut ready: BTreeSet<InvKey> = indeg
        .iter()
        .filter(|(_, d)| **d == 0)
        .map(|(&k, _)| k)
        .collect();
    let mut outv = Vec::with_capacity(invs.len());

    while let Some(k) = ready.pop_first() {
        outv.push(invs[&k].clone());
        if let Some(children) = succ.get(&k) {
            for &c in children {
                let d = indeg.get_mut(&c).expect("child in succ maps to indeg");
                *d -= 1;
                if *d == 0 {
                    ready.insert(c);
                }
            }
        }
    }

    if outv.len() != invs.len() {
        panic!("gnx-expr: cycle found in trace graph (topological sort failed)");
    }
    outv
}
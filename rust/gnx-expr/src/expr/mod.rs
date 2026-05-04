
use std::borrow::Cow;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt;
use std::ops::Deref;
use std::sync::Arc;

use self::trace::{Generic, Invocation, Tracer};

pub mod value;
pub mod trace;
mod attr;
pub use attr::*;

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

/// One equation: `let outs... = op [attrs] closure... ins...` in ANF (see [jaxpr](https://docs.jax.dev/en/latest/jaxpr.html)).
#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Eqn {
    op: Op,
    closure: Vec<Var>,
    inputs: Vec<Var>,
    outputs: Vec<Var>,
}

impl Eqn {
    pub fn new(op: Op, closure: Vec<Var>, inputs: Vec<Var>, outputs: Vec<Var>) -> Self {
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

    pub fn closure(&self) -> &[Var] {
        &self.closure
    }

    pub fn inputs(&self) -> &[Var] {
        &self.inputs
    }

    pub fn outputs(&self) -> &[Var] {
        &self.outputs
    }
}

impl fmt::Display for Eqn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.outputs.len() {
            0 => write!(f, "_")?,
            1 => write!(f, "{}", self.outputs[0])?,
            _ => {
                write!(f, "(")?;
                for (i, o) in self.outputs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{o}")?;
                }
                write!(f, ")")?;
            }
        }
        write!(f, " = {}", self.op)?;
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
    closure_inputs: Vec<Var>,
    explicit_inputs: Vec<Var>,
    eqns: Vec<Eqn>,
    outputs: Vec<Var>,
}

impl Expr {
    pub fn new(
        closure_inputs: Vec<Var>,
        explicit_inputs: Vec<Var>,
        eqns: Vec<Eqn>,
        outputs: Vec<Var>,
    ) -> Self {
        Expr {
            closure_inputs,
            explicit_inputs,
            eqns,
            outputs,
        }
    }

    pub fn closure_inputs(&self) -> &[Var] {
        &self.closure_inputs
    }

    pub fn explicit_inputs(&self) -> &[Var] {
        &self.explicit_inputs
    }

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
        for (i, c) in self.closure_inputs.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{c}")?;
        }
        write!(f, " ; ")?;
        for (i, x) in self.explicit_inputs.iter().enumerate() {
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

/// Snapshot of a trace subgraph in jaxpr-like [`Expr`] form, plus hoisted closure tracers
/// (analogous to `consts` in [`ClosedJaxpr`](https://docs.jax.dev/en/latest/jaxpr.html)).
pub struct Capture {
    expr: Expr,
    closure: Vec<Tracer<Generic>>,
}

impl Capture {
    /// Materialize the trace subgraph needed for `outputs` into an [`Expr`].
    ///
    /// `inputs` is the ordered formal parameter list (jaxpr `invars` / `constvars` split).
    /// A boundary tracer is classified as **closure** (and listed in [`Capture::closure`]) when
    /// it is not among `inputs`, or when it holds a concrete [`crate::expr::value::Value`]. Otherwise
    /// it is an **explicit** input slot.
    pub fn from_tracers<'a, I, O>(inputs: I, outputs: O) -> Self
    where
        I: IntoIterator<Item = &'a Tracer<Generic>>,
        O: IntoIterator<Item = &'a Tracer<Generic>>,
    {
        let inputs: Vec<&Tracer<Generic>> = inputs.into_iter().collect();
        let outputs: Vec<&Tracer<Generic>> = outputs.into_iter().collect();

        let input_keys: HashSet<TracerKey> = inputs.iter().copied().map(TracerKey::from).collect();

        let mut trace_to_var: HashMap<TracerKey, Var> = HashMap::new();
        let mut next_id: usize = 0;
        let mut explicit_inputs = Vec::new();
        let mut closure_inputs = Vec::new();
        let mut closure_tracers = Vec::new();

        for t in &inputs {
            let k = TracerKey::from(*t);
            let v = Var::bind(next_id);
            next_id += 1;
            trace_to_var.insert(k, v.clone());
            if goes_to_closure(*t, &input_keys) {
                closure_inputs.push(v);
                closure_tracers.push((*t).clone());
            } else {
                explicit_inputs.push(v);
            }
        }

        if outputs.is_empty() {
            return Capture {
                expr: Expr::new(closure_inputs, explicit_inputs, vec![], vec![]),
                closure: closure_tracers,
            };
        }

        let invocations = collect_invocations(&outputs);
        let inv_set: HashSet<InvKey> = invocations.keys().copied().collect();

        let mut boundary: HashMap<TracerKey, Tracer<Generic>> = HashMap::new();
        for inv in invocations.values() {
            for inp in inv.inputs() {
                if !is_internal(inp, &inv_set) {
                    let k = TracerKey::from(inp);
                    if !trace_to_var.contains_key(&k) {
                        boundary.entry(k).or_insert_with(|| inp.clone());
                    }
                }
            }
        }
        for out in &outputs {
            if !is_internal(out, &inv_set) {
                let k = TracerKey::from(*out);
                if !trace_to_var.contains_key(&k) {
                    boundary.entry(k).or_insert_with(|| (*out).clone());
                }
            }
        }

        let mut boundary_keys: Vec<TracerKey> = boundary.keys().copied().collect();
        boundary_keys.sort_unstable();

        for k in boundary_keys {
            let t = boundary[&k].clone();
            let v = Var::bind(next_id);
            next_id += 1;
            trace_to_var.insert(k, v.clone());
            if goes_to_closure(&t, &input_keys) {
                closure_inputs.push(v);
                closure_tracers.push(t);
            } else {
                explicit_inputs.push(v);
            }
        }

        let ordered_invs = topo_sort_invocations(invocations);
        let mut eqns = Vec::with_capacity(ordered_invs.len());

        for inv in &ordered_invs {
            let input_vars: Vec<Var> = inv
                .inputs()
                .iter()
                .map(|inp| trace_to_var[&TracerKey::from(inp)].clone())
                .collect();

            let n_out = (0usize..)
                .take_while(|&i| inv.output_weak(i).is_some())
                .count();

            let mut out_vars = Vec::with_capacity(n_out);
            for i in 0..n_out {
                let v = Var::bind(next_id);
                next_id += 1;
                if let Some(w) = inv.output_weak(i) {
                    if let Some(t) = w.upgrade() {
                        trace_to_var.insert(TracerKey::from(&t), v.clone());
                    }
                }
                out_vars.push(v);
            }

            eqns.push(Eqn::new(
                inv.op().clone(),
                vec![],
                input_vars,
                out_vars,
            ));
        }

        let output_vars: Vec<Var> = outputs
            .iter()
            .map(|t| trace_to_var[&TracerKey::from(*t)].clone())
            .collect();

        Capture {
            expr: Expr::new(
                closure_inputs,
                explicit_inputs,
                eqns,
                output_vars,
            ),
            closure: closure_tracers,
        }
    }

    pub fn expr(&self) -> &Expr {
        &self.expr
    }

    pub fn closure(&self) -> &[Tracer<Generic>] {
        &self.closure
    }

    pub fn into_parts(self) -> (Expr, Vec<Tracer<Generic>>) {
        (self.expr, self.closure)
    }
}

/// Identity key for a [`Tracer`] node (stable for the lifetime of that `Arc<Trace>`).
#[derive(Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct TracerKey(usize);

impl From<&Tracer<Generic>> for TracerKey {
    fn from(t: &Tracer<Generic>) -> Self {
        TracerKey(t.addr())
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

fn goes_to_closure(t: &Tracer<Generic>, input_keys: &HashSet<TracerKey>) -> bool {
    let k = TracerKey::from(t);
    !input_keys.contains(&k) || t.as_value().is_some()
}

fn collect_invocations(outputs: &[&Tracer<Generic>]) -> HashMap<InvKey, Arc<Invocation>> {
    let mut invs: HashMap<InvKey, Arc<Invocation>> = HashMap::new();
    let mut pending: Vec<Arc<Invocation>> = Vec::new();

    for out in outputs {
        if let Some((inv, _)) = out.producer() {
            pending.push(inv);
        }
    }

    while let Some(inv) = pending.pop() {
        let k = InvKey::from(&inv);
        if invs.contains_key(&k) {
            continue;
        }
        invs.insert(k, inv.clone());
        for inp in inv.inputs() {
            if let Some((pred, _)) = inp.producer() {
                pending.push(pred);
            }
        }
    }

    invs
}

fn is_internal(t: &Tracer<Generic>, inv_keys: &HashSet<InvKey>) -> bool {
    match t.producer() {
        Some((inv, _)) => inv_keys.contains(&InvKey::from(&inv)),
        None => false,
    }
}

fn topo_sort_invocations(invs: HashMap<InvKey, Arc<Invocation>>) -> Vec<Arc<Invocation>> {
    let n = invs.len();
    if n == 0 {
        return vec![];
    }

    let mut indeg: HashMap<InvKey, usize> = invs.keys().map(|&k| (k, 0)).collect();
    let mut succ: HashMap<InvKey, Vec<InvKey>> = HashMap::new();

    for inv in invs.values() {
        let kid = InvKey::from(inv);
        for inp in inv.inputs() {
            if let Some((pred, _)) = inp.producer() {
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
    let mut outv = Vec::with_capacity(n);

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

    if outv.len() != n {
        panic!("gnx-expr: cycle in trace graph (topological sort failed)");
    }

    outv
}

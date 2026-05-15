use crate::expr::{Expr, VarScope, Var, Input, Effects, Eqn, Effect};
use crate::trace::{Generic, Invocation, Tracer, TraceRef};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::Arc;

/// Snapshot of a trace subgraph in jaxpr-like [`Expr`] form, plus hoisted closure [`TraceRef`]s
/// (analogous to `consts` in [`ClosedJaxpr`](https://docs.jax.dev/en/latest/jaxpr.html)).
pub struct Capture {
    expr: Expr,
    closure: Vec<TraceRef>,
}

impl Capture {
    /// Materialize the trace subgraph needed for `outputs` into an [`Expr`].
    pub fn from_trace_refs<'a, A: 'a, B: 'a, I, O>(inputs: I, outputs: O) -> Result<Self, ()>
    where
        A: AsRef<TraceRef>,
        B: AsRef<TraceRef>,
        I: IntoIterator<Item = &'a A>,
        O: IntoIterator<Item = &'a B>,
    {
        // Convert the types to references.
        let inputs: Vec<&TraceRef> = inputs.into_iter().map(|t| t.as_ref()).collect();
        let outputs: Vec<&TraceRef> = outputs.into_iter().map(|t| t.as_ref()).collect();
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
                match trace_to_var.get(&TracerKey::from(inp)) {
                    Some(var) => Input { var: var.clone(), effect: Effect::Unused},
                    None => {
                        let v = var_scope.create_var();
                        trace_to_var.insert(TracerKey::from(inp), v.clone());
                        closure_vars.push(v.clone());
                        closure_refs.push(inp.clone());
                        Input { var: v, effect: Effect::Unused }
                    }
                }
            }).collect();
            let inputs : Vec<Input> = inv.inputs().iter().map(|inp| {
                match trace_to_var.get(&TracerKey::from(inp)) {
                    Some(var) => Input { var: var.clone(), effect: Effect::Unused},
                    None => {
                        // Capture the trace as a closure variable.
                        let v = var_scope.create_var();
                        trace_to_var.insert(TracerKey::from(inp), v.clone());
                        closure_vars.push(v.clone());
                        closure_refs.push(inp.clone());
                        Input { var: v, effect: Effect::Unused }
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
        // Propagate an empty effects set backwards through the equations.
        // to get the default input-output effect association.
        let mut effects = Effects::new();
        for eqn in eqns.iter().rev() {
            eqn.propagate_effects(&mut effects);
        }
        let closure: Vec<Input> = closure_vars.into_iter().map(|v| Input { var: v, effect: effects[&v] }).collect();
        let inputs: Vec<Input> = input_vars.into_iter().map(|v| Input { var: v, effect: effects[&v] }).collect();
        Ok(Capture {
            expr: Expr::new( closure, inputs,
                eqns, output_vars, var_scope),
            closure: closure_refs,
        })
    }

    pub fn expr(&self) -> &Expr { &self.expr }
    pub fn closure(&self) -> &[TraceRef] { &self.closure }
    pub fn into_parts(self) -> (Expr, Vec<TraceRef>) { (self.expr, self.closure) }
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
        TracerKey::from(t.trace_ref())
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
            if let Some((pred, out_idx)) = inp.producer() {
                let key = InvKey::from(&pred);
                pending.push(pred);
                output_vars.entry(key).or_default().insert(out_idx, inp.clone());
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
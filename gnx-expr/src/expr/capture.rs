use crate::expr::builtins::update_op;
use crate::expr::{Effect, Eqn, Expr, Input, Effects, Var, VarScope};
use crate::trace::{
    CellKey, ContextID, Generic, Invocation, TraceCellRef, TraceContext, TraceOperand,
    TraceRef, Tracer,
};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

/// Runtime value hoisted into a captured expression's closure.
#[derive(Clone, Debug)]
pub enum CapturedValue {
    Ref(TraceRef),
    Cell(TraceCellRef),
}

/// Snapshot of a trace subgraph in jaxpr-like [`Expr`] form, plus hoisted closure values.
pub struct Capture {
    expr: Expr,
    closure: Vec<CapturedValue>,
}

impl Capture {
    /// Materialize the trace subgraph for `outputs` within `ctx` into an [`Expr`].
    pub fn from_context<'a, A: 'a, B: 'a, I, O>(
        ctx: &TraceContext,
        inputs: I,
        outputs: O,
    ) -> Result<Self, ()>
    where
        A: AsRef<TraceRef>,
        B: AsRef<TraceRef>,
        I: IntoIterator<Item = &'a A>,
        O: IntoIterator<Item = &'a B>,
    {
        let target = ctx.id();
        let inputs: Vec<&TraceRef> = inputs.into_iter().map(|t| t.as_ref()).collect();
        let outputs: Vec<&TraceRef> = outputs.into_iter().map(|t| t.as_ref()).collect();

        let mut var_scope = VarScope::new();
        let mut binding_to_var: HashMap<BindingKey, Var> = HashMap::new();
        let mut input_vars = Vec::new();
        let mut closure_vars = Vec::new();
        let mut closure_values = Vec::new();

        for inp in &inputs {
            let v = var_scope.create_var();
            binding_to_var.insert(BindingKey::Trace(TracerKey::from(*inp)), v.clone());
            input_vars.push(v);
        }

        let (invocations, output_vars) = collect_invocations(&outputs, target);
        let ordered_invs = topo_sort_invocations(invocations, target);
        let mut eqns = Vec::with_capacity(ordered_invs.len() + ctx.updates().len());

        let alloc_operand =
            |op: &TraceOperand,
             binding_to_var: &mut HashMap<BindingKey, Var>,
             closure_vars: &mut Vec<Var>,
             closure_values: &mut Vec<CapturedValue>,
             var_scope: &mut VarScope|
             -> Input {
                match op {
                    TraceOperand::Ref(r) => {
                        let key = BindingKey::Trace(TracerKey::from(r));
                        if let Some(var) = binding_to_var.get(&key).cloned() {
                            return Input::new(var, Effect::Read);
                        }
                        if r.context_id() != target {
                            let v = var_scope.create_var();
                            binding_to_var.insert(key, v.clone());
                            closure_vars.push(v.clone());
                            closure_values.push(CapturedValue::Ref(r.clone()));
                            return Input::new(v, Effect::Read);
                        }
                        let v = var_scope.create_var();
                        binding_to_var.insert(key, v.clone());
                        Input::new(v, Effect::Read)
                    }
                    TraceOperand::Cell(c) => {
                        let key = BindingKey::Cell(CellKey::from(c));
                        if let Some(var) = binding_to_var.get(&key).cloned() {
                            return Input::new(var, Effect::Read);
                        }
                        let v = var_scope.create_var();
                        binding_to_var.insert(key, v.clone());
                        let resolved = c.get();
                        if resolved.context_id() != target {
                            closure_vars.push(v.clone());
                            closure_values.push(CapturedValue::Cell(c.clone()));
                        }
                        Input::new(v, Effect::Read)
                    }
                }
            };

        for inv in &ordered_invs {
            let closure: Vec<Input> = inv
                .closure()
                .iter()
                .map(|op| {
                    alloc_operand(
                        op,
                        &mut binding_to_var,
                        &mut closure_vars,
                        &mut closure_values,
                        &mut var_scope,
                    )
                })
                .collect();
            let inputs: Vec<Input> = inv
                .inputs()
                .iter()
                .map(|op| {
                    alloc_operand(
                        op,
                        &mut binding_to_var,
                        &mut closure_vars,
                        &mut closure_values,
                        &mut var_scope,
                    )
                })
                .collect();

            let outs = &output_vars[&InvKey::from(inv)];
            let mut out_vars = Vec::with_capacity(inv.outputs());
            for out_idx in 0..inv.outputs() {
                match outs.get(&out_idx) {
                    Some(out_tracer) => {
                        let v = var_scope.create_var();
                        binding_to_var.insert(
                            BindingKey::Trace(TracerKey::from(out_tracer)),
                            v.clone(),
                        );
                        out_vars.push(v);
                    }
                    None => out_vars.push(Var::hole()),
                }
            }
            eqns.push(Eqn::new(inv.op().clone(), closure, inputs, out_vars));
        }

        for update in ctx.updates() {
            let cell_key = BindingKey::Cell(CellKey::from(&update.cell));
            let cell_var = match binding_to_var.get(&cell_key) {
                Some(v) => v.clone(),
                None => {
                    let v = var_scope.create_var();
                    binding_to_var.insert(cell_key, v.clone());
                    v
                }
            };
            let value_key = BindingKey::Trace(TracerKey::from(&update.value));
            let value_var = if let Some(v) = binding_to_var.get(&value_key) {
                v.clone()
            } else if update.value.context_id() != target {
                let v = var_scope.create_var();
                binding_to_var.insert(value_key, v.clone());
                closure_vars.push(v.clone());
                closure_values.push(CapturedValue::Ref(update.value.clone()));
                v
            } else {
                let v = var_scope.create_var();
                binding_to_var.insert(value_key, v.clone());
                v
            };
            let out_var = var_scope.create_var();
            eqns.push(Eqn::new(
                update_op(),
                vec![],
                vec![
                    Input::new(cell_var, Effect::Write),
                    Input::new(value_var, Effect::Read),
                ],
                vec![out_var],
            ));
        }

        for trace in &outputs {
            let k = BindingKey::Trace(TracerKey::from(*trace));
            if !binding_to_var.contains_key(&k) {
                return Err(());
            }
        }
        let output_vars: Vec<Var> = outputs
            .iter()
            .map(|t| binding_to_var[&BindingKey::Trace(TracerKey::from(*t))].clone())
            .collect();

        let mut effects = Effects::new();
        for eqn in eqns.iter().rev() {
            eqn.propagate_effects(&mut effects);
        }
        let closure: Vec<Input> = closure_vars
            .into_iter()
            .map(|v| Input::new(v, effects[&v]))
            .collect();
        let inputs: Vec<Input> = input_vars
            .into_iter()
            .map(|v| Input::new(v, effects[&v]))
            .collect();

        Ok(Capture {
            expr: Expr::new(closure, inputs, eqns, output_vars, var_scope),
            closure: closure_values,
        })
    }

    pub fn expr(&self) -> &Expr {
        &self.expr
    }
    pub fn closure(&self) -> &[CapturedValue] {
        &self.closure
    }
    pub fn into_parts(self) -> (Expr, Vec<CapturedValue>) {
        (self.expr, self.closure)
    }
}

#[derive(Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
enum BindingKey {
    Trace(TracerKey),
    Cell(CellKey),
}

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

#[derive(Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct InvKey(usize);

impl From<&Arc<Invocation>> for InvKey {
    fn from(inv: &Arc<Invocation>) -> Self {
        InvKey(Arc::as_ptr(inv) as usize)
    }
}

fn collect_invocations(
    outputs: &[&TraceRef],
    target: ContextID,
) -> (
    HashMap<InvKey, Arc<Invocation>>,
    HashMap<InvKey, BTreeMap<usize, TraceRef>>,
) {
    let mut invs: HashMap<InvKey, Arc<Invocation>> = HashMap::new();
    let mut output_vars: HashMap<InvKey, BTreeMap<usize, TraceRef>> = HashMap::new();
    let mut pending: Vec<Arc<Invocation>> = Vec::new();

    for out in outputs {
        if out.context_id() != target {
            continue;
        }
        if let Some((inv, pos)) = out.producer() {
            if inv.context_id() != target {
                continue;
            }
            if let Some(ret_idx) = pos.return_index() {
                let key = InvKey::from(&inv);
                pending.push(inv);
                output_vars
                    .entry(key)
                    .or_default()
                    .insert(ret_idx, (*out).clone());
            }
        }
    }

    while let Some(inv) = pending.pop() {
        let k = InvKey::from(&inv);
        if invs.contains_key(&k) {
            continue;
        }
        if inv.context_id() != target {
            continue;
        }
        invs.insert(k, inv.clone());
        for op in inv.closure().iter().chain(inv.inputs().iter()) {
            let trace = op.resolve();
            if trace.context_id() != target {
                continue;
            }
            if let Some((pred, pos)) = trace.producer() {
                if pred.context_id() != target {
                    continue;
                }
                let key = InvKey::from(&pred);
                pending.push(pred);
                if let Some(out_idx) = pos.return_index() {
                    output_vars.entry(key).or_default().insert(out_idx, trace);
                }
            }
        }
    }

    (invs, output_vars)
}

fn topo_sort_invocations(
    invs: HashMap<InvKey, Arc<Invocation>>,
    target: ContextID,
) -> Vec<Arc<Invocation>> {
    if invs.is_empty() {
        return vec![];
    }

    let mut indeg: HashMap<InvKey, usize> = invs.keys().map(|&k| (k, 0)).collect();
    let mut succ: HashMap<InvKey, Vec<InvKey>> = HashMap::new();

    for inv in invs.values() {
        let kid = InvKey::from(inv);
        for op in inv.closure().iter().chain(inv.inputs().iter()) {
            let inp = op.resolve();
            if inp.context_id() != target {
                continue;
            }
            if let Some((pred, _)) = inp.producer() {
                if pred.context_id() != target {
                    continue;
                }
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

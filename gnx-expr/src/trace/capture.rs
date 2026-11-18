use super::deferred::Projection;
use crate::trace::{
    CellKey, ContextID, Generic, Invocation, ObjectTracer, Trace, TraceContext, Tracer,
};
use crate::{HEqn, HExpr, HInstr, HSubExpr, Kind, Var, VarScope};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;

/// A DAG captured as HExpr, with bindings for its imported values and cells.
pub struct Capture {
    expr: Arc<HExpr>,
    closure: Vec<ObjectTracer>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CaptureError {
    UndefinedValue,
    DuplicateInput,
    LocalCellInput,
    Cycle,
}
impl fmt::Display for CaptureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for CaptureError {}

impl Capture {
    /// Traverse the DAG and materialize HExpr. No cell aliases are resolved here.
    pub fn from_context<'a, I, O>(
        ctx: &TraceContext,
        inputs: I,
        outputs: O,
    ) -> Result<Self, CaptureError>
    where
        I: IntoIterator<Item = &'a ObjectTracer>,
        O: IntoIterator<Item = &'a ObjectTracer>,
    {
        let target = ctx.id();
        let outputs: Vec<_> = outputs.into_iter().collect();
        let mut builder = Builder::new(target.clone());
        for input in inputs {
            let key = object_key(input);
            if builder.bindings.contains_key(&key) {
                return Err(CaptureError::DuplicateInput);
            }
            if let ObjectTracer::Cell(cell) = input {
                if cell.context_id() == &target && !cell.is_placeholder() {
                    return Err(CaptureError::LocalCellInput);
                }
            }
            let var = builder.scope.create_var();
            let kind = object_kind(input);
            builder.bindings.insert(key, var);
            builder.inputs.push((var, kind));
        }
        let mut roots = Vec::new();
        for output in &outputs {
            match output {
                ObjectTracer::Ref(value) => roots.push(value.clone()),
                ObjectTracer::Cell(cell)
                    if cell.context_id() == &target && !cell.is_placeholder() =>
                {
                    roots.push(cell.read_home());
                }
                ObjectTracer::Cell(_) => {}
            }
        }
        let ordered =
            order_invocations(&roots, ctx.last_aliased_write(), &target, &builder.bindings)?;
        for invocation in ordered {
            builder.invocation(&invocation)?;
        }
        let output_vars = outputs
            .iter()
            .map(|obj| builder.object(obj))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            expr: Arc::new(HExpr {
                closure: builder.closure_params,
                inputs: builder.inputs,
                instructions: builder.instructions,
                outputs: output_vars,
            }),
            closure: builder.closure_values,
        })
    }

    pub fn expr(&self) -> &HExpr {
        &self.expr
    }

    /// Share the expression without retaining the captured runtime bindings.
    pub fn shared_expr(&self) -> Arc<HExpr> {
        self.expr.clone()
    }

    /// Bind the current arguments to this body without lowering it.
    /// Cell outputs and writes are represented by the resulting call's projections.
    pub fn call(&self, inputs: &[ObjectTracer]) -> Vec<ObjectTracer> {
        Invocation::defer_call(self.expr.clone(), &self.closure, inputs)
    }
    pub fn closure(&self) -> &[ObjectTracer] {
        &self.closure
    }
}
impl From<Capture> for (HExpr, Vec<ObjectTracer>) {
    fn from(capture: Capture) -> Self {
        (Arc::unwrap_or_clone(capture.expr), capture.closure)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TracerKey(usize);
impl From<&Trace> for TracerKey {
    fn from(trace: &Trace) -> Self {
        Self(Arc::as_ptr(&trace.0) as usize)
    }
}
impl From<&Tracer<Generic>> for TracerKey {
    fn from(trace: &Tracer<Generic>) -> Self {
        Self::from(trace.trace_ref())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct InvKey(usize);
impl From<&Arc<Invocation>> for InvKey {
    fn from(inv: &Arc<Invocation>) -> Self {
        Self(Arc::as_ptr(inv) as usize)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum BindingKey {
    Value(TracerKey),
    Result(InvKey, usize),
    Cell(CellKey),
}
fn value_key(value: &Trace) -> BindingKey {
    match value.producer_with_index() {
        Some((inv, index)) => BindingKey::Result(InvKey::from(inv), index),
        None => BindingKey::Value(TracerKey::from(value)),
    }
}
fn object_key(obj: &ObjectTracer) -> BindingKey {
    match obj {
        ObjectTracer::Ref(value) => value_key(value),
        ObjectTracer::Cell(cell) => BindingKey::Cell(CellKey::from(cell)),
    }
}
fn object_kind(obj: &ObjectTracer) -> Kind {
    match obj {
        ObjectTracer::Ref(value) => Kind::Value(value.info().clone()),
        ObjectTracer::Cell(cell) => Kind::Cell(cell.type_info().clone()),
    }
}

struct Builder {
    target: ContextID,
    scope: VarScope,
    bindings: BTreeMap<BindingKey, Var>,
    inputs: Vec<(Var, Kind)>,
    closure_params: Vec<(Var, Kind)>,
    closure_values: Vec<ObjectTracer>,
    instructions: Vec<HInstr>,
}
impl Builder {
    fn new(target: ContextID) -> Self {
        Self {
            target,
            scope: VarScope::new(),
            bindings: BTreeMap::new(),
            inputs: vec![],
            closure_params: vec![],
            closure_values: vec![],
            instructions: vec![],
        }
    }
    fn import(&mut self, obj: ObjectTracer) -> Var {
        let var = self.scope.create_var();
        let kind = object_kind(&obj);
        self.bindings.insert(object_key(&obj), var);
        self.closure_params.push((var, kind));
        self.closure_values.push(obj);
        var
    }
    fn value(&mut self, value: &Trace) -> Result<Var, CaptureError> {
        if let Some(var) = self.bindings.get(&value_key(value)) {
            return Ok(*var);
        }
        if value.context_id() != &self.target || value.try_concrete().is_some() {
            return Ok(self.import(ObjectTracer::Ref(value.clone())));
        }
        Err(CaptureError::UndefinedValue)
    }
    fn object(&mut self, obj: &ObjectTracer) -> Result<Var, CaptureError> {
        if let Some(var) = self.bindings.get(&object_key(obj)) {
            return Ok(*var);
        }
        match obj {
            ObjectTracer::Ref(value) => self.value(value),
            ObjectTracer::Cell(cell) if cell.context_id() != &self.target => {
                Ok(self.import(obj.clone()))
            }
            ObjectTracer::Cell(cell) if cell.is_placeholder() => Err(CaptureError::UndefinedValue),
            ObjectTracer::Cell(cell) => {
                // A local cell was already functionalized during tracing. Only
                // materialize its identity if the handle escapes as an output.
                let initial = self.value(&cell.read_home())?;
                let var = self.scope.create_var();
                self.bindings.insert(object_key(obj), var);
                self.instructions.push(HInstr::CreateCell {
                    cell: var,
                    initial,
                    kind: object_kind(obj),
                });
                Ok(var)
            }
        }
    }
    fn outputs(&mut self, inv: &Arc<Invocation>) -> Vec<(Var, Kind)> {
        inv.outputs()
            .iter()
            .enumerate()
            .map(|(index, info)| {
                let var = self.scope.create_var();
                self.bindings
                    .entry(BindingKey::Result(InvKey::from(inv), index))
                    .or_insert(var);
                (var, Kind::Value(info.clone()))
            })
            .collect()
    }

    fn invocation(&mut self, inv: &Arc<Invocation>) -> Result<(), CaptureError> {
        let instr = match inv.as_ref() {
            Invocation::Apply {
                op,
                inputs,
                subexprs,
                ..
            } => {
                let inputs = inputs
                    .iter()
                    .map(|value| self.value(value))
                    .collect::<Result<_, _>>()?;
                let subexprs = subexprs
                    .iter()
                    .map(|(expr, captures)| {
                        let captures = captures
                            .iter()
                            .map(|value| self.value(value))
                            .collect::<Result<_, _>>()?;
                        Ok(HSubExpr::new(expr.clone(), captures))
                    })
                    .collect::<Result<_, CaptureError>>()?;
                let outputs = self.outputs(inv);
                HInstr::Apply(HEqn::new(op.clone(), subexprs, inputs, outputs))
            }
            Invocation::DeferredCall(call) => {
                let captures = call
                    .closure
                    .iter()
                    .map(|object| self.object(object))
                    .collect::<Result<_, _>>()?;
                let inputs = call
                    .inputs
                    .iter()
                    .map(|object| self.object(object))
                    .collect::<Result<_, _>>()?;
                let results: Vec<_> = call
                    .body
                    .outputs()
                    .iter()
                    .map(|var| {
                        (
                            self.scope.create_var(),
                            call.body.kind(var).expect("call result kind").clone(),
                        )
                    })
                    .collect();
                self.instructions.push(HInstr::Apply(HEqn::new(
                    crate::builtins::call_op(),
                    vec![HSubExpr::new(call.body.clone(), captures)],
                    inputs,
                    results.clone(),
                )));
                for (index, projection) in call.results.iter().enumerate() {
                    let key = BindingKey::Result(InvKey::from(inv), index);
                    let cell = match projection {
                        Projection::Value(result_index) => {
                            self.bindings.insert(key, results[*result_index].0);
                            continue;
                        }
                        Projection::CellContents(index) => results[*index].0,
                        Projection::LocalUpdate(cell) => {
                            self.object(&ObjectTracer::Cell(cell.clone()))?
                        }
                    };
                    let output = self.scope.create_var();
                    self.bindings.insert(key, output);
                    let kind = Kind::Value(call.outputs[index].clone());
                    self.instructions
                        .push(HInstr::ReadCell { cell, output, kind });
                }
                return Ok(());
            }
            Invocation::ReadCell { cell, .. } => {
                let cell = self.object(&ObjectTracer::Cell(cell.clone()))?;
                let (output, kind) = self.outputs(inv).pop().expect("cell read has one result");
                HInstr::ReadCell { cell, output, kind }
            }
            Invocation::WriteCell { cell, value, .. } => {
                let cell = self.object(&ObjectTracer::Cell(cell.clone()))?;
                let value = self.value(value)?;
                HInstr::WriteCell { cell, value }
            }
        };
        self.instructions.push(instr);
        Ok(())
    }
}

/// Iterative DFS preserves both data and mutable-access edges without a trace
/// log. Explicit value inputs cut the graph at their producer boundaries.
fn order_invocations(
    roots: &[Trace],
    last_write: Option<Arc<Invocation>>,
    target: &ContextID,
    inputs: &BTreeMap<BindingKey, Var>,
) -> Result<Vec<Arc<Invocation>>, CaptureError> {
    let producer = |value: &Trace| {
        if value.context_id() == target && !inputs.contains_key(&value_key(value)) {
            value
                .producer()
                .filter(|inv| inv.context_id() == target)
                .cloned()
        } else {
            None
        }
    };
    let mut pending = Vec::new();
    if let Some(inv) = last_write {
        pending.push((inv, false));
    }
    for root in roots {
        if let Some(inv) = producer(root) {
            pending.push((inv, false));
        }
    }
    let mut visiting = BTreeSet::new();
    let mut finished = BTreeSet::new();
    let mut ordered = Vec::new();
    while let Some((inv, exit)) = pending.pop() {
        let key = InvKey::from(&inv);
        if finished.contains(&key) {
            continue;
        }
        if exit {
            visiting.remove(&key);
            finished.insert(key);
            ordered.push(inv);
            continue;
        }
        if !visiting.insert(key) {
            return Err(CaptureError::Cycle);
        }
        pending.push((inv.clone(), true));
        for parent in inv.aliasing_parents() {
            if parent.context_id() == target {
                pending.push((parent.clone(), false));
            }
        }
        for value in inv.value_parents() {
            if let Some(pred) = producer(value) {
                pending.push((pred, false));
            }
        }
    }
    Ok(ordered)
}

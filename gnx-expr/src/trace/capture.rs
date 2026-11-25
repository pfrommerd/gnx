//! Capture a slice of the trace DAG as an HExpr. Both entry points use the
//! same producer walk and emitter; they differ in their capture boundaries.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;

use crate::eval::EVAL_DAG;
use crate::{
    HEqnSource, HExpr, HExprBuilder, HInstrSource, HRootExpr, HSubExprBuilder, Kind, Symbol, Var,
    VarInfo, VarScope,
};

use super::{
    CellKey, CellTrace, ContextID, Generic, Invocation, ObjectTracer, ObjectTracerRef, Trace,
    TraceContext, Tracer, apply::Projection,
};

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{self:?}")]
pub enum CaptureError {
    UndefinedValue,
    DuplicateInput,
    LocalCellInput,
    Cycle,
    #[error("capture roots and deferred dependencies must belong to the same scope")]
    MixedScopes,
    UndefinedSymbol(Symbol),
    UnboundSymbolInput(Symbol),
    /// The captured expression is not well formed.
    #[error("invalid capture: {0}")]
    Invalid(String),
}

/// A captured DAG slice and its bindings in expression order.
/// `from_context` preserves trace effects; `from_roots` selects the live results
/// of a slice for evaluation.
pub struct Capture {
    expr: Arc<HRootExpr>,
    closure: Vec<ObjectTracer>,
    inputs: Vec<ObjectTracer>,
    outputs: Vec<ObjectTracer>,
}

// Show the IR and its boundary metadata, not the retained runtime values.
impl fmt::Display for Capture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.expr.view(), f)
    }
}

impl fmt::Debug for Capture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl Capture {
    /// Capture the requested outputs and the context's pending cell effects.
    /// Explicit inputs cut their producer edges. Cell aliases remain unresolved.
    pub fn from_context<'a>(
        ctx: &TraceContext,
        inputs: impl IntoIterator<Item = &'a ObjectTracer>,
        outputs: impl IntoIterator<Item = &'a ObjectTracer>,
    ) -> Result<Self, CaptureError> {
        let mut emitter = Emitter::new(ctx.id(), Mode::Traced);
        for input in inputs {
            emitter.bind_input(input.as_ref())?;
        }
        let outputs: Vec<_> = outputs.into_iter().cloned().collect();
        let roots = outputs.iter().filter_map(|output| match output.as_ref() {
            ObjectTracerRef::Ref(value) => Some(value.internal_clone()),
            ObjectTracerRef::Cell(cell)
                if cell.context_id() == &emitter.target && !cell.is_placeholder() =>
            {
                Some(cell.read_home())
            }
            ObjectTracerRef::Cell(_) => None,
        });
        let slice = emitter.discover(roots, ctx.last_aliased_write())?;
        emitter.emit(&slice, outputs)?.finish(Some(ctx), true)
    }

    /// Snapshot cell home values without recording a cell read. The home
    /// value, not the cell itself, determines the root's scope.
    pub fn from_objects<'a>(
        roots: impl IntoIterator<Item = ObjectTracerRef<'a>>,
    ) -> Result<Self, CaptureError> {
        Self::capture(roots.into_iter().map(|root| match root {
            ObjectTracerRef::Ref(trace) => trace.internal_clone(),
            ObjectTracerRef::Cell(cell) => cell.read_home(),
        }))
    }

    /// Capture existing values in one scope, cutting at concrete storage.
    /// Placeholders become bindings. Requested pending results come before
    /// live-outside results; concrete roots need no work.
    pub fn from_roots<'a>(
        roots: impl IntoIterator<Item = &'a Trace>,
    ) -> Result<Self, CaptureError> {
        Self::capture(roots.into_iter().map(Trace::internal_clone))
    }

    fn capture(roots: impl IntoIterator<Item = Trace>) -> Result<Self, CaptureError> {
        // Run the caller's iterator outside the publication lock.
        let roots: Vec<_> = roots.into_iter().collect();
        let scope = roots
            .first()
            .map_or(ContextID::ROOT, |root| root.context_id().clone());
        if roots.iter().any(|root| root.context_id() != &scope) {
            return Err(CaptureError::MixedScopes);
        }
        let emitter = Emitter::new(scope, Mode::Eval);
        let pinned;
        let capture = {
            let _guard = EVAL_DAG.lock().unwrap();
            let slice = emitter.discover(roots.iter().map(Trace::internal_clone), None)?;
            // Weak result slots can disappear if another thread drops their
            // last public handle. Keep any upgraded slots until we unlock.
            pinned = slice.pin_results();
            let outputs = slice.eval_outputs(&roots, &pinned);
            emitter.emit(&slice, outputs)
        };
        capture?.finish(TraceContext::current().as_ref(), false)
    }

    pub fn expr(&self) -> HExpr<'_> {
        self.expr.view()
    }

    /// The root the captured expression is stored in.
    pub fn root(&self) -> &Arc<HRootExpr> {
        &self.expr
    }

    /// Imported values and cells, in `expr().closure()` order.
    pub fn closure(&self) -> &[ObjectTracer] {
        &self.closure
    }

    /// Explicit inputs; root capture imports dependencies through the closure.
    pub fn inputs(&self) -> &[ObjectTracer] {
        &self.inputs
    }

    /// The original tracer handles corresponding to `expr().outputs()`.
    pub fn outputs(&self) -> &[ObjectTracer] {
        &self.outputs
    }

    pub fn into_parts(
        self,
    ) -> (
        Arc<HRootExpr>,
        Vec<ObjectTracer>,
        Vec<ObjectTracer>,
        Vec<ObjectTracer>,
    ) {
        (self.expr, self.closure, self.inputs, self.outputs)
    }
}

impl From<Capture> for (Arc<HRootExpr>, Vec<ObjectTracer>) {
    fn from(capture: Capture) -> Self {
        (capture.expr, capture.closure)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TracerKey(usize);

impl From<&Trace> for TracerKey {
    fn from(trace: &Trace) -> Self {
        Self(Arc::as_ptr(&trace.data) as usize)
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

impl From<&Trace> for BindingKey {
    fn from(value: &Trace) -> Self {
        match value.origin() {
            Some((inv, index)) => Self::Result(InvKey(inv.as_ptr() as usize), index),
            None => Self::Value(value.into()),
        }
    }
}

fn binding(object: ObjectTracerRef<'_>) -> (BindingKey, Kind, VarInfo) {
    match object {
        ObjectTracerRef::Ref(value) => (value.into(), Kind::Value, value.info().clone()),
        ObjectTracerRef::Cell(cell) => (
            BindingKey::Cell(CellKey::from(cell)),
            Kind::Cell,
            cell.type_info().clone(),
        ),
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Traced,
    Eval,
}

struct Slice {
    ordered: Vec<Arc<Invocation>>,
    members: BTreeSet<InvKey>,
}

impl Slice {
    fn pin_results(&self) -> Vec<Trace> {
        let mut pinned = Vec::new();
        for inv in &self.ordered {
            let slots = inv.result_slots();
            pinned.extend(
                slots
                    .iter()
                    .filter_map(|slot| slot.upgrade())
                    .map(Trace::internal),
            );
        }
        pinned
    }

    fn eval_outputs(&self, roots: &[Trace], pinned: &[Trace]) -> Vec<ObjectTracer> {
        let mut selected = BTreeSet::new();
        roots
            .iter()
            .chain(
                pinned.iter().filter(|trace| {
                    trace.is_live_outside(|user| self.members.contains(&InvKey(user)))
                }),
            )
            .filter(|trace| {
                trace.try_concrete().is_none() && selected.insert(TracerKey::from(*trace))
            })
            .map(|trace| ObjectTracer::Ref(trace.internal_clone()))
            .collect()
    }
}

type BoundParam = ((Var, Kind, VarInfo), ObjectTracer);

struct Emitter {
    target: ContextID,
    mode: Mode,
    vars: VarScope,
    bindings: BTreeMap<BindingKey, Var>,
    inputs: Vec<BoundParam>,
    closure: Vec<BoundParam>,
    instructions: Vec<HInstrSource>,
}

impl Emitter {
    fn new(target: ContextID, mode: Mode) -> Self {
        Self {
            target,
            mode,
            vars: VarScope::new(),
            bindings: BTreeMap::new(),
            inputs: vec![],
            closure: vec![],
            instructions: vec![],
        }
    }

    fn bind_input(&mut self, object: ObjectTracerRef<'_>) -> Result<(), CaptureError> {
        let (key, kind, info) = binding(object);
        if self.bindings.contains_key(&key) {
            return Err(CaptureError::DuplicateInput);
        }
        if let ObjectTracerRef::Cell(cell) = object {
            if cell.context_id() == &self.target && !cell.is_placeholder() {
                return Err(CaptureError::LocalCellInput);
            }
        }
        let var = self.vars.create_var();
        self.bindings.insert(key, var);
        self.inputs.push(((var, kind, info), object.to_owned()));
        Ok(())
    }

    /// Postorder DFS over producers. Explicit inputs and concrete values cut
    /// dependencies. Only a traced region has an ambient cell-effect root.
    fn discover(
        &self,
        roots: impl IntoIterator<Item = Trace>,
        last_write: Option<Arc<Invocation>>,
    ) -> Result<Slice, CaptureError> {
        let producer = |value: &Trace| {
            if value.context_id() != &self.target {
                if self.mode == Mode::Eval && value.try_concrete().is_none() {
                    return Err(CaptureError::MixedScopes);
                }
                return Ok(None);
            }
            if value.try_concrete().is_some() || self.bindings.contains_key(&value.into()) {
                return Ok(None);
            }
            Ok(value
                .producer()
                .filter(|inv| inv.context_id() == &self.target))
        };
        let mut pending: Vec<_> = last_write.into_iter().map(|inv| (inv, false)).collect();
        for root in roots {
            if let Some(inv) = producer(&root)? {
                pending.push((inv, false));
            }
        }
        let mut visiting = BTreeSet::new();
        let mut slice = Slice {
            ordered: Vec::new(),
            members: BTreeSet::new(),
        };
        while let Some((inv, exit)) = pending.pop() {
            let key = InvKey::from(&inv);
            if slice.members.contains(&key) {
                continue;
            }
            if exit {
                visiting.remove(&key);
                slice.members.insert(key);
                slice.ordered.push(inv);
                continue;
            }
            if !visiting.insert(key) {
                return Err(CaptureError::Cycle);
            }
            pending.push((inv.clone(), true));
            pending.extend(
                inv.aliasing_parents()
                    .iter()
                    .filter(|inv| inv.context_id() == &self.target)
                    .map(|inv| (inv.clone(), false)),
            );
            for value in inv.value_parents() {
                if let Some(pred) = producer(value)? {
                    pending.push((pred, false));
                }
            }
        }
        Ok(slice)
    }

    fn emit(
        mut self,
        slice: &Slice,
        outputs: Vec<ObjectTracer>,
    ) -> Result<Unfinished, CaptureError> {
        for inv in &slice.ordered {
            self.emit_invocation(inv)?;
        }
        let output_vars = self.object_args(&outputs)?;
        let (closure_params, closure) = self.closure.into_iter().unzip();
        let (input_params, inputs) = self.inputs.into_iter().unzip();
        Ok(Unfinished {
            expr: HExprBuilder::from_parts(
                vec![],
                closure_params,
                input_params,
                self.instructions,
                output_vars,
            ),
            closure,
            inputs,
            outputs,
        })
    }

    fn import(&mut self, object: ObjectTracerRef<'_>) -> Var {
        let (key, kind, info) = binding(object);
        let var = self.vars.create_var();
        self.bindings.insert(key, var);
        let value = match object {
            ObjectTracerRef::Ref(trace) if self.mode == Mode::Eval => {
                ObjectTracer::Ref(trace.internal_clone())
            }
            _ => object.to_owned(),
        };
        self.closure.push(((var, kind, info), value));
        var
    }

    fn value(&mut self, value: &Trace) -> Result<Var, CaptureError> {
        if let Some(&var) = self.bindings.get(&value.into()) {
            return Ok(var);
        }
        if value.context_id() != &self.target
            || value.try_concrete().is_some()
            || (self.mode == Mode::Eval && value.is_placeholder())
        {
            return Ok(self.import(ObjectTracerRef::Ref(value)));
        }
        Err(CaptureError::UndefinedValue)
    }

    fn object(&mut self, object: ObjectTracerRef<'_>) -> Result<Var, CaptureError> {
        match object {
            ObjectTracerRef::Ref(value) => self.value(value),
            ObjectTracerRef::Cell(cell) => self.cell(cell),
        }
    }

    fn cell(&mut self, cell: &CellTrace) -> Result<Var, CaptureError> {
        let key = BindingKey::Cell(CellKey::from(cell));
        if let Some(&var) = self.bindings.get(&key) {
            return Ok(var);
        }
        if cell.context_id() != &self.target || (self.mode == Mode::Eval && cell.is_placeholder()) {
            return Ok(self.import(ObjectTracerRef::Cell(cell)));
        }
        if cell.is_placeholder() {
            return Err(CaptureError::UndefinedValue);
        }
        // A local cell has already been functionalized. Only build a cell
        // identity at the boundary if it escapes as an output.
        let initial = self.value(&cell.read_home())?;
        let var = self.vars.create_var();
        self.bindings.insert(key, var);
        self.instructions.push(HInstrSource::CreateCell {
            cell: var,
            initial,
            info: cell.type_info().clone(),
        });
        Ok(var)
    }

    fn object_args(&mut self, objects: &[ObjectTracer]) -> Result<Vec<Var>, CaptureError> {
        objects
            .iter()
            .map(|value| self.object(value.as_ref()))
            .collect()
    }

    fn define_results(&mut self, inv: &Arc<Invocation>) -> Vec<(Var, Kind, VarInfo)> {
        inv.outputs()
            .iter()
            .enumerate()
            .map(|(index, info)| {
                let var = self.vars.create_var();
                self.bindings
                    .entry(BindingKey::Result(inv.into(), index))
                    .or_insert(var);
                (var, Kind::Value, info.clone())
            })
            .collect()
    }

    fn emit_invocation(&mut self, inv: &Arc<Invocation>) -> Result<(), CaptureError> {
        let instr = match inv.as_ref() {
            Invocation::Apply(apply) => return self.emit_apply(inv.into(), apply),
            Invocation::ReadCell { cell, .. } => {
                let cell = self.cell(cell)?;
                let (output, _, info) = self
                    .define_results(inv)
                    .pop()
                    .expect("cell read has one result");
                HInstrSource::ReadCell { cell, output, info }
            }
            Invocation::WriteCell { cell, value, .. } => HInstrSource::WriteCell {
                cell: self.cell(cell)?,
                value: self.value(value)?,
            },
        };
        self.instructions.push(instr);
        Ok(())
    }

    fn emit_apply(&mut self, key: InvKey, apply: &super::Apply) -> Result<(), CaptureError> {
        use super::ApplyProjection::{CellContents, Value};
        let regions = apply
            .regions
            .iter()
            .map(|region| {
                Ok(HSubExprBuilder::new(region.body.clone())
                    .with_vars(self.object_args(&region.captures)?)
                    .with_symbols(region.symbol_captures.iter().copied()))
            })
            .collect::<Result<_, CaptureError>>()?;
        let inputs = self.object_args(&apply.inputs)?;
        let results: Vec<_> = apply
            .kinds
            .iter()
            .map(|(kind, info)| (self.vars.create_var(), *kind, info.clone()))
            .collect();
        self.instructions.push(HInstrSource::Apply(HEqnSource::new(
            apply.op.clone(),
            regions,
            inputs,
            results.clone(),
        )));
        for (index, projection) in apply.results.iter().enumerate() {
            let var = match projection {
                Projection::Output(Value(result)) => results[*result].0,
                projection => {
                    let cell = match projection {
                        Projection::Output(CellContents(result)) => results[*result].0,
                        Projection::LocalUpdate(cell) => self.cell(cell)?,
                        Projection::Output(Value(_)) => unreachable!(),
                    };
                    let output = self.vars.create_var();
                    self.instructions.push(HInstrSource::ReadCell {
                        cell,
                        output,
                        info: apply.outputs[index].clone(),
                    });
                    output
                }
            };
            // A result already bound as an input keeps that binding.
            self.bindings
                .entry(BindingKey::Result(key, index))
                .or_insert(var);
        }
        Ok(())
    }
}

// A captured expression whose symbols are yet to be declared.
struct Unfinished {
    expr: HExprBuilder,
    closure: Vec<ObjectTracer>,
    inputs: Vec<ObjectTracer>,
    outputs: Vec<ObjectTracer>,
}

impl Unfinished {
    // Declares the symbols the expression takes from its context: those it
    // uses that its inputs do not bind.
    fn finish(mut self, ctx: Option<&TraceContext>, traced: bool) -> Result<Capture, CaptureError> {
        let e = &mut self.expr;
        let mut seen = BTreeSet::new();
        let mut declare = |s: Symbol| -> Result<bool, CaptureError> {
            if !seen.insert(s) {
                return Ok(false);
            }
            if !ctx.is_some_and(|c| c.has_symbol(s)) {
                return Err(CaptureError::UndefinedSymbol(s));
            }
            Ok(true)
        };
        for (_, _, info) in &e.inputs {
            for &s in info.symbols() {
                declare(s)?;
            }
        }
        let mut free = Vec::new();
        for s in body_symbols(e) {
            if declare(s)? {
                free.push(s);
            }
        }
        e.closure_symbols = free;
        if traced
            && let Some(ctx) = ctx
            && let Some(s) = e.closure_symbols.iter().find(|s| ctx.owns_symbol(**s))
        {
            return Err(CaptureError::UnboundSymbolInput(*s));
        }
        e.reserve();
        let expr = (self.expr.build()).map_err(|error| CaptureError::Invalid(error.to_string()))?;
        Ok(Capture {
            expr,
            closure: self.closure,
            inputs: self.inputs,
            outputs: self.outputs,
        })
    }
}

// Symbols used by closure metadata, instructions and nested attachments.
fn body_symbols(expr: &HExprBuilder) -> Vec<Symbol> {
    let mut symbols = Vec::new();
    for (_, _, info) in &expr.closure_inputs {
        symbols.extend(info.symbols());
    }
    for instr in &expr.instrs {
        match instr {
            HInstrSource::ReadCell { info, .. } | HInstrSource::CreateCell { info, .. } => {
                symbols.extend(info.symbols())
            }
            HInstrSource::Apply(eqn) => {
                for (_, _, info) in eqn.outputs() {
                    symbols.extend(info.symbols());
                }
                for region in eqn.subexprs() {
                    symbols.extend(region.provided_symbols());
                }
            }
            HInstrSource::WriteCell { .. } => {}
        }
    }
    symbols
}

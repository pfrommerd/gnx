//! HExpr-to-Expr lowering, input specifications, and execution-boundary metadata.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;

use crate::{
    Eqn, Expr, HEqn, HExpr, HInstr, HSubExpr, Kind, Op, Ownership, SubExpr, ValueInfo, Var,
    VarScope,
};

/// Canonical source of a supplied value or cell, indexed in the original HExpr
/// argument lists. Aliases share a source. Explicit inputs take precedence over
/// closure positions; kind and ownership must agree across all aliases.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum InputId {
    Closure(usize),
    Input(usize),
}

/// One (identity, kind, ownership) entry per HExpr parameter, in declaration
/// order. Cell ownership applies to its initial immutable contents, not its handle.
#[derive(Clone, Debug, Default)]
pub struct InputSpec {
    pub closure: Vec<(InputId, Kind, Ownership)>,
    pub inputs: Vec<(InputId, Kind, Ownership)>,
}
impl InputSpec {
    /// Derive shared identities and kinds from actual arguments. Ownership is
    /// explicit; equal values are not merged unless their trace identities match.
    pub fn from_args(
        closure: &[(crate::ObjectTracer, Ownership)],
        inputs: &[(crate::ObjectTracer, Ownership)],
    ) -> Self {
        #[derive(PartialEq, Eq, PartialOrd, Ord)]
        enum Identity {
            Value(crate::TracerKey),
            Cell(crate::CellKey),
        }
        let mut identities = BTreeMap::new();
        let mut bind =
            |(index, (object, ownership)): (InputId, &(crate::ObjectTracer, Ownership))| {
                let (identity, kind) = match object {
                    crate::ObjectTracer::Ref(value) => (
                        Identity::Value(crate::TracerKey::from(value)),
                        Kind::Value(value.info().clone()),
                    ),
                    crate::ObjectTracer::Cell(cell) => (
                        Identity::Cell(crate::CellKey::from(cell)),
                        Kind::Cell(cell.type_info().clone()),
                    ),
                };
                (
                    *identities.entry(identity).or_insert(index),
                    kind,
                    *ownership,
                )
            };
        // Visit explicit inputs first so captures of the same handle use the
        // first explicit-input offset, not a closure offset or a dense ID.
        let inputs = inputs
            .iter()
            .enumerate()
            .map(|(index, value)| bind((InputId::Input(index), value)))
            .collect();
        let closure = closure
            .iter()
            .enumerate()
            .map(|(index, value)| bind((InputId::Closure(index), value)))
            .collect();
        Self { closure, inputs }
    }
}

// Internal identity used while functionalizing cells; never part of the boundary API.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum CellBinding {
    Input(InputId),
    Local(usize),
}

/// OutputValue/NewOutputCell index the owned return list of expr. InputValue/
/// NewInputCell reference an original supplied immutable value (or initial cell
/// contents), not a low-level return. InputCell returns the original cell handle.
/// Equal new-cell IDs must materialize the same fresh handle per execution.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Output {
    OutputValue { output_idx: usize },
    InputValue { input_id: InputId },
    InputCell { input_id: InputId },
    NewInputCell { id: usize, input_id: InputId },
    NewOutputCell { id: usize, output_idx: usize },
}

/// Pure program and its execution-boundary mappings. Closure and inputs correspond
/// one to one with expr.closure() and expr.inputs(), respectively. Closure-only
/// identities stay in closure; any identity appearing in explicit inputs belongs
/// to inputs, even when also captured. InputSpec supplies kinds and ownership.
///
/// outputs starts with user results, followed by one value descriptor per update.
/// updates indexes that descriptor table, NOT expr.outputs(). Its sources are
/// always InputValue or OutputValue. results() excludes the update-source suffix.
/// Physical owned returns are deduplicated; descriptors may share their slots.
/// A wrapper must retain/copy when distributing a value to multiple owners or
/// storing an input passthrough beyond its lifetime. No implicit clone is emitted.
#[derive(Clone, Debug)]
pub struct LoweredExpr {
    spec: InputSpec,
    snapshot_inputs: BTreeSet<InputId>,
    pub expr: Arc<Expr>,
    trace_body: Arc<HExpr>,
    /// Metadata for each physical owned return, in expr.outputs() order.
    pub output_infos: Vec<ValueInfo>,
    pub closure: Vec<InputId>,
    pub inputs: Vec<InputId>,
    pub outputs: Vec<Output>,
    pub updates: Vec<(InputId, usize)>,
}
impl LoweredExpr {
    /// The ownership-free body used when emitting this specialization into a trace.
    pub fn trace_body(&self) -> &Arc<HExpr> {
        &self.trace_body
    }
    pub fn results(&self) -> &[Output] {
        &self.outputs[..self.outputs.len() - self.updates.len()]
    }

    /// Emit one immutable call and materialize its boundary. All required cell
    /// snapshots are taken before any updates; repeated local-cell IDs share one
    /// fresh handle. Returning an input cell alone does not read its contents.
    ///
    /// The caller must supply the alias specialization used during lowering.
    /// Sources are resolved directly through stored InputId offsets; this method
    /// checks arity and metadata but does not rediscover or validate identities.
    pub fn call(
        &self,
        closure: &[crate::ObjectTracer],
        inputs: &[crate::ObjectTracer],
    ) -> Result<Vec<crate::ObjectTracer>, CallError> {
        use crate::{CellTrace, ObjectTracer, Trace};
        let spec = &self.spec;
        if closure.len() != spec.closure.len() || inputs.len() != spec.inputs.len() {
            return Err(CallError::Arity);
        }
        let binding = |id: InputId| {
            match id {
                InputId::Closure(index) => closure.get(index),
                InputId::Input(index) => inputs.get(index),
            }
            .ok_or(CallError::MissingInput(id))
        };
        for ((id, kind, _), object) in spec
            .closure
            .iter()
            .chain(&spec.inputs)
            .zip(closure.iter().chain(inputs))
        {
            let valid = match (kind, object) {
                (Kind::Value(info), ObjectTracer::Ref(value)) => info == value.info(),
                (Kind::Cell(info), ObjectTracer::Cell(cell)) => info == cell.type_info(),
                _ => false,
            };
            if !valid {
                return Err(CallError::ArgumentType(*id));
            }
        }
        self.expr.check().map_err(CallError::InvalidExpr)?;
        let mut initial = BTreeMap::<InputId, Trace>::new();
        for &id in &self.snapshot_inputs {
            let object = binding(id)?;
            initial.insert(
                id,
                match object {
                    ObjectTracer::Ref(value) => value.clone(),
                    ObjectTracer::Cell(cell) => cell.get(),
                },
            );
        }
        let bind_value = |id: &InputId| {
            if let Some(value) = initial.get(id) {
                return Ok(value.clone());
            }
            match binding(*id)? {
                ObjectTracer::Ref(value) => Ok(value.clone()),
                ObjectTracer::Cell(cell) => Ok(cell.read_home()), // unused parameter, not an observable read
            }
        };
        let captures = self
            .closure
            .iter()
            .map(bind_value)
            .collect::<Result<Vec<_>, CallError>>()?;
        let arguments = self
            .inputs
            .iter()
            .map(bind_value)
            .collect::<Result<Vec<_>, CallError>>()?;
        let returned = crate::builtins::call(
            self.trace_body.clone(),
            captures,
            arguments,
            self.output_infos.clone(),
        );
        let returned_value = |index: usize| match &returned[index] {
            ObjectTracer::Ref(value) => value,
            ObjectTracer::Cell(_) => unreachable!("immutable calls return values"),
        };
        let mut locals = BTreeMap::new();
        let results = self
            .results()
            .iter()
            .map(|output| {
                Ok(match *output {
                    Output::OutputValue { output_idx } => {
                        ObjectTracer::Ref(returned_value(output_idx).clone())
                    }
                    Output::InputValue { input_id } => {
                        ObjectTracer::Ref(initial[&input_id].clone())
                    }
                    Output::InputCell { input_id } => binding(input_id)?.clone(),
                    Output::NewInputCell { id, input_id } => ObjectTracer::Cell(
                        locals
                            .entry(id)
                            .or_insert_with(|| CellTrace::new(initial[&input_id].clone()))
                            .clone(),
                    ),
                    Output::NewOutputCell { id, output_idx } => ObjectTracer::Cell(
                        locals
                            .entry(id)
                            .or_insert_with(|| CellTrace::new(returned_value(output_idx).clone()))
                            .clone(),
                    ),
                })
            })
            .collect::<Result<Vec<_>, CallError>>()?;
        for &(input_id, descriptor) in &self.updates {
            let ObjectTracer::Cell(cell) = binding(input_id)? else {
                return Err(CallError::ArgumentType(input_id));
            };
            let value = match self.outputs[descriptor] {
                Output::InputValue { input_id } => &initial[&input_id],
                Output::OutputValue { output_idx } => returned_value(output_idx),
                _ => unreachable!("cell update sources are immutable values"),
            };
            cell.set(value.clone());
        }
        Ok(results)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CallError {
    Arity,
    ArgumentType(InputId),
    MissingInput(InputId),
    InvalidExpr(LoweringError),
}
impl fmt::Display for CallError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for CallError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LoweringError {
    InputSpecArity {
        closure: bool,
        expected: usize,
        actual: usize,
    },
    KindMismatch {
        closure: bool,
        index: usize,
    },
    AliasedInputKindMismatch(InputId),
    AliasedInputOwnershipMismatch(InputId),
    UndefinedVariable(Var),
    ExpectedValue(Var),
    ExpectedCell(Var),
    MoveBorrowed(Var),
    BorrowedOutput(Var),
    InvalidComputation,
    DuplicateDefinition(Var),
    CaptureArity {
        expected: usize,
        actual: usize,
    },
    CaptureOwnership(Var),
    CallArity,
    UnsupportedNestedOperation,
}
impl fmt::Display for LoweringError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for LoweringError {}

/// Lower a captured HExpr. There is intentionally no trace-DAG-to-Expr shortcut.
pub fn lower(expr: &HExpr, spec: &InputSpec) -> Result<LoweredExpr, LoweringError> {
    Lowerer::new(expr, spec)?.lower(expr)
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum NestedBinding {
    Value(Var),
    Cell(CellBinding),
}

struct Lowerer {
    spec: InputSpec,
    scope: VarScope,
    values: BTreeMap<Var, Var>,
    cells: BTreeMap<Var, CellBinding>,
    states: BTreeMap<CellBinding, Var>,
    bound_inputs: BTreeMap<InputId, (Kind, Ownership, Var)>,
    written: BTreeSet<CellBinding>,
    closure: Vec<(Var, Ownership, ValueInfo)>,
    closure_sources: Vec<InputId>,
    inputs: Vec<(Var, Ownership, ValueInfo)>,
    input_sources: Vec<InputId>,
    eqns: Vec<Eqn>,
    trace_eqns: Vec<HInstr>,
    local_count: usize,
}
impl Lowerer {
    fn new(expr: &HExpr, spec: &InputSpec) -> Result<Self, LoweringError> {
        let mut lowerer = Self {
            spec: spec.clone(),
            scope: VarScope::new(),
            values: BTreeMap::new(),
            cells: BTreeMap::new(),
            states: BTreeMap::new(),
            bound_inputs: BTreeMap::new(),
            written: BTreeSet::new(),
            closure: vec![],
            closure_sources: vec![],
            inputs: vec![],
            input_sources: vec![],
            eqns: vec![],
            trace_eqns: vec![],
            local_count: 0,
        };
        for (is_closure, params, specs) in [
            (true, expr.closure(), spec.closure.as_slice()),
            (false, expr.inputs(), spec.inputs.as_slice()),
        ] {
            if params.len() != specs.len() {
                return Err(LoweringError::InputSpecArity {
                    closure: is_closure,
                    expected: params.len(),
                    actual: specs.len(),
                });
            }
            for (index, ((param, param_kind), (id, kind, ownership))) in
                params.iter().zip(specs).enumerate()
            {
                if param_kind != kind {
                    return Err(LoweringError::KindMismatch {
                        closure: is_closure,
                        index,
                    });
                }
                let var = if let Some((previous_kind, previous_ownership, var)) =
                    lowerer.bound_inputs.get(id)
                {
                    if previous_kind != kind {
                        return Err(LoweringError::AliasedInputKindMismatch(*id));
                    }
                    if previous_ownership != ownership {
                        return Err(LoweringError::AliasedInputOwnershipMismatch(*id));
                    }
                    *var
                } else {
                    let var = lowerer.scope.create_var();
                    lowerer
                        .bound_inputs
                        .insert(*id, (kind.clone(), *ownership, var));
                    var
                };
                match kind {
                    Kind::Value(_) => {
                        lowerer.values.insert(*param, var);
                    }
                    Kind::Cell(_) => {
                        let resource = CellBinding::Input(*id);
                        lowerer.cells.insert(*param, resource);
                        lowerer.states.entry(resource).or_insert(var);
                    }
                }
            }
        }
        // Preserve declaration order within each list, deduplicating identities.
        // An identity present in explicit inputs is never also a closure parameter.
        let input_ids: BTreeSet<_> = spec.inputs.iter().map(|(id, _, _)| *id).collect();
        let mut seen = BTreeSet::new();
        for (id, kind, ownership) in &spec.closure {
            if !input_ids.contains(id) && seen.insert(*id) {
                lowerer.closure.push((
                    lowerer.bound_inputs[id].2,
                    *ownership,
                    value_info(kind).clone(),
                ));
                lowerer.closure_sources.push(*id);
            }
        }
        for (id, kind, ownership) in &spec.inputs {
            if seen.insert(*id) {
                lowerer.inputs.push((
                    lowerer.bound_inputs[id].2,
                    *ownership,
                    value_info(kind).clone(),
                ));
                lowerer.input_sources.push(*id);
            }
        }
        Ok(lowerer)
    }
    fn value(&self, var: Var) -> Result<Var, LoweringError> {
        self.values.get(&var).copied().ok_or_else(|| {
            if self.cells.contains_key(&var) {
                LoweringError::ExpectedValue(var)
            } else {
                LoweringError::UndefinedVariable(var)
            }
        })
    }
    fn cell(&self, var: Var) -> Result<CellBinding, LoweringError> {
        self.cells.get(&var).copied().ok_or_else(|| {
            if self.values.contains_key(&var) {
                LoweringError::ExpectedCell(var)
            } else {
                LoweringError::UndefinedVariable(var)
            }
        })
    }
    // Emit both representations from the specialized SSA operands. Only the
    // executable equation has ownership; no Expr is embedded in the trace.
    fn emit_eqn(
        &mut self,
        op: Op,
        inputs: Vec<(Var, Ownership)>,
        outputs: Vec<(Var, Kind)>,
        nested: Option<&LoweredExpr>,
        captures: Vec<(Var, Ownership)>,
    ) {
        let high = nested.map(|body| {
            HSubExpr::new(
                body.trace_body.clone(),
                captures.iter().map(|(var, _)| *var).collect(),
            )
        });
        self.trace_eqns.push(HInstr::Apply(HEqn::new(
            op.clone(),
            high.into_iter().collect(),
            inputs.iter().map(|(var, _)| *var).collect(),
            outputs.clone(),
        )));
        let low = nested.map(|body| SubExpr::new(body.expr.clone(), captures));
        self.eqns.push(Eqn::new_with_subexprs(
            op,
            low.into_iter().collect(),
            inputs,
            outputs,
        ));
    }

    fn lower_equation(&mut self, eqn: &HEqn) -> Result<(), LoweringError> {
        if eqn.subexprs().is_empty() {
            let inputs = eqn
                .inputs()
                .iter()
                .map(|var| self.value(*var).map(|var| (var, Ownership::Borrow)))
                .collect::<Result<_, _>>()?;
            let mut outputs = Vec::new();
            for (output, kind) in eqn.outputs() {
                if !matches!(kind, Kind::Value(_)) {
                    return Err(LoweringError::ExpectedValue(*output));
                }
                let var = self.scope.create_var();
                self.values.insert(*output, var);
                outputs.push((var, kind.clone()));
            }
            self.emit_eqn(eqn.op().clone(), inputs, outputs, None, vec![]);
            return Ok(());
        }
        // Functionalization defines a call ABI for mutating regions. Other
        // operations need their own rule for threading cell state.
        if eqn.op() != &crate::builtins::call_op() {
            return Err(LoweringError::UnsupportedNestedOperation);
        }
        let [sub] = eqn.subexprs() else {
            return Err(LoweringError::CallArity);
        };
        let body = sub.expr();
        if sub.closure().len() != body.closure().len()
            || eqn.inputs().len() != body.inputs().len()
            || eqn.outputs().len() != body.outputs().len()
        {
            return Err(LoweringError::CallArity);
        }
        let mut identities = BTreeMap::new();
        let mut bindings = BTreeMap::new();
        let mut bind = |operands: &[Var], params: &[(Var, Kind)], id: fn(usize) -> InputId| {
            operands
                .iter()
                .zip(params)
                .enumerate()
                .map(|(index, (&var, (_, kind)))| {
                    let binding = match kind {
                        Kind::Value(_) => NestedBinding::Value(self.value(var)?),
                        Kind::Cell(_) => NestedBinding::Cell(self.cell(var)?),
                    };
                    let id = *identities.entry(binding).or_insert_with(|| id(index));
                    bindings.insert(id, binding);
                    Ok((id, kind.clone(), Ownership::Borrow))
                })
                .collect::<Result<Vec<_>, LoweringError>>()
        };
        let inputs = bind(eqn.inputs(), body.inputs(), InputId::Input)?;
        let closure = bind(sub.closure(), body.closure(), InputId::Closure)?;
        let lowered = body.lower(&InputSpec { closure, inputs })?;

        // Snapshot all operands from current SSA state before processing updates.
        let initial: BTreeMap<_, _> = bindings
            .iter()
            .map(|(&id, &binding)| {
                let var = match binding {
                    NestedBinding::Value(var) => var,
                    NestedBinding::Cell(cell) => self.states[&cell],
                };
                (id, var)
            })
            .collect();
        let bind_values = |ids: &[InputId]| {
            ids.iter()
                .map(|id| (initial[id], Ownership::Borrow))
                .collect()
        };
        let outputs: Vec<_> = lowered
            .output_infos
            .iter()
            .map(|info| (self.scope.create_var(), Kind::Value(info.clone())))
            .collect();
        let value_source = |output: &Output| match *output {
            Output::OutputValue { output_idx } | Output::NewOutputCell { output_idx, .. } => {
                outputs[output_idx].0
            }
            Output::InputValue { input_id } | Output::NewInputCell { input_id, .. } => {
                initial[&input_id]
            }
            Output::InputCell { .. } => unreachable!("expected a value source"),
        };
        let mut new_cells = BTreeMap::new();
        for ((var, kind), result) in eqn.outputs().iter().zip(lowered.results()) {
            match (kind, result) {
                (Kind::Value(_), Output::OutputValue { .. } | Output::InputValue { .. }) => {
                    self.values.insert(*var, value_source(result));
                }
                (Kind::Cell(_), Output::InputCell { input_id }) => {
                    let NestedBinding::Cell(cell) = bindings[input_id] else {
                        unreachable!()
                    };
                    self.cells.insert(*var, cell);
                }
                (
                    Kind::Cell(_),
                    Output::NewInputCell { id, .. } | Output::NewOutputCell { id, .. },
                ) => {
                    let cell = *new_cells.entry(*id).or_insert_with(|| {
                        let cell = CellBinding::Local(self.local_count);
                        self.local_count += 1;
                        self.states.insert(cell, value_source(result));
                        cell
                    });
                    self.cells.insert(*var, cell);
                }
                _ => return Err(LoweringError::InvalidComputation),
            }
        }
        for (id, descriptor) in &lowered.updates {
            let NestedBinding::Cell(cell) = bindings[id] else {
                unreachable!()
            };
            self.states
                .insert(cell, value_source(&lowered.outputs[*descriptor]));
            self.written.insert(cell);
        }
        if !lowered.expr.eqns().is_empty() || !outputs.is_empty() {
            self.emit_eqn(
                eqn.op().clone(),
                bind_values(&lowered.inputs),
                outputs,
                Some(&lowered),
                bind_values(&lowered.closure),
            );
        }
        Ok(())
    }

    fn lower(mut self, expr: &HExpr) -> Result<LoweredExpr, LoweringError> {
        for instr in expr.instructions() {
            match instr {
                HInstr::Apply(eqn) => self.lower_equation(eqn)?,
                HInstr::CreateCell { cell, initial, .. } => {
                    let initial = self.value(*initial)?;
                    let resource = CellBinding::Local(self.local_count);
                    self.local_count += 1;
                    self.cells.insert(*cell, resource);
                    self.states.insert(resource, initial);
                }
                HInstr::ReadCell { cell, output, .. } => {
                    let resource = self.cell(*cell)?;
                    self.values.insert(*output, self.states[&resource]);
                }
                HInstr::WriteCell { cell, value } => {
                    let resource = self.cell(*cell)?;
                    let value = self.value(*value)?;
                    self.states.insert(resource, value);
                    self.written.insert(resource);
                }
            }
        }
        let borrowed_inputs: BTreeMap<Var, InputId> = self
            .closure
            .iter()
            .chain(&self.inputs)
            .zip(self.closure_sources.iter().chain(&self.input_sources))
            .filter(|((_, ownership, _), _)| *ownership == Ownership::Borrow)
            .map(|((var, _, _), id)| (*var, *id))
            .collect();
        let mut return_vars = Vec::new();
        let mut indices = BTreeMap::new();
        let mut source = |var: Var| {
            if let Some(&input_id) = borrowed_inputs.get(&var) {
                return Output::InputValue { input_id };
            }
            let output_idx = *indices.entry(var).or_insert_with(|| {
                let index = return_vars.len();
                return_vars.push(var);
                index
            });
            Output::OutputValue { output_idx }
        };
        let mut outputs = Vec::new();
        for &var in expr.outputs() {
            let output = if self.cells.contains_key(&var) {
                match self.cell(var)? {
                    CellBinding::Input(input_id) => Output::InputCell { input_id },
                    cell @ CellBinding::Local(id) => match source(self.states[&cell]) {
                        Output::InputValue { input_id } => Output::NewInputCell { id, input_id },
                        Output::OutputValue { output_idx } => {
                            Output::NewOutputCell { id, output_idx }
                        }
                        _ => unreachable!("value sources never describe cell handles"),
                    },
                }
            } else {
                source(self.value(var)?)
            };
            outputs.push(output);
        }
        // One trailing descriptor per external update keeps the user result
        // prefix recoverable from outputs.len() - updates.len(), without another
        // count/table. Sources can be either borrowed inputs or owned returns.
        let mut updates = Vec::new();
        for &cell in &self.written {
            if let CellBinding::Input(input_id) = cell {
                let descriptor_idx = outputs.len();
                outputs.push(source(self.states[&cell]));
                updates.push((input_id, descriptor_idx));
            }
        }
        // Every physical return is either a parameter's immutable value (a
        // cell's initial contents) or an equation result.
        let mut output_infos = vec![None; return_vars.len()];
        let definitions = self
            .bound_inputs
            .values()
            .map(|(kind, _, var)| (var, kind))
            .chain(
                self.eqns
                    .iter()
                    .flat_map(|eqn| eqn.outputs().iter().map(|(var, kind)| (var, kind))),
            );
        for (var, kind) in definitions {
            if let Some(&index) = indices.get(var) {
                output_infos[index] = Some(value_info(kind).clone());
            }
        }
        let output_infos = output_infos
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or(LoweringError::InvalidComputation)?;
        let trace_params = |params: &[(Var, Ownership, ValueInfo)]| {
            params
                .iter()
                .map(|(var, _, info)| (*var, Kind::Value(info.clone())))
                .collect()
        };
        let trace_body = Arc::new(HExpr {
            closure: trace_params(&self.closure),
            inputs: trace_params(&self.inputs),
            instructions: self.trace_eqns,
            outputs: return_vars.clone(),
        });
        let mut expr = Expr {
            closure: self.closure,
            inputs: self.inputs,
            eqns: self.eqns,
            outputs: return_vars,
            name_hint: None,
        };
        assign_ownership(&mut expr);
        expr.check()?;
        let mut needed = BTreeSet::new();
        for (id, (var, _, _)) in self
            .closure_sources
            .iter()
            .chain(&self.input_sources)
            .zip(expr.closure().iter().chain(expr.inputs()))
        {
            let used = expr.outputs().contains(var)
                || expr.eqns().iter().any(|eqn| {
                    eqn.inputs()
                        .iter()
                        .chain(eqn.subexprs().iter().flat_map(|sub| sub.closure()))
                        .any(|(input, _)| input == var)
                });
            if used {
                needed.insert(*id);
            }
        }
        for output in &outputs {
            if let Output::InputValue { input_id } | Output::NewInputCell { input_id, .. } = output
            {
                needed.insert(*input_id);
            }
        }
        Ok(LoweredExpr {
            snapshot_inputs: needed,
            spec: self.spec,
            expr: Arc::new(expr),
            trace_body,
            output_infos,
            closure: self.closure_sources,
            inputs: self.input_sources,
            outputs,
            updates,
        })
    }
}

fn value_info(kind: &Kind) -> &ValueInfo {
    let (Kind::Value(info) | Kind::Cell(info)) = kind;
    info
}

fn assign_ownership(expr: &mut Expr) {
    let owned: BTreeSet<Var> = expr
        .closure
        .iter()
        .chain(&expr.inputs)
        .filter(|(_, ownership, _)| *ownership == Ownership::Move)
        .map(|(var, _, _)| *var)
        .chain(
            expr.eqns
                .iter()
                .flat_map(|eqn| eqn.outputs.iter().map(|(var, _)| *var)),
        )
        .collect();
    let mut remaining = BTreeMap::<Var, usize>::new();
    for &(var, _) in expr.eqns.iter().flat_map(|eqn| {
        eqn.inputs
            .iter()
            .chain(eqn.subexprs.iter().flat_map(|s| &s.captures))
    }) {
        *remaining.entry(var).or_default() += 1;
    }
    // Returns implicitly move, so keep their values alive through all equations.
    for &output in &expr.outputs {
        *remaining.entry(output).or_default() += 1;
    }
    for (var, ownership) in expr.eqns.iter_mut().flat_map(|eqn| {
        eqn.inputs
            .iter_mut()
            .chain(eqn.subexprs.iter_mut().flat_map(|s| &mut s.captures))
    }) {
        let count = remaining.get_mut(var).unwrap();
        *count -= 1;
        *ownership = if *count == 0 && owned.contains(var) {
            Ownership::Move
        } else {
            Ownership::Borrow
        };
    }
}

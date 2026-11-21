//! HExpr-to-Expr lowering, input specifications, and execution-boundary metadata.

mod structured;
pub use structured::{FixedRegion, RegionArgument, RegionCell, StructuredRegion};

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;

use crate::{
    Eqn, Expr, HEqn, HExpr, HInstr, HSubExpr, Kind, Op, SubExpr, ValueInfo, Var, VarScope,
};

/// Canonical source of a supplied value or cell, indexed in the original HExpr
/// argument lists. Aliases share a source. Explicit inputs take precedence over
/// closure positions; kind must agree across all aliases.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum InputId {
    Closure(usize),
    Input(usize),
}

/// One (identity, kind) entry per HExpr parameter, in declaration order.
#[derive(Clone, Debug, Default)]
pub struct InputSpec {
    pub closure: Vec<(InputId, Kind, ValueInfo)>,
    pub inputs: Vec<(InputId, Kind, ValueInfo)>,
}
impl InputSpec {
    /// Derive shared identities and kinds from actual arguments. Equal values
    /// are not merged unless their trace identities match.
    pub fn from_args(closure: &[crate::ObjectTracer], inputs: &[crate::ObjectTracer]) -> Self {
        #[derive(PartialEq, Eq, PartialOrd, Ord)]
        enum Identity {
            Value(crate::TracerKey),
            Cell(crate::CellKey),
        }
        let mut identities = BTreeMap::new();
        let mut bind = |(index, object): (InputId, &crate::ObjectTracer)| {
            let (identity, kind, info) = match object {
                crate::ObjectTracer::Ref(value) => (
                    Identity::Value(crate::TracerKey::from(value)),
                    Kind::Value,
                    value.info().clone(),
                ),
                crate::ObjectTracer::Cell(cell) => (
                    Identity::Cell(crate::CellKey::from(cell)),
                    Kind::Cell,
                    cell.type_info().clone(),
                ),
            };
            (*identities.entry(identity).or_insert(index), kind, info)
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

/// How to assemble high-level results from the immutable Expr results and
/// original cell handles. Equal new-cell IDs materialize the same fresh handle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Output {
    Value { result_idx: usize },
    InputCell { input_id: InputId },
    NewCell { id: usize, result_idx: usize },
}

/// Pure program and its execution-boundary mappings. Closure and inputs correspond
/// one to one with expr.closure() and expr.inputs(), respectively. Closure-only
/// identities stay in closure; any identity appearing in explicit inputs belongs
/// to inputs, even when also captured. InputSpec supplies kinds and aliases.
///
/// `outputs` assembles the user results. `updates` maps each written external
/// cell to a final value in `expr.outputs()`. Parameter forwarding is expressed
/// directly by returning its variable.
#[derive(Clone, Debug)]
pub struct LoweredExpr {
    spec: InputSpec,
    snapshot_inputs: BTreeSet<InputId>,
    pub expr: Arc<Expr>,
    trace_body: Arc<HExpr>,
    /// Metadata for distinct computed results, in replay-body output order.
    pub output_infos: Vec<ValueInfo>,
    pub closure: Vec<InputId>,
    pub inputs: Vec<InputId>,
    pub outputs: Vec<Output>,
    pub updates: Vec<(InputId, usize)>,
}
impl LoweredExpr {
    /// Replay body returning only distinct computed results. The wrapper
    /// forwards any parameters directly from their snapshots.
    pub fn trace_body(&self) -> &Arc<HExpr> {
        &self.trace_body
    }
    pub fn results(&self) -> &[Output] {
        &self.outputs
    }

    pub(crate) fn input_id(&self, var: Var) -> Option<InputId> {
        self.expr
            .closure()
            .iter()
            .zip(&self.closure)
            .chain(self.expr.inputs().iter().zip(&self.inputs))
            .find_map(|((parameter, _), &id)| (*parameter == var).then_some(id))
    }

    pub(crate) fn result_source(&self, result_idx: usize) -> Option<ResultSource> {
        let &var = self.expr.outputs().get(result_idx)?;
        if let Some(id) = self.input_id(var) {
            Some(ResultSource::Parameter(id))
        } else {
            self.trace_body
                .outputs()
                .iter()
                .position(|&result| result == var)
                .map(ResultSource::Computed)
        }
    }

    pub(crate) fn computation_expr(&self) -> Arc<Expr> {
        if self.expr.outputs() == self.trace_body.outputs() {
            self.expr.clone()
        } else {
            let mut expr = self.expr.as_ref().clone();
            expr.outputs = self.trace_body.outputs().to_vec();
            Arc::new(expr)
        }
    }

    /// Emit an immutable application using `emit` and materialize its boundary. All required cell
    /// snapshots are taken before any updates; repeated local-cell IDs share one
    /// fresh handle. Returning an input cell alone does not read its contents.
    ///
    /// The caller must supply the alias specialization used during lowering.
    /// Sources are resolved directly through stored InputId offsets; this method
    /// checks arity and metadata but does not rediscover or validate identities.
    pub fn replay(
        &self,
        closure: &[crate::ObjectTracer],
        inputs: &[crate::ObjectTracer],
        emit: impl FnOnce(
            Arc<HExpr>,
            Vec<crate::Trace>,
            Vec<crate::Trace>,
            Vec<ValueInfo>,
        ) -> Vec<crate::Trace>,
    ) -> Result<Vec<crate::ObjectTracer>, CallError> {
        let symbols = self
            .expr
            .symbol_closure
            .iter()
            .map(|(s, _)| *s)
            .collect::<Vec<_>>();
        self.replay_with_symbols(&symbols, closure, inputs, emit)
    }
    /// Replay with explicit symbols from the enclosing trace.
    pub fn replay_with_symbols(
        &self,
        symbol_captures: &[crate::Symbol],
        closure: &[crate::ObjectTracer],
        inputs: &[crate::ObjectTracer],
        emit: impl FnOnce(
            Arc<HExpr>,
            Vec<crate::Trace>,
            Vec<crate::Trace>,
            Vec<ValueInfo>,
        ) -> Vec<crate::Trace>,
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
        let actuals = closure.iter().chain(inputs).map(|v| match v {
            ObjectTracer::Ref(v) => v.info(),
            ObjectTracer::Cell(c) => c.type_info(),
        });
        let symbols = crate::symbol::substitution(
            &self.expr.symbol_closure,
            symbol_captures,
            &self.expr.symbol_inputs,
            spec.closure
                .iter()
                .chain(&spec.inputs)
                .map(|v| &v.2)
                .zip(actuals),
            |s| crate::TraceContext::current()?.symbol_info(s),
        )
        .map_err(CallError::InvalidExpr)?;
        for ((id, kind, info), object) in spec
            .closure
            .iter()
            .chain(&spec.inputs)
            .zip(closure.iter().chain(inputs))
        {
            let actual = info.remap_symbols(|s| symbols[&s]);
            let valid = match (kind, object) {
                (Kind::Value, ObjectTracer::Ref(value)) => &actual == value.info(),
                (Kind::Cell, ObjectTracer::Cell(cell)) => &actual == cell.type_info(),
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
        let body = if symbols.iter().all(|(s, a)| s == a) {
            self.trace_body.clone()
        } else {
            Arc::new(self.trace_body.substitute_symbols(&symbols))
        };
        let returned = emit(
            body,
            captures,
            arguments,
            self.output_infos
                .iter()
                .map(|i| i.remap_symbols(|s| symbols[&s]))
                .collect(),
        );
        if returned.len() != self.output_infos.len() {
            return Err(CallError::Arity);
        }
        let result_value = |result_idx: usize| match self
            .result_source(result_idx)
            .expect("lowered result source")
        {
            ResultSource::Computed(index) => &returned[index],
            ResultSource::Parameter(id) => &initial[&id],
        };
        let mut locals = BTreeMap::new();
        let results = self
            .results()
            .iter()
            .map(|output| {
                Ok(match *output {
                    Output::Value { result_idx } => {
                        ObjectTracer::Ref(result_value(result_idx).clone())
                    }
                    Output::InputCell { input_id } => binding(input_id)?.clone(),
                    Output::NewCell { id, result_idx } => ObjectTracer::Cell(
                        locals
                            .entry(id)
                            .or_insert_with(|| CellTrace::new(result_value(result_idx).clone()))
                            .clone(),
                    ),
                })
            })
            .collect::<Result<Vec<_>, CallError>>()?;
        for &(input_id, result_idx) in &self.updates {
            let ObjectTracer::Cell(cell) = binding(input_id)? else {
                return Err(CallError::ArgumentType(input_id));
            };
            cell.set(result_value(result_idx).clone());
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
    UndefinedVariable(Var),
    ExpectedValue(Var),
    ExpectedCell(Var),
    InvalidResultIndex(usize),
    InvalidComputation,
    DuplicateDefinition(Var),
    CaptureArity {
        expected: usize,
        actual: usize,
    },
    CallArity,
    UnsupportedNestedOperation,
    RegionSignature(&'static str),
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

/// Source for the optimized execution boundary, derived from the Expr's SSA
/// result variables. Not part of the IR or its boundary result descriptors.
pub(crate) enum ResultSource {
    Computed(usize),
    Parameter(InputId),
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum NestedBinding {
    Value(Var),
    Cell(CellBinding),
}

/// A recursively lowered region with bindings resolved in its parent scope.
pub struct LoweredRegion {
    output_infos: Vec<ValueInfo>,
    symbol_captures: Vec<crate::Symbol>,
    program: LoweredExpr,
    bindings: BTreeMap<InputId, NestedBinding>,
    initial: BTreeMap<InputId, Var>,
}

impl LoweredRegion {
    /// Computed result metadata in the caller’s symbol namespace.
    pub fn output_infos(&self) -> &[ValueInfo] {
        &self.output_infos
    }
    pub fn program(&self) -> &LoweredExpr {
        &self.program
    }

    pub fn inputs(&self) -> Vec<Var> {
        self.program
            .inputs
            .iter()
            .map(|id| self.initial[id])
            .collect()
    }

    pub fn subexpr(&self) -> RegionUse {
        RegionUse {
            symbol_captures: self.symbol_captures.clone(),
            expr: self.program.computation_expr(),
            trace_body: self.program.trace_body.clone(),
            captures: self
                .program
                .closure
                .iter()
                .map(|id| self.initial[id])
                .collect(),
        }
    }

    fn source(&self, result_idx: usize, physical: &[(Var, ValueInfo)]) -> Var {
        match self
            .program
            .result_source(result_idx)
            .expect("lowered region result")
        {
            ResultSource::Computed(index) => physical[index].0,
            ResultSource::Parameter(id) => self.initial[&id],
        }
    }
}

/// A lowered child and the parent SSA values bound to its captures.
pub struct RegionUse {
    symbol_captures: Vec<crate::Symbol>,
    expr: Arc<Expr>,
    trace_body: Arc<HExpr>,
    captures: Vec<Var>,
}

/// Shared SSA construction and cell bookkeeping for operation lowering rules.
pub struct LoweringContext<'a> {
    lowerer: &'a mut Lowerer,
}

struct Lowerer {
    symbol_sorts: BTreeMap<crate::Symbol, crate::SymbolInfo>,
    spec: InputSpec,
    scope: VarScope,
    values: BTreeMap<Var, Var>,
    cells: BTreeMap<Var, CellBinding>,
    states: BTreeMap<CellBinding, Var>,
    bound_inputs: BTreeMap<InputId, (Kind, Var, ValueInfo)>,
    written: BTreeSet<CellBinding>,
    closure: Vec<(Var, ValueInfo)>,
    closure_sources: Vec<InputId>,
    inputs: Vec<(Var, ValueInfo)>,
    input_sources: Vec<InputId>,
    eqns: Vec<Eqn>,
    trace_eqns: Vec<HInstr>,
    local_count: usize,
}
impl Lowerer {
    fn new(expr: &HExpr, spec: &InputSpec) -> Result<Self, LoweringError> {
        expr.check_symbols()?;
        let mut lowerer = Self {
            symbol_sorts: expr
                .symbol_closure
                .iter()
                .chain(&expr.symbol_inputs)
                .cloned()
                .collect(),
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
            for (index, ((param, param_kind, param_info), (id, kind, info))) in
                params.iter().zip(specs).enumerate()
            {
                if param_kind != kind || param_info != info {
                    return Err(LoweringError::KindMismatch {
                        closure: is_closure,
                        index,
                    });
                }
                let var = if let Some((previous_kind, var, previous_info)) =
                    lowerer.bound_inputs.get(id)
                {
                    if previous_kind != kind || previous_info != info {
                        return Err(LoweringError::AliasedInputKindMismatch(*id));
                    }
                    *var
                } else {
                    let var = lowerer.scope.create_var();
                    lowerer.bound_inputs.insert(*id, (*kind, var, info.clone()));
                    var
                };
                match kind {
                    Kind::Value => {
                        lowerer.values.insert(*param, var);
                    }
                    Kind::Cell => {
                        let resource = CellBinding::Input(*id);
                        lowerer.cells.insert(*param, resource);
                        lowerer.states.entry(resource).or_insert(var);
                    }
                }
            }
        }
        // Preserve declaration order within each list, deduplicating identities.
        // An identity present in explicit inputs is never also a closure parameter.
        let input_ids: BTreeSet<_> = spec.inputs.iter().map(|(id, ..)| *id).collect();
        let mut seen = BTreeSet::new();
        for (id, _, info) in &spec.closure {
            if !input_ids.contains(id) && seen.insert(*id) {
                lowerer
                    .closure
                    .push((lowerer.bound_inputs[id].1, info.clone()));
                lowerer.closure_sources.push(*id);
            }
        }
        for (id, _, info) in &spec.inputs {
            if seen.insert(*id) {
                lowerer
                    .inputs
                    .push((lowerer.bound_inputs[id].1, info.clone()));
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
    fn lower_equation(&mut self, eqn: &HEqn) -> Result<(), LoweringError> {
        eqn.op().lower(eqn, &mut LoweringContext { lowerer: self })
    }
}

impl LoweringContext<'_> {
    pub fn lower_value_apply(&mut self, eqn: &HEqn) -> Result<(), LoweringError> {
        if !eqn.subexprs().is_empty() {
            return Err(LoweringError::UnsupportedNestedOperation);
        }
        self.lower_value_apply_with_regions(eqn, vec![])
    }

    /// Lower an operation whose regions accept independent value arguments and
    /// have no captures or mutable state, such as a reduction combiner.
    pub fn lower_closed_regions_apply(&mut self, eqn: &HEqn) -> Result<(), LoweringError> {
        let regions = eqn
            .subexprs()
            .iter()
            .map(|region| {
                let body = region.expr();
                if !region.closure().is_empty()
                    || !body.closure().is_empty()
                    || body
                        .inputs()
                        .iter()
                        .any(|(_, kind, _)| !matches!(kind, Kind::Value))
                    || body
                        .outputs()
                        .iter()
                        .any(|var| !matches!(body.kind(var), Some(Kind::Value)))
                {
                    return Err(LoweringError::RegionSignature(
                        "expected a closed value region",
                    ));
                }
                let spec = InputSpec {
                    closure: vec![],
                    inputs: body
                        .inputs()
                        .iter()
                        .enumerate()
                        .map(|(i, (_, kind, info))| (InputId::Input(i), *kind, info.clone()))
                        .collect(),
                };
                let program = lower(body, &spec)?;
                Ok(RegionUse {
                    symbol_captures: region.symbol_captures.clone(),
                    expr: program.expr,
                    trace_body: body.clone(),
                    captures: vec![],
                })
            })
            .collect::<Result<_, _>>()?;
        self.lower_value_apply_with_regions(eqn, regions)
    }

    fn lower_value_apply_with_regions(
        &mut self,
        eqn: &HEqn,
        regions: Vec<RegionUse>,
    ) -> Result<(), LoweringError> {
        let inputs = eqn
            .inputs()
            .iter()
            .map(|&var| self.lowerer.value(var))
            .collect::<Result<_, _>>()?;
        let mut outputs = Vec::new();
        for (output, kind, info) in eqn.outputs() {
            let Kind::Value = kind else {
                return Err(LoweringError::ExpectedValue(*output));
            };
            let var = self.lowerer.scope.create_var();
            self.lowerer.values.insert(*output, var);
            outputs.push((var, info.clone()));
        }
        self.emit(eqn.op().clone(), inputs, outputs, regions);
        Ok(())
    }

    fn object_info(&self, var: Var, kind: Kind) -> Result<ValueInfo, LoweringError> {
        match kind {
            Kind::Value => Ok(self.value_operand(var)?.1),
            Kind::Cell => Ok(self.cell_state(RegionCell(self.lowerer.cell(var)?))?.1),
        }
    }
    fn child_bindings(
        &self,
        sub: &HSubExpr,
        actual_inputs: &[ValueInfo],
    ) -> Result<BTreeMap<crate::Symbol, crate::Symbol>, LoweringError> {
        let body = sub.expr().as_ref();
        if body.symbol_inputs.is_empty() && body.symbol_closure.is_empty() {
            return Ok(BTreeMap::new());
        }
        let captures = sub
            .closure()
            .iter()
            .zip(&body.closure)
            .map(|(v, (_, k, _))| self.object_info(*v, *k))
            .collect::<Result<Vec<_>, _>>()?;
        body.bind_symbols(&sub.symbol_captures, &captures, actual_inputs, |s| {
            self.lowerer.symbol_sorts.get(&s).cloned()
        })
    }
    /// Lower a region whose explicit arguments are bound to parent variables.
    /// The operation picks the argument layout; aliases are shared across
    /// those arguments and the region's captures.
    pub fn lower_region(
        &self,
        sub: &HSubExpr,
        operands: &[Var],
    ) -> Result<LoweredRegion, LoweringError> {
        let body = sub.expr();
        if sub.closure().len() != body.closure().len() {
            return Err(LoweringError::CaptureArity {
                expected: body.closure().len(),
                actual: sub.closure().len(),
            });
        }
        if operands.len() != body.inputs().len() {
            return Err(LoweringError::InputSpecArity {
                closure: false,
                expected: body.inputs().len(),
                actual: operands.len(),
            });
        }
        let actual = operands
            .iter()
            .zip(&body.inputs)
            .map(|(v, (_, k, _))| self.object_info(*v, *k))
            .collect::<Result<Vec<_>, _>>()?;
        let symbols = self.child_bindings(sub, &actual)?;
        let mut identities = BTreeMap::new();
        let mut bindings = BTreeMap::new();
        let mut bind =
            |operands: &[Var], params: &[(Var, Kind, ValueInfo)], id: fn(usize) -> InputId| {
                operands
                    .iter()
                    .zip(params)
                    .enumerate()
                    .map(|(index, (&var, (_, kind, info)))| {
                        let binding = match kind {
                            Kind::Value => NestedBinding::Value(self.lowerer.value(var)?),
                            Kind::Cell => NestedBinding::Cell(self.lowerer.cell(var)?),
                        };
                        let id = *identities.entry(binding).or_insert_with(|| id(index));
                        bindings.insert(id, binding);
                        Ok((id, *kind, info.clone()))
                    })
                    .collect::<Result<Vec<_>, LoweringError>>()
            };
        let inputs = bind(operands, body.inputs(), InputId::Input)?;
        let closure = bind(sub.closure(), body.closure(), InputId::Closure)?;
        let lowered = body.lower(&InputSpec { closure, inputs })?;

        // Snapshot all operands from current SSA state before processing updates.
        let initial: BTreeMap<_, _> = bindings
            .iter()
            .map(|(&id, &binding)| {
                let var = match binding {
                    NestedBinding::Value(var) => var,
                    NestedBinding::Cell(cell) => self.lowerer.states[&cell],
                };
                (id, var)
            })
            .collect();
        Ok(LoweredRegion {
            symbol_captures: sub.symbol_captures.clone(),
            output_infos: lowered
                .output_infos
                .iter()
                .map(|info| info.remap_symbols(|s| symbols[&s]))
                .collect(),
            program: lowered,
            bindings,
            initial,
        })
    }

    pub fn allocate_results(&mut self, infos: &[ValueInfo]) -> Vec<(Var, ValueInfo)> {
        infos
            .iter()
            .map(|info| (self.lowerer.scope.create_var(), info.clone()))
            .collect()
    }

    /// Bind the high-level results of one region known to have executed.
    pub fn bind_region_results(
        &mut self,
        outputs: &[(Var, Kind, ValueInfo)],
        region: &LoweredRegion,
        physical: &[(Var, ValueInfo)],
    ) -> Result<(), LoweringError> {
        if physical.len() != region.program.output_infos.len()
            || outputs.len() != region.program.results().len()
        {
            return Err(LoweringError::InvalidComputation);
        }
        let mut new_cells = BTreeMap::new();
        for ((var, kind, info), result) in outputs.iter().zip(region.program.results()) {
            match (kind, result) {
                (Kind::Value, Output::Value { result_idx }) => {
                    let source = region.source(*result_idx, physical);
                    let actual = physical
                        .iter()
                        .find_map(|(v, i)| (*v == source).then_some(i.clone()))
                        .map(Ok)
                        .unwrap_or_else(|| self.ssa_info(source))?;
                    if actual != *info {
                        return Err(LoweringError::RegionSignature(
                            "call result metadata mismatch",
                        ));
                    }
                    self.lowerer.values.insert(*var, source);
                }
                (Kind::Cell, Output::InputCell { input_id }) => {
                    let NestedBinding::Cell(cell) = region.bindings[input_id] else {
                        unreachable!()
                    };
                    self.lowerer.cells.insert(*var, cell);
                }
                (Kind::Cell, Output::NewCell { id, result_idx }) => {
                    let cell = *new_cells.entry(*id).or_insert_with(|| {
                        let cell = CellBinding::Local(self.lowerer.local_count);
                        self.lowerer.local_count += 1;
                        self.lowerer
                            .states
                            .insert(cell, region.source(*result_idx, physical));
                        cell
                    });
                    self.lowerer.cells.insert(*var, cell);
                }
                _ => return Err(LoweringError::InvalidComputation),
            }
        }
        Ok(())
    }

    /// Advance parent state from one region known to have executed.
    pub fn apply_region_updates(
        &mut self,
        region: &LoweredRegion,
        physical: &[(Var, ValueInfo)],
    ) -> Result<(), LoweringError> {
        if physical.len() != region.program.output_infos.len() {
            return Err(LoweringError::InvalidComputation);
        }
        for (id, result_idx) in &region.program.updates {
            let NestedBinding::Cell(cell) = region.bindings[id] else {
                unreachable!()
            };
            self.lowerer
                .states
                .insert(cell, region.source(*result_idx, physical));
            self.lowerer.written.insert(cell);
        }
        Ok(())
    }

    /// Emit the same operation and regions into both representations. The
    /// lowered parameter and operand lists are shared between the two forms.
    pub fn emit(
        &mut self,
        op: Op,
        inputs: Vec<Var>,
        outputs: Vec<(Var, ValueInfo)>,
        regions: Vec<RegionUse>,
    ) {
        let high = regions
            .iter()
            .map(|region| {
                HSubExpr::new(region.trace_body.clone(), region.captures.clone())
                    .with_symbol_captures(region.symbol_captures.clone())
            })
            .collect();
        let low = regions
            .into_iter()
            .map(|region| {
                SubExpr::new(region.expr, region.captures)
                    .with_symbol_captures(region.symbol_captures)
            })
            .collect();
        self.lowerer.trace_eqns.push(HInstr::Apply(HEqn::new(
            op.clone(),
            high,
            inputs.clone(),
            outputs
                .iter()
                .map(|(var, info)| (*var, Kind::Value, info.clone()))
                .collect(),
        )));
        self.lowerer
            .eqns
            .push(Eqn::new_with_subexprs(op, low, inputs, outputs));
    }
}

impl Lowerer {
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
        let mut return_vars = Vec::new();
        let mut result = |var: Var| {
            let index = return_vars.len();
            return_vars.push(var);
            index
        };
        let mut outputs = Vec::new();
        let mut new_cells = BTreeMap::new();
        for &var in expr.outputs() {
            let output = if self.cells.contains_key(&var) {
                match self.cell(var)? {
                    CellBinding::Input(input_id) => Output::InputCell { input_id },
                    cell @ CellBinding::Local(id) => Output::NewCell {
                        id,
                        result_idx: *new_cells
                            .entry(id)
                            .or_insert_with(|| result(self.states[&cell])),
                    },
                }
            } else {
                Output::Value {
                    result_idx: result(self.value(var)?),
                }
            };
            outputs.push(output);
        }
        // External updates are extra immutable results, not extra user results.
        let mut updates = Vec::new();
        for &cell in &self.written {
            if let CellBinding::Input(input_id) = cell {
                updates.push((input_id, result(self.states[&cell])));
            }
        }
        let expr = Expr {
            symbol_inputs: expr.symbol_inputs.clone(),
            symbol_closure: expr.symbol_closure.clone(),
            closure: self.closure,
            inputs: self.inputs,
            eqns: self.eqns,
            outputs: return_vars,
            name_hint: expr.name_hint.clone(),
        };
        expr.check()?;
        // At the execution boundary, skip forwarded parameters and compute
        // repeated SSA results only once. Expr itself still yields all results.
        let mut seen: BTreeSet<_> = expr
            .closure()
            .iter()
            .chain(expr.inputs())
            .map(|(var, _)| *var)
            .collect();
        let computed: Vec<_> = expr
            .outputs()
            .iter()
            .copied()
            .filter(|var| seen.insert(*var))
            .collect();
        let output_infos = computed
            .iter()
            .map(|var| {
                expr.value_info(var)
                    .cloned()
                    .ok_or(LoweringError::UndefinedVariable(*var))
            })
            .collect::<Result<_, _>>()?;
        let trace_params = |params: &[(Var, ValueInfo)]| {
            params
                .iter()
                .map(|(var, info)| (*var, Kind::Value, info.clone()))
                .collect()
        };
        let trace_body = Arc::new(HExpr {
            symbol_inputs: expr.symbol_inputs.clone(),
            symbol_closure: expr.symbol_closure.clone(),
            closure: trace_params(expr.closure()),
            inputs: trace_params(expr.inputs()),
            instructions: self.trace_eqns,
            outputs: computed,
            name_hint: expr.name_hint.clone(),
        });
        let mut needed = BTreeSet::new();
        let returned: BTreeSet<_> = expr.outputs().iter().copied().collect();
        for (id, (var, _)) in self
            .closure_sources
            .iter()
            .chain(&self.input_sources)
            .zip(expr.closure().iter().chain(expr.inputs()))
        {
            let used = returned.contains(var)
                || expr.eqns().iter().any(|eqn| {
                    eqn.inputs()
                        .iter()
                        .chain(eqn.subexprs().iter().flat_map(|sub| sub.closure()))
                        .any(|input| input == var)
                });
            if used {
                needed.insert(*id);
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

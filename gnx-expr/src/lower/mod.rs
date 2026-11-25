//! HExpr-to-Expr lowering, input specifications, and execution-boundary metadata.

mod replay;
mod structured;
pub use replay::CallError;
pub use structured::{FixedRegion, RegionArgument, RegionCell, StructuredRegion};

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;

use crate::check::BuildError;
use crate::{
    EqnSource, ExprBuilder, ExprRef, HEqn, HEqnSource, HExpr, HExprBuilder, HExprRef, HInstr,
    HInstrSource, HParam, HRootExpr, HSubExpr, HSubExprBuilder, Kind, Op, RootExpr, SubExprBuilder,
    Symbol, Var, VarInfo, VarScope,
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
    pub closure: Vec<(InputId, Kind, VarInfo)>,
    pub inputs: Vec<(InputId, Kind, VarInfo)>,
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
/// one to one with the closure inputs and inputs of `expr`, respectively.
/// Closure-only identities stay in closure; any identity appearing in explicit
/// inputs belongs to inputs, even when also captured. InputSpec supplies kinds
/// and aliases.
///
/// `outputs` assembles the user results. `updates` maps each written external
/// cell to a final value in the outputs of `expr`. Parameter forwarding is
/// expressed directly by returning its variable.
#[derive(Clone, Debug)]
pub struct LoweredExpr {
    spec: InputSpec,
    snapshot_inputs: BTreeSet<InputId>,
    pub expr: Arc<RootExpr>,
    // `expr` returning only the outputs of the trace body
    computation: Arc<RootExpr>,
    trace_body: Arc<HRootExpr>,
    // The owned forms of `expr` and `trace_body`, to derive variants from
    source: ExprBuilder,
    trace_source: HExprBuilder,
    /// Metadata for distinct computed results, in replay-body output order.
    pub output_infos: Vec<VarInfo>,
    pub closure: Vec<InputId>,
    pub inputs: Vec<InputId>,
    pub outputs: Vec<Output>,
    pub updates: Vec<(InputId, usize)>,
}
impl LoweredExpr {
    /// Replay body returning only distinct computed results. The wrapper
    /// forwards any parameters directly from their snapshots.
    pub fn trace_body(&self) -> &Arc<HRootExpr> {
        &self.trace_body
    }
    pub fn results(&self) -> &[Output] {
        &self.outputs
    }

    pub(crate) fn input_id(&self, var: Var) -> Option<InputId> {
        (self.source.closure_inputs.iter().zip(&self.closure))
            .chain(self.source.inputs.iter().zip(&self.inputs))
            .find_map(|((parameter, _), &id)| (*parameter == var).then_some(id))
    }

    pub(crate) fn result_source(&self, result_idx: usize) -> Option<ResultSource> {
        let &var = self.source.outputs.get(result_idx)?;
        if let Some(id) = self.input_id(var) {
            Some(ResultSource::Parameter(id))
        } else {
            (self.trace_source.outputs.iter())
                .position(|&result| result == var)
                .map(ResultSource::Computed)
        }
    }

    /// `expr`, returning only the distinct computed results: the outputs
    /// of the trace body.
    pub fn computation_expr(&self) -> Arc<RootExpr> {
        self.computation.clone()
    }
}

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
    /// An operation or info that cannot be stored in a root.
    Unstorable(String),
}
impl fmt::Display for LoweringError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for LoweringError {}

impl From<BuildError> for LoweringError {
    fn from(error: BuildError) -> Self {
        match error {
            BuildError::UndefinedVar(var) => Self::UndefinedVariable(var),
            BuildError::DuplicateVar(var) => Self::DuplicateDefinition(var),
            BuildError::Hole => Self::InvalidComputation,
            BuildError::UndefinedSymbol(_) => Self::RegionSignature("undeclared metadata symbol"),
            BuildError::Kind { var, expected } => match expected {
                Kind::Value => Self::ExpectedValue(var),
                Kind::Cell => Self::ExpectedCell(var),
            },
            BuildError::SymbolArity { .. } => Self::RegionSignature("symbol capture arity"),
            BuildError::VarArity { expected, actual } => Self::CaptureArity { expected, actual },
            error => Self::Unstorable(error.to_string()),
        }
    }
}

/// Lower a high-level expression. There is intentionally no trace-DAG-to-Expr
/// shortcut.
pub fn lower(expr: HExpr<'_>, spec: &InputSpec) -> Result<LoweredExpr, LoweringError> {
    Lowerer::new(expr, spec)?.lower(expr)
}

impl HExpr<'_> {
    /// Resolve cell aliases and replace cell actions with immutable SSA flow.
    pub fn lower(self, spec: &InputSpec) -> Result<LoweredExpr, LoweringError> {
        lower(self, spec)
    }
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
    output_infos: Vec<VarInfo>,
    symbol_captures: Vec<Symbol>,
    program: LoweredExpr,
    bindings: BTreeMap<InputId, NestedBinding>,
    initial: BTreeMap<InputId, Var>,
}

impl LoweredRegion {
    /// Computed result metadata in the caller’s symbol namespace.
    pub fn output_infos(&self) -> &[VarInfo] {
        &self.output_infos
    }
    pub fn program(&self) -> &LoweredExpr {
        &self.program
    }

    pub fn inputs(&self) -> Vec<Var> {
        (self.program.inputs.iter())
            .map(|id| self.initial[id])
            .collect()
    }

    pub fn subexpr(&self) -> RegionUse {
        RegionUse {
            symbol_captures: self.symbol_captures.clone(),
            expr: self.program.computation_expr().handle().into(),
            trace_body: self.program.trace_body.clone().handle().into(),
            captures: (self.program.closure.iter())
                .map(|id| self.initial[id])
                .collect(),
        }
    }

    fn source(&self, result_idx: usize, physical: &[(Var, VarInfo)]) -> Var {
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
    symbol_captures: Vec<Symbol>,
    expr: ExprRef,
    trace_body: HExprRef,
    captures: Vec<Var>,
}

/// Shared SSA construction and cell bookkeeping for operation lowering rules.
pub struct LoweringContext<'a> {
    lowerer: &'a mut Lowerer,
}

struct Lowerer {
    symbols: BTreeSet<Symbol>,
    spec: InputSpec,
    scope: VarScope,
    values: BTreeMap<Var, Var>,
    cells: BTreeMap<Var, CellBinding>,
    states: BTreeMap<CellBinding, Var>,
    bound_inputs: BTreeMap<InputId, (Kind, Var, VarInfo)>,
    written: BTreeSet<CellBinding>,
    closure: Vec<(Var, VarInfo)>,
    closure_sources: Vec<InputId>,
    inputs: Vec<(Var, VarInfo)>,
    input_sources: Vec<InputId>,
    eqns: Vec<EqnSource>,
    trace_eqns: Vec<HInstrSource>,
    local_count: usize,
}
impl Lowerer {
    fn new(expr: HExpr<'_>, spec: &InputSpec) -> Result<Self, LoweringError> {
        let mut lowerer = Self {
            symbols: (expr.closure_symbols().iter().copied())
                .chain(expr.input_symbols())
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
        // Symbols of the lowered expression keep their names.
        lowerer.scope.reserve([], lowerer.symbols.iter().copied());
        let closure: Vec<_> = expr.closure_inputs().collect();
        let inputs: Vec<_> = expr.inputs().collect();
        for (is_closure, params, specs) in [
            (true, closure, spec.closure.as_slice()),
            (false, inputs, spec.inputs.as_slice()),
        ] {
            if params.len() != specs.len() {
                return Err(LoweringError::InputSpecArity {
                    closure: is_closure,
                    expected: params.len(),
                    actual: specs.len(),
                });
            }
            for (index, (param, (id, kind, info))) in params.iter().zip(specs).enumerate() {
                if param.kind() != *kind || param.info_ref() != *info {
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
                        lowerer.values.insert(param.var(), var);
                    }
                    Kind::Cell => {
                        let resource = CellBinding::Input(*id);
                        lowerer.cells.insert(param.var(), resource);
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
    fn lower_equation(&mut self, eqn: HEqn<'_>) -> Result<(), LoweringError> {
        (eqn.op().get()).lower(eqn, &mut LoweringContext { lowerer: self })
    }
}

impl LoweringContext<'_> {
    pub fn lower_value_apply(&mut self, eqn: HEqn<'_>) -> Result<(), LoweringError> {
        if eqn.subexprs().len() != 0 {
            return Err(LoweringError::UnsupportedNestedOperation);
        }
        self.lower_value_apply_with_regions(eqn, vec![])
    }

    /// Lower an operation whose regions accept independent value arguments and
    /// have no captures or mutable state, such as a reduction combiner.
    pub fn lower_closed_regions_apply(&mut self, eqn: HEqn<'_>) -> Result<(), LoweringError> {
        let regions = eqn
            .subexprs()
            .map(|region| {
                let body = region.expr();
                if !region.provided_vars().is_empty()
                    || body.closure_inputs().len() != 0
                    || body.inputs().any(|param| param.kind() != Kind::Value)
                    || (body.outputs().iter()).any(|var| body.kind(*var) != Some(Kind::Value))
                {
                    return Err(LoweringError::RegionSignature(
                        "expected a closed value region",
                    ));
                }
                let spec = InputSpec {
                    closure: vec![],
                    inputs: body
                        .inputs()
                        .enumerate()
                        .map(|(i, param)| {
                            (InputId::Input(i), param.kind(), param.info_ref().to_owned())
                        })
                        .collect(),
                };
                let program = lower(body, &spec)?;
                Ok(RegionUse {
                    symbol_captures: region.provided_symbols().to_vec(),
                    expr: program.expr.handle().into(),
                    trace_body: region.to_ref(),
                    captures: vec![],
                })
            })
            .collect::<Result<_, _>>()?;
        self.lower_value_apply_with_regions(eqn, regions)
    }

    fn lower_value_apply_with_regions(
        &mut self,
        eqn: HEqn<'_>,
        regions: Vec<RegionUse>,
    ) -> Result<(), LoweringError> {
        let inputs = eqn
            .inputs()
            .iter()
            .map(|&var| self.lowerer.value(var))
            .collect::<Result<_, _>>()?;
        let mut outputs = Vec::new();
        for output in eqn.outputs() {
            let Kind::Value = output.kind() else {
                return Err(LoweringError::ExpectedValue(output.var()));
            };
            if output.var().is_hole() {
                outputs.push((Var::hole(), output.info_ref().to_owned()));
                continue;
            }
            let var = self.lowerer.scope.create_var();
            self.lowerer.values.insert(output.var(), var);
            outputs.push((var, output.info_ref().to_owned()));
        }
        self.emit(eqn.op().to_owned(), inputs, outputs, regions);
        Ok(())
    }

    fn object_info(&self, var: Var, kind: Kind) -> Result<VarInfo, LoweringError> {
        match kind {
            Kind::Value => Ok(self.value_operand(var)?.1),
            Kind::Cell => Ok(self.cell_state(RegionCell(self.lowerer.cell(var)?))?.1),
        }
    }
    fn child_bindings(
        &self,
        sub: HSubExpr<'_>,
        actual_inputs: &[VarInfo],
    ) -> Result<BTreeMap<Symbol, Symbol>, LoweringError> {
        let body = sub.expr();
        if body.input_symbols().is_empty() && body.closure_symbols().is_empty() {
            return Ok(BTreeMap::new());
        }
        let captures = (sub.provided_vars().iter())
            .zip(body.closure_inputs())
            .map(|(v, param)| self.object_info(*v, param.kind()))
            .collect::<Result<Vec<_>, _>>()?;
        body.bind_symbols(sub.provided_symbols(), &captures, actual_inputs, |s| {
            self.lowerer.symbols.contains(&s)
        })
    }
    /// Lower a region whose explicit arguments are bound to parent variables.
    /// The operation picks the argument layout; aliases are shared across
    /// those arguments and the region's captures.
    pub fn lower_region(
        &self,
        sub: HSubExpr<'_>,
        operands: &[Var],
    ) -> Result<LoweredRegion, LoweringError> {
        let body = sub.expr();
        if sub.provided_vars().len() != body.closure_inputs().len() {
            return Err(LoweringError::CaptureArity {
                expected: body.closure_inputs().len(),
                actual: sub.provided_vars().len(),
            });
        }
        if operands.len() != body.inputs().len() {
            return Err(LoweringError::InputSpecArity {
                closure: false,
                expected: body.inputs().len(),
                actual: operands.len(),
            });
        }
        let actual = (operands.iter())
            .zip(body.inputs())
            .map(|(v, param)| self.object_info(*v, param.kind()))
            .collect::<Result<Vec<_>, _>>()?;
        let symbols = self.child_bindings(sub, &actual)?;
        let mut identities = BTreeMap::new();
        let mut bindings = BTreeMap::new();
        let mut bind = |operands: &[Var],
                        params: &mut dyn Iterator<Item = HParam<'_>>,
                        id: fn(usize) -> InputId| {
            operands
                .iter()
                .zip(params)
                .enumerate()
                .map(|(index, (&var, param))| {
                    let binding = match param.kind() {
                        Kind::Value => NestedBinding::Value(self.lowerer.value(var)?),
                        Kind::Cell => NestedBinding::Cell(self.lowerer.cell(var)?),
                    };
                    let id = *identities.entry(binding).or_insert_with(|| id(index));
                    bindings.insert(id, binding);
                    Ok((id, param.kind(), param.info_ref().to_owned()))
                })
                .collect::<Result<Vec<_>, LoweringError>>()
        };
        let inputs = bind(operands, &mut body.inputs(), InputId::Input)?;
        let closure = bind(
            sub.provided_vars(),
            &mut body.closure_inputs(),
            InputId::Closure,
        )?;
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
            symbol_captures: sub.provided_symbols().to_vec(),
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

    pub fn allocate_results(&mut self, infos: &[VarInfo]) -> Vec<(Var, VarInfo)> {
        infos
            .iter()
            .map(|info| (self.lowerer.scope.create_var(), info.clone()))
            .collect()
    }

    /// Bind the high-level results of one region known to have executed.
    pub fn bind_region_results<'b>(
        &mut self,
        outputs: impl ExactSizeIterator<Item = HParam<'b>>,
        region: &LoweredRegion,
        physical: &[(Var, VarInfo)],
    ) -> Result<(), LoweringError> {
        if physical.len() != region.program.output_infos.len()
            || outputs.len() != region.program.results().len()
        {
            return Err(LoweringError::InvalidComputation);
        }
        let mut new_cells = BTreeMap::new();
        for (output, result) in outputs.zip(region.program.results()) {
            let var = output.var();
            match (output.kind(), result) {
                (Kind::Value, Output::Value { result_idx }) => {
                    let source = region.source(*result_idx, physical);
                    let actual = physical
                        .iter()
                        .find_map(|(v, i)| (*v == source).then_some(i.clone()))
                        .map(Ok)
                        .unwrap_or_else(|| self.ssa_info(source))?;
                    if output.info_ref() != actual {
                        return Err(LoweringError::RegionSignature(
                            "call result metadata mismatch",
                        ));
                    }
                    self.lowerer.values.insert(var, source);
                }
                (Kind::Cell, Output::InputCell { input_id }) => {
                    let NestedBinding::Cell(cell) = region.bindings[input_id] else {
                        unreachable!()
                    };
                    self.lowerer.cells.insert(var, cell);
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
                    self.lowerer.cells.insert(var, cell);
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
        physical: &[(Var, VarInfo)],
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
        outputs: Vec<(Var, VarInfo)>,
        regions: Vec<RegionUse>,
    ) {
        let high = regions
            .iter()
            .map(|region| {
                HSubExprBuilder::new(region.trace_body.clone())
                    .with_vars(region.captures.iter().copied())
                    .with_symbols(region.symbol_captures.iter().copied())
            })
            .collect();
        let low = regions
            .into_iter()
            .map(|region| {
                SubExprBuilder::new(region.expr)
                    .with_vars(region.captures)
                    .with_symbols(region.symbol_captures)
            })
            .collect();
        (self.lowerer.trace_eqns).push(HInstrSource::Apply(HEqnSource::new(
            op.clone(),
            high,
            inputs.clone(),
            outputs
                .iter()
                .map(|(var, info)| (*var, Kind::Value, info.clone()))
                .collect(),
        )));
        (self.lowerer.eqns).push(EqnSource::new(op, low, inputs, outputs));
    }
}

impl Lowerer {
    fn lower(mut self, expr: HExpr<'_>) -> Result<LoweredExpr, LoweringError> {
        for instr in expr.instrs() {
            match instr {
                HInstr::Apply(eqn) => self.lower_equation(eqn)?,
                HInstr::CreateCell { cell, initial, .. } => {
                    let initial = self.value(initial)?;
                    let resource = CellBinding::Local(self.local_count);
                    self.local_count += 1;
                    self.cells.insert(cell, resource);
                    self.states.insert(resource, initial);
                }
                HInstr::ReadCell { cell, output, .. } => {
                    let resource = self.cell(cell)?;
                    self.values.insert(output, self.states[&resource]);
                }
                HInstr::WriteCell { cell, value } => {
                    let resource = self.cell(cell)?;
                    let value = self.value(value)?;
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
        let closure_symbols = expr.closure_symbols().to_vec();
        let name = expr.name_hint().map(str::to_owned);
        let mut source = ExprBuilder::from_parts(
            closure_symbols.clone(),
            self.closure,
            self.inputs,
            self.eqns,
            return_vars,
        );
        source.set_name_hint(name.clone());
        let expr = source.clone().build()?;
        expr.view().check()?;
        // At the execution boundary, skip forwarded parameters and compute
        // repeated SSA results only once. Expr itself still yields all results.
        let mut seen: BTreeSet<_> = (source.closure_inputs.iter().chain(&source.inputs))
            .map(|(var, _)| *var)
            .collect();
        let computed: Vec<_> = (source.outputs.iter().copied())
            .filter(|var| seen.insert(*var))
            .collect();
        let output_infos = computed
            .iter()
            .map(|var| {
                (source.value_info(*var).cloned()).ok_or(LoweringError::UndefinedVariable(*var))
            })
            .collect::<Result<_, _>>()?;
        let trace_params = |params: &[(Var, VarInfo)]| {
            params
                .iter()
                .map(|(var, info)| (*var, Kind::Value, info.clone()))
                .collect()
        };
        let mut trace_source = HExprBuilder::from_parts(
            closure_symbols,
            trace_params(&source.closure_inputs),
            trace_params(&source.inputs),
            self.trace_eqns,
            computed.clone(),
        );
        trace_source.set_name_hint(name);
        let trace_body = trace_source.clone().build()?;
        let computation = if source.outputs == computed {
            expr.clone()
        } else {
            let mut computation = source.clone();
            computation.outputs = computed;
            computation.build()?
        };
        let mut needed = BTreeSet::new();
        let returned: BTreeSet<_> = source.outputs.iter().copied().collect();
        for (id, (var, _)) in (self.closure_sources.iter().chain(&self.input_sources))
            .zip(source.closure_inputs.iter().chain(&source.inputs))
        {
            let used = returned.contains(var)
                || source.eqns.iter().any(|eqn| {
                    (eqn.inputs().iter())
                        .chain(eqn.subexprs().iter().flat_map(|sub| sub.provided_vars()))
                        .any(|input| input == var)
                });
            if used {
                needed.insert(*id);
            }
        }
        Ok(LoweredExpr {
            snapshot_inputs: needed,
            spec: self.spec,
            expr,
            computation,
            trace_body,
            source,
            trace_source,
            output_infos,
            closure: self.closure_sources,
            inputs: self.input_sources,
            outputs,
            updates,
        })
    }
}

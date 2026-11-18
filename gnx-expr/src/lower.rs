//! HExpr-to-lexpr lowering, input specifications, and execution-boundary metadata.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::lexpr;
use crate::{Eqn, Expr, HExpr, HInstr, Kind, Ownership, Var, VarScope};

/// Identity of a supplied value or cell. Repeated IDs share one low-level input,
/// including across captures and ordinary arguments. Kind and ownership must agree.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct InputId(pub usize);

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
        let mut bind = |(object, ownership): &(crate::ObjectTracer, Ownership)| {
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
            let next = InputId(identities.len());
            (
                *identities.entry(identity).or_insert(next),
                kind,
                *ownership,
            )
        };
        let closure = closure.iter().map(&mut bind).collect();
        let inputs = inputs.iter().map(&mut bind).collect();
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

/// Pure program and its execution-boundary mappings. Inputs correspond one to
/// one with expr.inputs(); InputSpec supplies their kinds and ownership.
///
/// outputs starts with user results, followed by one value descriptor per update.
/// updates indexes that descriptor table, NOT expr.outputs(). Its sources are
/// always InputValue or OutputValue. results() excludes the update-source suffix.
/// Physical owned returns are deduplicated; descriptors may share their slots.
/// A wrapper must retain/copy when distributing a value to multiple owners or
/// storing an input passthrough beyond its lifetime. No implicit clone is emitted.
#[derive(Clone, Debug)]
pub struct LoweredExpr {
    pub expr: lexpr::Expr,
    pub inputs: Vec<InputId>,
    pub outputs: Vec<Output>,
    pub updates: Vec<(InputId, usize)>,
}
impl LoweredExpr {
    pub fn results(&self) -> &[Output] {
        &self.outputs[..self.outputs.len() - self.updates.len()]
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
    AliasedInputOwnershipMismatch(InputId),
    UndefinedVariable(Var),
    ExpectedValue(Var),
    ExpectedCell(Var),
    MoveBorrowed(Var),
    BorrowedOutput(Var),
    NonFreshComputationOutput,
    InvalidComputation,
}
impl fmt::Display for LoweringError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for LoweringError {}

/// Lower a captured HExpr. There is intentionally no trace-DAG-to-lexpr shortcut.
pub fn lower(expr: &HExpr, spec: &InputSpec) -> Result<LoweredExpr, LoweringError> {
    Lowerer::new(expr, spec)?.lower(expr)
}

struct Lowerer {
    scope: VarScope,
    values: BTreeMap<Var, Var>,
    cells: BTreeMap<Var, CellBinding>,
    states: BTreeMap<CellBinding, Var>,
    bound_inputs: BTreeMap<InputId, (Kind, Ownership, Var)>,
    written: BTreeSet<CellBinding>,
    inputs: Vec<(Var, Ownership)>,
    input_sources: Vec<InputId>,
    eqns: Vec<lexpr::Eqn>,
    local_count: usize,
}
impl Lowerer {
    fn new(expr: &HExpr, spec: &InputSpec) -> Result<Self, LoweringError> {
        let mut lowerer = Self {
            scope: VarScope::new(),
            values: BTreeMap::new(),
            cells: BTreeMap::new(),
            states: BTreeMap::new(),
            bound_inputs: BTreeMap::new(),
            written: BTreeSet::new(),
            inputs: vec![],
            input_sources: vec![],
            eqns: vec![],
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
            for (index, (param, (id, kind, ownership))) in params.iter().zip(specs).enumerate() {
                if param.kind() != kind {
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
                    lowerer.inputs.push((var, *ownership));
                    lowerer.input_sources.push(*id);
                    var
                };
                match kind {
                    Kind::Value(_) => {
                        lowerer.values.insert(*param.var(), var);
                    }
                    Kind::Cell(_) => {
                        let resource = CellBinding::Input(*id);
                        lowerer.cells.insert(*param.var(), resource);
                        lowerer.states.entry(resource).or_insert(var);
                    }
                }
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
    fn lower(mut self, expr: &HExpr) -> Result<LoweredExpr, LoweringError> {
        for instr in expr.instructions() {
            match instr {
                HInstr::Apply(eqn) => {
                    lower_computation(eqn, &mut self.scope, &mut self.values, &mut self.eqns)?
                }
                HInstr::CreateCell { cell, initial } => {
                    let initial = self.value(*initial)?;
                    let resource = CellBinding::Local(self.local_count);
                    self.local_count += 1;
                    self.cells.insert(*cell, resource);
                    self.states.insert(resource, initial);
                }
                HInstr::ReadCell { cell, output } => {
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
            .inputs
            .iter()
            .zip(&self.input_sources)
            .filter(|((_, ownership), _)| *ownership == Ownership::Borrow)
            .map(|((var, _), id)| (*var, *id))
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
        let mut expr = lexpr::Expr {
            closure: vec![],
            inputs: self.inputs,
            eqns: self.eqns,
            outputs: return_vars,
        };
        assign_ownership(&mut expr);
        expr.check()?;
        Ok(LoweredExpr {
            expr,
            inputs: self.input_sources,
            outputs,
            updates,
        })
    }
}

fn assign_ownership(expr: &mut lexpr::Expr) {
    let owned: BTreeSet<Var> = expr
        .closure
        .iter()
        .chain(&expr.inputs)
        .filter(|(_, ownership)| *ownership == Ownership::Move)
        .map(|(var, _)| *var)
        .chain(expr.eqns.iter().flat_map(|eqn| eqn.outputs.iter().copied()))
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

fn lookup(values: &BTreeMap<Var, Var>, var: &Var) -> Result<Var, LoweringError> {
    values
        .get(var)
        .copied()
        .ok_or(LoweringError::UndefinedVariable(*var))
}

fn lower_computation(
    eqn: &Eqn,
    scope: &mut VarScope,
    values: &mut BTreeMap<Var, Var>,
    equations: &mut Vec<lexpr::Eqn>,
) -> Result<(), LoweringError> {
    if eqn.outputs().iter().any(|o| !o.is_new()) {
        return Err(LoweringError::NonFreshComputationOutput);
    }
    // An immutable call can forward parameters. Inline it so those aliases reach
    // the enclosing boundary, where borrowed passthroughs can be removed.
    if eqn.op() == &crate::builtins::call_op() {
        if eqn.subexprs().len() != 1 {
            return Err(LoweringError::InvalidComputation);
        }
        let sub = &eqn.subexprs()[0];
        let body = sub.expr();
        validate_immutable_expr(body)?;
        if sub.closure().len() != body.closure().len()
            || eqn.inputs().len() != body.inputs().len()
            || eqn.outputs().len() != body.outputs().len()
        {
            return Err(LoweringError::InvalidComputation);
        }
        let mut inner = BTreeMap::new();
        for (formal, actual) in body
            .closure()
            .iter()
            .zip(sub.closure())
            .chain(body.inputs().iter().zip(eqn.inputs()))
        {
            inner.insert(*formal.var(), lookup(values, actual.var())?);
        }
        for nested in body.eqns() {
            lower_computation(nested, scope, &mut inner, equations)?;
        }
        for (formal, actual) in eqn.outputs().iter().zip(body.outputs()) {
            values.insert(*formal.var(), lookup(&inner, actual)?);
        }
        return Ok(());
    }
    let inputs = eqn
        .inputs()
        .iter()
        .map(|i| lookup(values, i.var()).map(|var| (var, Ownership::Borrow)))
        .collect::<Result<_, _>>()?;
    let subexprs = eqn
        .subexprs()
        .iter()
        .map(|sub| {
            Ok(lexpr::SubExpr {
                captures: sub
                    .closure()
                    .iter()
                    .map(|i| lookup(values, i.var()).map(|var| (var, Ownership::Borrow)))
                    .collect::<Result<_, _>>()?,
                expr: lower_immutable_expr(sub.expr())?,
            })
        })
        .collect::<Result<_, LoweringError>>()?;
    let mut outputs = Vec::new();
    for output in eqn.outputs() {
        let var = scope.create_var();
        values.insert(*output.var(), var);
        outputs.push(var);
    }
    equations.push(lexpr::Eqn {
        op: eqn.op().clone(),
        inputs,
        outputs,
        subexprs,
    });
    Ok(())
}

pub(crate) fn validate_immutable_expr(expr: &Expr) -> Result<(), LoweringError> {
    expr.check()
        .map_err(|_| LoweringError::InvalidComputation)?;
    for eqn in expr.eqns() {
        if eqn.outputs().iter().any(|o| !o.is_new()) {
            return Err(LoweringError::NonFreshComputationOutput);
        }
        for sub in eqn.subexprs() {
            validate_immutable_expr(sub.expr())?;
        }
    }
    Ok(())
}
fn lower_immutable_expr(expr: &Expr) -> Result<lexpr::Expr, LoweringError> {
    validate_immutable_expr(expr)?;
    let mut scope = VarScope::new();
    let mut values = BTreeMap::new();
    for input in expr.closure().iter().chain(expr.inputs()) {
        values.insert(*input.var(), scope.create_var());
    }
    let closure = expr
        .closure()
        .iter()
        .map(|i| (values[i.var()], Ownership::Borrow))
        .collect();
    let inputs = expr
        .inputs()
        .iter()
        .map(|i| (values[i.var()], Ownership::Borrow))
        .collect();
    let mut eqns = Vec::new();
    for eqn in expr.eqns() {
        lower_computation(eqn, &mut scope, &mut values, &mut eqns)?;
    }
    let outputs = expr
        .outputs()
        .iter()
        .map(|v| lookup(&values, v))
        .collect::<Result<_, _>>()?;
    let mut expr = lexpr::Expr {
        closure,
        inputs,
        eqns,
        outputs,
    };
    assign_ownership(&mut expr);
    expr.check()?;
    Ok(expr)
}

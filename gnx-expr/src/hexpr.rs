//! Mutation-aware expressions captured from a trace DAG.
//!
//! Cell accesses are structural instructions, never dialect operations. Lowering
//! requires explicit alias and ownership bindings and produces an immutable [`crate::lexpr::Expr`].

use std::collections::BTreeMap;
use std::fmt;

use crate::{Eqn, InputSpec, Kind, LoweredExpr, LoweringError, Var};

/// Derived summary of structural cell accesses. Not a computation effect or
/// ownership annotation; immutable value parameters have no cell-access summary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CellAccess {
    Unused,
    Read,
    Write,
    ReadWrite,
}
impl CellAccess {
    pub fn combine(self, other: Self) -> Self {
        match (self, other) {
            (Self::Unused, _) => other,
            (_, Self::Unused) => self,
            (Self::ReadWrite, _) | (_, Self::ReadWrite) => Self::ReadWrite,
            (Self::Read, Self::Write) | (Self::Write, Self::Read) => Self::ReadWrite,
            _ => self,
        }
    }
}

#[derive(Clone, Debug)]
pub struct HParam {
    pub(crate) var: Var,
    pub(crate) kind: Kind,
}
impl HParam {
    pub fn var(&self) -> &Var {
        &self.var
    }
    pub fn kind(&self) -> &Kind {
        &self.kind
    }
}

#[derive(Clone, Debug)]
pub enum HInstr {
    Apply(Eqn),
    CreateCell { cell: Var, initial: Var },
    ReadCell { cell: Var, output: Var },
    WriteCell { cell: Var, value: Var },
}
impl HInstr {
    pub fn as_apply(&self) -> Option<&Eqn> {
        match self {
            Self::Apply(eqn) => Some(eqn),
            _ => None,
        }
    }
}

/// Captured high-level expression. Variables distinguish values from cells.
///
/// ```
/// use gnx_expr::{Capture, Generic, InputId, InputSpec, Kind, Ownership,
///                CellTrace, TraceContext, ObjectTracer, Tracer, ValueInfo};
/// let guard = TraceContext::enter();
/// let info = ValueInfo::new(0usize);
/// let x = Tracer::<Generic>::placeholder(info.clone());
/// let cell = CellTrace::placeholder(info.clone());
/// cell.set(x.trace_ref().clone());
/// let result = cell.get();
/// let captured = Capture::from_context(
///     guard.context(),
///     &[ObjectTracer::from(cell.clone()), ObjectTracer::from(x)],
///     &[ObjectTracer::from(result)],
/// ).unwrap();
/// let spec = InputSpec {
///     closure: vec![],
///     inputs: vec![
///         (InputId(0), Kind::Cell(info.clone()), Ownership::Borrow),
///         (InputId(1), Kind::Value(info), Ownership::Move),
///     ],
/// };
/// let lowered = captured.expr().lower(&spec).unwrap();
/// assert!(lowered.expr.eqns().is_empty()); // Cell actions became value flow.
/// ```
#[derive(Clone, Debug)]
pub struct HExpr {
    pub(crate) closure: Vec<HParam>,
    pub(crate) inputs: Vec<HParam>,
    pub(crate) instructions: Vec<HInstr>,
    pub(crate) outputs: Vec<Var>,
    pub(crate) kinds: BTreeMap<Var, Kind>,
}
impl HExpr {
    pub fn closure(&self) -> &[HParam] {
        &self.closure
    }
    pub fn inputs(&self) -> &[HParam] {
        &self.inputs
    }
    pub fn instructions(&self) -> &[HInstr] {
        &self.instructions
    }
    pub fn outputs(&self) -> &[Var] {
        &self.outputs
    }
    pub fn kind(&self, var: &Var) -> Option<&Kind> {
        self.kinds.get(var)
    }

    /// Derive access to this cell directly from structural instructions. Merely
    /// returning a handle is not a read of its contents. Values return None.
    pub fn cell_access(&self, var: &Var) -> Option<CellAccess> {
        if !matches!(self.kind(var), Some(Kind::Cell(_))) {
            return None;
        }
        Some(
            self.instructions
                .iter()
                .fold(CellAccess::Unused, |access, instr| {
                    let next = match instr {
                        HInstr::ReadCell { cell, .. } if cell == var => CellAccess::Read,
                        HInstr::WriteCell { cell, .. } if cell == var => CellAccess::Write,
                        _ => CellAccess::Unused,
                    };
                    access.combine(next)
                }),
        )
    }

    /// Resolve cell aliases, eliminate cell actions, and assign borrow/move
    /// uses according to the caller-provided ownership of the initial inputs.
    pub fn lower(&self, spec: &InputSpec) -> Result<LoweredExpr, LoweringError> {
        crate::lower::lower(self, spec)
    }

    /// Instantiate a captured body into the active trace DAG, without rerunning
    /// the Rust function that originally produced it. Cell actions use the
    /// caller's handles, so local reference forwarding and aliases are preserved.
    /// Calls are currently inlined into the DAG rather than represented by a
    /// separate higher-order trace node.
    pub fn apply(
        &self,
        closure: &[crate::ObjectTracer],
        inputs: &[crate::ObjectTracer],
    ) -> Result<Vec<crate::ObjectTracer>, HApplyError> {
        use crate::{Invocation, CellTrace, TraceInput, ObjectTracer};
        if crate::TraceContext::current().is_none() {
            return Err(HApplyError::NoActiveContext);
        }
        if closure.len() != self.closure.len() || inputs.len() != self.inputs.len() {
            return Err(HApplyError::Arity);
        }
        let mut objects = BTreeMap::new();
        for (param, object) in self
            .closure
            .iter()
            .chain(&self.inputs)
            .zip(closure.iter().chain(inputs))
        {
            let matches = match (&param.kind, object) {
                (Kind::Value(info), ObjectTracer::Ref(value)) => info == value.info(),
                (Kind::Cell(info), ObjectTracer::Cell(cell)) => info == cell.type_info(),
                _ => false,
            };
            if !matches {
                return Err(HApplyError::ArgumentType(param.var));
            }
            objects.insert(param.var, object.clone());
        }
        for instr in self.instructions() {
            match instr {
                HInstr::Apply(eqn) => {
                    let arguments = eqn
                        .inputs()
                        .iter()
                        .map(|i| applied_value(&objects, *i.var()).map(TraceInput::from))
                        .collect::<Result<Vec<_>, _>>()?;
                    let subexprs = eqn
                        .subexprs()
                        .iter()
                        .map(|sub| {
                            let bindings = sub
                                .closure()
                                .iter()
                                .map(|i| applied_value(&objects, *i.var()).map(TraceInput::from))
                                .collect::<Result<_, _>>()?;
                            Ok((sub.expr().clone(), bindings))
                        })
                        .collect::<Result<_, HApplyError>>()?;
                    let infos = eqn
                        .outputs()
                        .iter()
                        .map(|out| match self.kind(out.var()) {
                            Some(Kind::Value(info)) => Ok(info.clone()),
                            _ => Err(HApplyError::InvalidBody),
                        })
                        .collect::<Result<_, _>>()?;
                    let results = Invocation::invoke(eqn.op().clone(), subexprs, arguments, infos);
                    for (out, result) in eqn.outputs().iter().zip(results) {
                        objects.insert(*out.var(), ObjectTracer::from(result));
                    }
                }
                HInstr::CreateCell { cell, initial } => {
                    let value = applied_value(&objects, *initial)?;
                    objects.insert(*cell, ObjectTracer::Cell(CellTrace::new(value)));
                }
                HInstr::ReadCell { cell, output } => {
                    let value = applied_cell(&objects, *cell)?.get();
                    objects.insert(*output, ObjectTracer::Ref(value));
                }
                HInstr::WriteCell { cell, value } => {
                    applied_cell(&objects, *cell)?.set(applied_value(&objects, *value)?)
                }
            }
        }
        self.outputs
            .iter()
            .map(|var| objects.get(var).cloned().ok_or(HApplyError::InvalidBody))
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HApplyError {
    NoActiveContext,
    Arity,
    ArgumentType(Var),
    InvalidBody,
}
impl fmt::Display for HApplyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for HApplyError {}

fn applied_value(
    objects: &BTreeMap<Var, crate::ObjectTracer>,
    var: Var,
) -> Result<crate::Trace, HApplyError> {
    match objects.get(&var) {
        Some(crate::ObjectTracer::Ref(value)) => Ok(value.clone()),
        _ => Err(HApplyError::InvalidBody),
    }
}
fn applied_cell(
    objects: &BTreeMap<Var, crate::ObjectTracer>,
    var: Var,
) -> Result<&crate::CellTrace, HApplyError> {
    match objects.get(&var) {
        Some(crate::ObjectTracer::Cell(cell)) => Ok(cell),
        _ => Err(HApplyError::InvalidBody),
    }
}

impl fmt::Display for HExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "hexpr {{")?;
        for param in self.closure.iter().chain(&self.inputs) {
            writeln!(f, "  {:?}: {:?}", param.var, param.kind)?;
        }
        for instr in &self.instructions {
            match instr {
                HInstr::Apply(eqn) => {
                    writeln!(f, "  {:?} = {} {:?}", eqn.outputs(), eqn.op(), eqn.inputs())?
                }
                HInstr::CreateCell { cell, initial } => {
                    writeln!(f, "  {cell:?} = create_cell {initial:?}")?
                }
                HInstr::ReadCell { cell, output } => {
                    writeln!(f, "  {output:?} = read_cell {cell:?}")?
                }
                HInstr::WriteCell { cell, value } => {
                    writeln!(f, "  write_cell {cell:?} {value:?}")?
                }
            }
        }
        writeln!(f, "  return {:?}", self.outputs)?;
        write!(f, "}}")
    }
}

//! Mutation-aware expressions captured from a trace DAG.
//!
//! Cell accesses are structural instructions, never dialect operations. Lowering
//! requires explicit alias bindings and produces an immutable [`crate::Expr`].

use std::borrow::Cow;
use std::fmt;
use std::sync::Arc;

use crate::{InputSpec, Kind, LoweredExpr, LoweringError, Op, Var};

#[derive(Clone, Debug)]
pub struct HSubExpr {
    pub expr: Arc<HExpr>,
    pub captures: Vec<Var>,
}
impl HSubExpr {
    pub fn new(expr: Arc<HExpr>, captures: Vec<Var>) -> Self {
        Self { expr, captures }
    }
    pub fn expr(&self) -> &Arc<HExpr> {
        &self.expr
    }
    pub fn closure(&self) -> &[Var] {
        &self.captures
    }
}

/// A high-level equation with value or cell operands.
#[derive(Clone, Debug)]
pub struct HEqn {
    op: Op,
    subexprs: Vec<HSubExpr>,
    inputs: Vec<Var>,
    outputs: Vec<(Var, Kind)>,
}
impl HEqn {
    pub fn new(
        op: Op,
        subexprs: Vec<HSubExpr>,
        inputs: Vec<Var>,
        outputs: Vec<(Var, Kind)>,
    ) -> Self {
        Self {
            op,
            subexprs,
            inputs,
            outputs,
        }
    }
    pub fn op(&self) -> &Op {
        &self.op
    }
    pub fn subexprs(&self) -> &[HSubExpr] {
        &self.subexprs
    }
    pub fn inputs(&self) -> &[Var] {
        &self.inputs
    }
    pub fn outputs(&self) -> &[(Var, Kind)] {
        &self.outputs
    }
}

/// Derived summary of structural cell accesses. Immutable value parameters
/// have no cell-access summary.
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
pub enum HInstr {
    Apply(HEqn),
    /// Defines `cell` with the contents metadata of `initial`.
    CreateCell {
        cell: Var,
        initial: Var,
        kind: Kind,
    },
    /// Defines `output` with the contents metadata of `cell`.
    ReadCell {
        cell: Var,
        output: Var,
        kind: Kind,
    },
    WriteCell {
        cell: Var,
        value: Var,
    },
}
impl HInstr {
    pub fn as_apply(&self) -> Option<&HEqn> {
        match self {
            Self::Apply(eqn) => Some(eqn),
            _ => None,
        }
    }
}

/// Captured high-level expression. Variables distinguish values from cells.
///
/// ```
/// use gnx_expr::{Capture, Generic, InputId, InputSpec, Kind,
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
///         (InputId::Input(0), Kind::Cell(info.clone())),
///         (InputId::Input(1), Kind::Value(info)),
///     ],
/// };
/// let lowered = captured.expr().lower(&spec).unwrap();
/// assert!(lowered.expr.eqns().is_empty()); // Cell actions became value flow.
/// ```
#[derive(Clone, Debug)]
pub struct HExpr {
    pub(crate) closure: Vec<(Var, Kind)>,
    pub(crate) inputs: Vec<(Var, Kind)>,
    pub(crate) instructions: Vec<HInstr>,
    pub(crate) outputs: Vec<Var>,
    pub(crate) name_hint: Option<Cow<'static, str>>,
}
impl HExpr {
    /// Move the flagged explicit inputs, in order, to the end of the closure.
    /// Transformations use this to turn arguments they don't map or
    /// differentiate into captures of the body.
    ///
    /// # Panics
    /// If `captured` doesn't have one flag per input.
    pub fn capture_inputs(&self, captured: &[bool]) -> HExpr {
        assert_eq!(captured.len(), self.inputs.len(), "one flag per input");
        let mut closure = self.closure.clone();
        let mut inputs = Vec::new();
        for (param, &capture) in self.inputs.iter().zip(captured) {
            if capture {
                closure.push(param.clone());
            } else {
                inputs.push(param.clone());
            }
        }
        HExpr {
            closure,
            inputs,
            instructions: self.instructions.clone(),
            outputs: self.outputs.clone(),
            name_hint: self.name_hint.clone(),
        }
    }

    pub fn with_name_hint(mut self, hint: impl Into<Cow<'static, str>>) -> Self {
        self.name_hint = Some(hint.into());
        self
    }

    pub fn name_hint(&self) -> Option<&str> {
        self.name_hint.as_deref()
    }

    pub fn pretty(&self) -> crate::util::pretty::PrettyHExpr<'_> {
        crate::util::pretty::PrettyHExpr::new(self)
    }

    pub fn closure(&self) -> &[(Var, Kind)] {
        &self.closure
    }
    pub fn inputs(&self) -> &[(Var, Kind)] {
        &self.inputs
    }
    pub fn instructions(&self) -> &[HInstr] {
        &self.instructions
    }
    pub fn outputs(&self) -> &[Var] {
        &self.outputs
    }
    /// Find a variable's kind at its definition site.
    pub fn kind(&self, var: &Var) -> Option<&Kind> {
        self.closure
            .iter()
            .chain(&self.inputs)
            .find_map(|(defined, kind)| (defined == var).then_some(kind))
            .or_else(|| {
                self.instructions
                    .iter()
                    .rev()
                    .find_map(|instr| match instr {
                        HInstr::Apply(eqn) => eqn
                            .outputs()
                            .iter()
                            .find_map(|(defined, kind)| (defined == var).then_some(kind)),
                        HInstr::CreateCell { cell, kind, .. } if cell == var => Some(kind),
                        HInstr::ReadCell { output, kind, .. } if output == var => Some(kind),
                        _ => None,
                    })
            })
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
                        HInstr::Apply(eqn) => eqn.subexprs().iter().enumerate().fold(
                            CellAccess::Unused,
                            |access, (region, sub)| {
                                let body = sub.expr();
                                body.closure()
                                    .iter()
                                    .zip(sub.captures.iter().copied().map(Some))
                                    .chain(body.inputs().iter().enumerate().map(
                                        |(param, binding)| {
                                            (
                                                binding,
                                                eqn.op()
                                                    .operation()
                                                    .region_operand(eqn, region, param),
                                            )
                                        },
                                    ))
                                    .filter(|(_, operand)| *operand == Some(*var))
                                    .fold(access, |access, ((param, _), _)| {
                                        access.combine(
                                            body.cell_access(param).unwrap_or(CellAccess::Unused),
                                        )
                                    })
                            },
                        ),
                        _ => CellAccess::Unused,
                    };
                    access.combine(next)
                }),
        )
    }

    /// Resolve cell aliases and replace cell actions with immutable SSA flow.
    pub fn lower(&self, spec: &InputSpec) -> Result<LoweredExpr, LoweringError> {
        crate::lower::lower(self, spec)
    }
}

/// Formats the high-level IR, hoisting large nested bodies. Metadata is displayed
/// as a value, not serialized back to Rust source.
impl fmt::Display for HExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        crate::util::pretty::PrettyVarScope::new().fmt_hexpr(self, f)
    }
}

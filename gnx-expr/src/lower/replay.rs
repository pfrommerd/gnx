//! Using a lowered expression from a trace: deriving the input
//! specification from actual arguments, and replaying the expression.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use super::{InputId, InputSpec, LoweredExpr, LoweringError, Output, ResultSource};
use crate::symbol::symbol_substitution;
use crate::trace::{CellKey, CellTrace, ObjectTracer, Trace, TraceContext, TracerKey};
use crate::{HRootExpr, Kind, Symbol, VarInfo};

impl InputSpec {
    /// Derive shared identities and kinds from actual arguments. Equal values
    /// are not merged unless their trace identities match.
    pub fn from_args(closure: &[ObjectTracer], inputs: &[ObjectTracer]) -> Self {
        #[derive(PartialEq, Eq, PartialOrd, Ord)]
        enum Identity {
            Value(TracerKey),
            Cell(CellKey),
        }
        let mut identities = BTreeMap::new();
        let mut bind = |(index, object): (InputId, &ObjectTracer)| {
            let (identity, kind, info) = match object {
                ObjectTracer::Ref(value) => (
                    Identity::Value(TracerKey::from(value)),
                    Kind::Value,
                    value.info().clone(),
                ),
                ObjectTracer::Cell(cell) => (
                    Identity::Cell(CellKey::from(cell)),
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

impl LoweredExpr {
    /// Emit an immutable application using `emit` and materialize its boundary. All required cell
    /// snapshots are taken before any updates; repeated local-cell IDs share one
    /// fresh handle. Returning an input cell alone does not read its contents.
    ///
    /// The caller must supply the alias specialization used during lowering.
    /// Sources are resolved directly through stored InputId offsets; this method
    /// checks arity and metadata but does not rediscover or validate identities.
    pub fn replay(
        &self,
        closure: &[ObjectTracer],
        inputs: &[ObjectTracer],
        emit: impl FnOnce(Arc<HRootExpr>, Vec<Trace>, Vec<Trace>, Vec<VarInfo>) -> Vec<Trace>,
    ) -> Result<Vec<ObjectTracer>, CallError> {
        let symbols = self.source.closure_symbols.clone();
        self.replay_with_symbols(&symbols, closure, inputs, emit)
    }
    /// Replay with explicit symbols from the enclosing trace.
    pub fn replay_with_symbols(
        &self,
        symbol_captures: &[Symbol],
        closure: &[ObjectTracer],
        inputs: &[ObjectTracer],
        emit: impl FnOnce(Arc<HRootExpr>, Vec<Trace>, Vec<Trace>, Vec<VarInfo>) -> Vec<Trace>,
    ) -> Result<Vec<ObjectTracer>, CallError> {
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
        let symbols = symbol_substitution(
            &self.source.closure_symbols,
            symbol_captures,
            &self.expr.view().input_symbols(),
            (spec.closure.iter().chain(&spec.inputs))
                .map(|v| v.2.symbols())
                .zip(actuals.map(|info| info.symbols())),
            |s| TraceContext::current().is_some_and(|c| c.has_symbol(s)),
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
        (self.expr.view().check()).map_err(CallError::InvalidExpr)?;
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
            (self.trace_body.view().substitute_symbols(&symbols).build())
                .map_err(|error| CallError::InvalidExpr(error.into()))?
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

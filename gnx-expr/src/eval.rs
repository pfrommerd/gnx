//! Prepare a captured DAG for execution and publish its concrete results.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use crate::{
    Capture, ConcreteValue, Expr, InputId, InputSpec, LoweringError, ObjectTracer, Trace, Value,
    ValueInfo,
};

// Capture and publication share this gate so a capture sees a consistent
// set of pending producers. Never hold it across execution or await.
pub(crate) static EVAL_DAG: Mutex<()> = Mutex::new(());

#[derive(Debug, thiserror::Error)]
pub enum EvalPlanError {
    #[error("cannot execute an eval capture with an unbound placeholder")]
    UnboundPlaceholder,
    #[error("eval cannot publish outer cell updates or reconstruct cell outputs")]
    CellEffect,
    #[error("could not lower the eval capture: {0}")]
    Lowering(#[from] LoweringError),
    #[error("invalid eval boundary mapping")]
    Boundary,
    #[error("eval input {input:?} has metadata {actual}, expected {expected}")]
    InputInfo {
        input: InputId,
        expected: ValueInfo,
        actual: ValueInfo,
    },
    #[error("device returned {actual} values, expected {expected}")]
    OutputArity { expected: usize, actual: usize },
    #[error("eval output {index} has metadata {actual}, expected {expected}")]
    OutputInfo {
        index: usize,
        expected: ValueInfo,
        actual: ValueInfo,
    },
}

impl Capture {
    /// Prepare a capture without outer cell writes or cell results. Lowering
    /// handles local cell state; eval only publishes immutable values.
    pub fn prepare(self) -> Result<PreparedEval, EvalPlanError> {
        let (expr, closure, inputs, outputs) = self.into_parts();
        let outputs = outputs
            .into_iter()
            .map(value_trace)
            .collect::<Result<Vec<_>, _>>()?;
        let spec = InputSpec::from_args(&closure, &inputs);
        let program = crate::lower::lower(&expr, &spec)?;
        if !program.updates.is_empty() {
            return Err(EvalPlanError::CellEffect);
        }
        if program.results().len() != outputs.len() {
            return Err(EvalPlanError::Boundary);
        }
        // The lowered parameter lists contain each canonical binding once.
        // Use the exact same snapshot for both backend args and passthroughs.
        let mut snapshots = BTreeMap::new();
        let args = program
            .closure
            .iter()
            .chain(&program.inputs)
            .copied()
            .map(|id| {
                let object = match id {
                    InputId::Closure(index) => closure.get(index),
                    InputId::Input(index) => inputs.get(index),
                }
                .ok_or(EvalPlanError::Boundary)?;
                let value = snapshot(id, object)?;
                snapshots.insert(id, value.clone());
                Ok(value)
            })
            .collect::<Result<_, EvalPlanError>>()?;
        let sources = program
            .results()
            .iter()
            .map(|source| {
                let crate::lower::Output::Value { result_idx } = *source else {
                    return Err(EvalPlanError::CellEffect);
                };
                match program
                    .result_source(result_idx)
                    .ok_or(EvalPlanError::Boundary)?
                {
                    crate::lower::ResultSource::Computed(index) => Ok(Source::Result(index)),
                    crate::lower::ResultSource::Parameter(id) => snapshots
                        .get(&id)
                        .cloned()
                        .map(Source::Input)
                        .ok_or(EvalPlanError::Boundary),
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let expr = program.computation_expr();
        let commit = EvalCommit {
            expected: program.output_infos,
            writes: outputs.into_iter().zip(sources).collect(),
        };
        drop(program.expr);
        Ok(PreparedEval {
            expr: Arc::unwrap_or_clone(expr),
            args,
            commit,
        })
    }
}

fn value_trace(object: ObjectTracer) -> Result<Trace, EvalPlanError> {
    match object {
        ObjectTracer::Ref(trace) => Ok(trace),
        ObjectTracer::Cell(_) => Err(EvalPlanError::CellEffect),
    }
}

fn snapshot(input: InputId, object: &ObjectTracer) -> Result<Value, EvalPlanError> {
    let home;
    let trace = match object {
        ObjectTracer::Ref(trace) => trace,
        ObjectTracer::Cell(cell) => {
            home = cell.read_home();
            &home
        }
    };
    let value = trace.try_concrete().ok_or_else(|| {
        if trace.is_placeholder() {
            EvalPlanError::UnboundPlaceholder
        } else {
            EvalPlanError::Boundary
        }
    })?;
    let actual = value.to_info();
    let expected = trace.info();
    if actual != *expected {
        return Err(EvalPlanError::InputInfo {
            input,
            expected: expected.clone(),
            actual,
        });
    }
    Ok(value.clone())
}

pub struct PreparedEval {
    expr: Expr,
    args: Vec<Value>,
    commit: EvalCommit,
}

impl PreparedEval {
    pub fn into_parts(self) -> (Expr, Vec<Value>, EvalCommit) {
        (self.expr, self.args, self.commit)
    }
}

enum Source {
    Input(Value),
    Result(usize),
}

impl Source {
    fn value<'a>(&'a self, outputs: &'a [Value]) -> &'a Value {
        match self {
            Self::Input(value) => value,
            Self::Result(index) => &outputs[*index],
        }
    }
}

pub struct EvalCommit {
    expected: Vec<ValueInfo>,
    writes: Vec<(Trace, Source)>,
}

impl EvalCommit {
    /// Check the full output batch before materializing any slot. Concurrent
    /// evaluations may duplicate work; they never replace a concrete slot.
    pub fn publish(self, outputs: impl IntoIterator<Item = Value>) -> Result<(), EvalPlanError> {
        let outputs: Vec<_> = outputs.into_iter().collect();
        if outputs.len() != self.expected.len() {
            return Err(EvalPlanError::OutputArity {
                expected: self.expected.len(),
                actual: outputs.len(),
            });
        }
        for (index, (value, expected)) in outputs.iter().zip(&self.expected).enumerate() {
            check_info(index, value, expected)?;
        }
        for (index, (trace, source)) in self.writes.iter().enumerate() {
            check_info(index, source.value(&outputs), trace.info())?;
        }
        let mut detached = Vec::new();
        {
            let _guard = EVAL_DAG.lock().unwrap();
            for (target, source) in &self.writes {
                detached.extend(target.publish(source.value(&outputs).clone()));
            }
        }
        // Dropping a producer may release concrete buffers; do that unlocked.
        drop(detached);
        Ok(())
    }
}

fn check_info(index: usize, value: &Value, expected: &ValueInfo) -> Result<(), EvalPlanError> {
    let actual = value.to_info();
    if actual != *expected {
        return Err(EvalPlanError::OutputInfo {
            index,
            expected: expected.clone(),
            actual,
        });
    }
    Ok(())
}

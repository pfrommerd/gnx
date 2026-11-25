use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::io;

use std::sync::Arc;

use gnx_expr::{ConcreteValue, Expr, ExprBuilder, Param, RootExpr, Value, Var};

use crate::buffer::TransferPolicy;

#[derive(Clone, Copy, Debug, Default)]
pub struct ExecOpts {
    pub transfers: TransferPolicy,
}

/// Executes a lowered expression. Arguments are captures followed by explicit
/// inputs; values are returned in `Expr::outputs()` order, including forwarded
/// parameters. This interface does not apply cell updates or reconstruct cell handles.
pub trait Device: Send + Sync {
    type Outputs: IntoIterator<Item = Value>;

    fn execute(
        &self,
        expr: Arc<RootExpr>,
        args: impl IntoIterator<Item = Value>,
        opts: ExecOpts,
    ) -> impl Future<Output = io::Result<Self::Outputs>> + Send;
}

/// Output routing for an execution. Forwarding never imports or transfers a
/// concrete parameter, and repeated results retain the same allocation.
#[derive(Clone, Debug)]
pub struct ExecutionBoundary {
    argument_indices: Vec<usize>,
    results: Vec<ResultSlot>,
    computed: usize,
}

#[derive(Clone, Copy, Debug)]
enum ResultSlot {
    Argument(usize),
    Computed(usize),
}

impl ExecutionBoundary {
    pub fn project(expr: Expr<'_>) -> io::Result<(Option<Arc<RootExpr>>, Self)> {
        expr.check()
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidInput, err))?;
        let parameters: HashMap<_, _> = expr
            .closure_inputs()
            .chain(expr.inputs())
            .enumerate()
            .map(|(i, param)| (param.var(), i))
            .collect();
        let mut outputs = Vec::new();
        let mut computed = HashMap::<Var, usize>::new();
        let results = expr
            .outputs()
            .iter()
            .map(|var| match parameters.get(var) {
                Some(&index) => ResultSlot::Argument(index),
                None => ResultSlot::Computed(*computed.entry(*var).or_insert_with(|| {
                    outputs.push(*var);
                    outputs.len() - 1
                })),
            })
            .collect();
        let computed = outputs.len();
        let mut argument_indices = Vec::new();
        let program = if computed == 0 {
            None
        } else {
            let mut used: HashSet<_> = outputs.iter().copied().collect();
            let mut eqns = Vec::new();
            for eqn in expr.eqns().rev() {
                if eqn.outputs().any(|output| used.contains(&output.var())) {
                    used.extend(eqn.inputs().iter().copied());
                    used.extend(eqn.subexprs().flat_map(|sub| sub.provided_vars()).copied());
                    eqns.push(eqn.to_source());
                }
            }
            eqns.reverse();
            let mut pick = |param: Param<'_>| {
                let var = param.var();
                (used.contains(&var) || !param.info().symbols().is_empty()).then(|| {
                    argument_indices.push(parameters[&var]);
                    (var, param.info_ref().to_owned())
                })
            };
            let closure = expr.closure_inputs().filter_map(&mut pick).collect();
            let inputs = expr.inputs().filter_map(&mut pick).collect();
            let mut program = ExprBuilder::from_parts(
                expr.closure_symbols().to_vec(),
                closure,
                inputs,
                eqns,
                outputs,
            );
            if let Some(name) = expr.name_hint() {
                program = program.with_name_hint(name.to_owned());
            }
            let program = program
                .build()
                .map_err(|err| io::Error::new(io::ErrorKind::InvalidInput, err))?;
            Some(program)
        };
        Ok((
            program,
            Self {
                argument_indices,
                results,
                computed,
            },
        ))
    }

    pub fn validate_arguments(expr: Expr<'_>, args: &[Value]) -> io::Result<()> {
        let count = expr.closure_inputs().len() + expr.inputs().len();
        if args.len() != count {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("expression expects {count} arguments, got {}", args.len()),
            ));
        }
        for (i, (param, value)) in (expr.closure_inputs().chain(expr.inputs()))
            .zip(args)
            .enumerate()
        {
            let expected = param.info_ref();
            let actual = value.to_info();
            if expected != actual {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("argument {i}: expected {expected}, got {actual}"),
                ));
            }
        }
        Ok(())
    }

    pub fn arguments<'a>(
        &'a self,
        args: &'a [Value],
    ) -> impl Iterator<Item = io::Result<&'a Value>> {
        self.argument_indices.iter().map(|&i| {
            args.get(i).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "missing computation argument")
            })
        })
    }

    pub fn finish(&self, args: &[Value], computed: &[Value]) -> io::Result<Vec<Value>> {
        if computed.len() != self.computed {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unexpected number of computed results",
            ));
        }
        self.results
            .iter()
            .map(|slot| match *slot {
                ResultSlot::Argument(i) => args.get(i).cloned().ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "missing forwarded argument")
                }),
                ResultSlot::Computed(i) => Ok(computed[i].clone()),
            })
            .collect()
    }
}

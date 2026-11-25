//! Expr-to-Expr passes and the reusable machinery to write them.
//!
//! A [`Pass`] transforms a lowered expression without changing its closure,
//! input, or output signature. A [`Pipeline`] runs passes in order and
//! verifies the result of each. [`Builder`] and [`rewrite`] are the SSA
//! construction kit shared by passes and by the rules they dispatch through
//! op interfaces.

mod builder;
mod dce;
mod rewrite;

pub use builder::Builder;
pub use dce::{Dce, dce};
pub use rewrite::rewrite;

use std::fmt;
use std::marker::PhantomData;
use std::sync::{Arc, LazyLock};

use crate::{Expr, RootExpr, Symbol, VarInfo};

pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// A stage of a [`Pipeline`]. `Pass<Arc<RootExpr>>` (the default) is an
/// Expr-to-Expr transformation that must preserve the closure, input, and
/// output metadata. A pass producing another `T` (HLO, MLIR, a wire
/// program, ...) can only be the final stage, installed with
/// [`Pipeline::finalize`].
pub trait Pass<T = Arc<RootExpr>>: Send + Sync {
    fn name(&self) -> &'static str;
    fn run(&self, expr: Arc<RootExpr>) -> Result<T, PassError>;
}

/// Any `Fn(Arc<RootExpr>) -> Result<T, _>` can be a final stage.
pub struct FnPass<F> {
    name: &'static str,
    f: F,
}

impl<F> FnPass<F> {
    pub fn new(name: &'static str, f: F) -> Self {
        Self { name, f }
    }
}

impl<T, E, F> Pass<T> for FnPass<F>
where
    F: Fn(Arc<RootExpr>) -> Result<T, E> + Send + Sync,
    E: Into<BoxError>,
{
    fn name(&self) -> &'static str {
        self.name
    }
    fn run(&self, expr: Arc<RootExpr>) -> Result<T, PassError> {
        (self.f)(expr).map_err(|error| PassError::new(self.name, error))
    }
}

#[derive(Debug, thiserror::Error)]
#[error("pass `{pass}`: {source}")]
pub struct PassError {
    pub pass: &'static str,
    #[source]
    pub source: BoxError,
}

impl PassError {
    pub fn new(pass: &'static str, source: impl Into<BoxError>) -> Self {
        Self {
            pass,
            source: source.into(),
        }
    }
}

/// An ordered list of Expr-to-Expr passes, optionally followed by one final
/// stage producing a `T`:
///
/// ```ignore
/// let pipeline: Pipeline<XlaComputation> = Pipeline::new()
///     .with(ExpandTransforms)
///     .with(Dce)
///     .finalize(LowerXla);
/// ```
///
/// After each Expr-to-Expr pass the result is checked with [`Expr::check`] and
/// must keep the original signature. Set `GNX_DUMP_IR` to a comma-separated
/// list of pass names (or `all`) to print the expression after those passes.
pub struct Pipeline<T = Arc<RootExpr>> {
    passes: Vec<Box<dyn Pass>>,
    last: Box<dyn Pass<T>>,
}

struct Identity;
impl Pass for Identity {
    fn name(&self) -> &'static str {
        "identity"
    }
    fn run(&self, expr: Arc<RootExpr>) -> Result<Arc<RootExpr>, PassError> {
        Ok(expr)
    }
}

impl Default for Pipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl Pipeline {
    pub fn new() -> Self {
        Self {
            passes: Vec::new(),
            last: Box::new(Identity),
        }
    }

    pub fn with(mut self, pass: impl Pass + 'static) -> Self {
        self.passes.push(Box::new(pass));
        self
    }

    /// Append every pass of another unfinalized pipeline.
    pub fn with_all(mut self, other: Pipeline) -> Self {
        self.passes.extend(other.passes);
        self
    }

    /// Install the final stage, which may produce something other than an Expr.
    pub fn finalize<T>(self, last: impl Pass<T> + 'static) -> Pipeline<T> {
        Pipeline {
            passes: self.passes,
            last: Box::new(last),
        }
    }
}

impl<T> Pipeline<T> {
    /// The Expr-to-Expr passes, excluding the final stage.
    pub fn passes(&self) -> impl ExactSizeIterator<Item = &dyn Pass> {
        self.passes.iter().map(|pass| pass.as_ref())
    }

    /// Run the Expr-to-Expr passes only, skipping the final stage.
    pub fn optimize(&self, mut expr: Arc<RootExpr>) -> Result<Arc<RootExpr>, PassError> {
        let signature = Signature::of(expr.view());
        for pass in &self.passes {
            let name = pass.name();
            expr = pass.run(expr)?;
            (expr.view().check()).map_err(|error| PassError::new(name, error))?;
            let actual = Signature::of(expr.view());
            if actual != signature {
                return Err(PassError::new(
                    name,
                    format!("changed the expression signature from {signature} to {actual}"),
                ));
            }
            if dump_after(name) {
                eprintln!("// gnx: after pass `{name}`\n{}", expr.view());
            }
        }
        Ok(expr)
    }

    pub fn run(&self, expr: Arc<RootExpr>) -> Result<T, PassError> {
        self.last.run(self.optimize(expr)?)
    }
}

fn dump_after(pass: &str) -> bool {
    static DUMP: LazyLock<Vec<String>> = LazyLock::new(|| {
        std::env::var("GNX_DUMP_IR")
            .map(|value| {
                value
                    .split(',')
                    .map(|name| name.trim().to_owned())
                    .collect()
            })
            .unwrap_or_default()
    });
    DUMP.iter().any(|name| name == "all" || name == pass)
}

#[derive(PartialEq, Eq)]
struct Signature {
    closure_symbols: Vec<Symbol>,
    closure: Vec<VarInfo>,
    inputs: Vec<VarInfo>,
    outputs: Vec<Option<VarInfo>>,
}

impl Signature {
    fn of(expr: Expr<'_>) -> Self {
        Self {
            closure_symbols: expr.closure_symbols().to_vec(),
            closure: (expr.closure_inputs())
                .map(|param| param.info_ref().to_owned())
                .collect(),
            inputs: (expr.inputs())
                .map(|param| param.info_ref().to_owned())
                .collect(),
            outputs: (expr.outputs().iter())
                .map(|var| expr.value_info(*var).map(|info| info.to_owned()))
                .collect(),
        }
    }
}

impl fmt::Display for Signature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let list = |f: &mut fmt::Formatter<'_>, infos: &mut dyn Iterator<Item = String>| {
            write!(f, "(")?;
            for (i, info) in infos.enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{info}")?;
            }
            write!(f, ")")
        };
        list(f, &mut self.closure.iter().map(ToString::to_string))?;
        list(f, &mut self.inputs.iter().map(ToString::to_string))?;
        write!(f, " -> ")?;
        list(
            f,
            &mut self.outputs.iter().map(|info| {
                info.as_ref()
                    .map_or_else(|| "?".to_owned(), ToString::to_string)
            }),
        )
    }
}

/// Fail if any operation, including those in nested regions, lacks an
/// implementation of interface `I`. Use before a backend walks the expression.
pub struct CheckSupported<I: ?Sized + 'static>(PhantomData<fn() -> &'static I>);

impl<I: ?Sized + 'static> CheckSupported<I> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<I: ?Sized + 'static> Default for CheckSupported<I> {
    fn default() -> Self {
        Self::new()
    }
}

impl<I: ?Sized + 'static> Pass for CheckSupported<I> {
    fn name(&self) -> &'static str {
        "check_supported"
    }

    fn run(&self, expr: Arc<RootExpr>) -> Result<Arc<RootExpr>, PassError> {
        fn check<I: ?Sized + 'static>(expr: Expr<'_>) -> Result<(), String> {
            for (index, eqn) in expr.eqns().enumerate() {
                (eqn.op().get())
                    .interface::<I>()
                    .map_err(|error| format!("equation {index}: {error}"))?;
                for (region, sub) in eqn.subexprs().enumerate() {
                    check::<I>(sub.expr()).map_err(|error| {
                        format!("equation {index} ({}) region {region}: {error}", eqn.op())
                    })?;
                }
            }
            Ok(())
        }
        check::<I>(expr.view()).map_err(|error| PassError::new(self.name(), error))?;
        Ok(expr)
    }
}

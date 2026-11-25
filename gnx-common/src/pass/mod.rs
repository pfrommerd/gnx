//! Expr-to-Expr passes and the reusable machinery to write them.
//!
//! A [`Pass`] transforms a lowered expression without changing its closure,
//! input, or output signature. A [`Pipeline`] runs passes in order and
//! verifies the result of each, keeping every result in a
//! [`Cache`](crate::cache::Cache) so that work already done is not repeated.
//! [`Builder`] and [`rewrite`] are the SSA construction kit shared by passes
//! and by the rules they dispatch through op interfaces.

mod builder;
mod dce;
mod rewrite;

pub use builder::Builder;
pub use dce::{Dce, dce};
pub use rewrite::rewrite;

use std::fmt;
use std::marker::PhantomData;
use std::sync::{Arc, LazyLock};

use anyhash::HasherWrite;
use gnx_expr::{Blake3Hash, Expr, RootExpr, Symbol, VarInfo};
use rkyv::api::high::HighSerializer;
use rkyv::ser::allocator::ArenaHandle;
use rkyv::util::AlignedVec;
use rkyv::{Archive, Portable, Serialize, rancor};

use crate::cache::{Blob, Cache, CacheError, StableHasher};

pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// [`anyhash::Hash`], for use through a trait object. Implemented by
/// everything that implements `anyhash::Hash`.
pub trait DynHash {
    fn dyn_hash(&self, state: &mut dyn HasherWrite);
}

impl<T: anyhash::Hash + ?Sized> DynHash for T {
    fn dyn_hash(&self, state: &mut dyn HasherWrite) {
        self.hash(&mut DynWriter(state));
    }
}

struct DynWriter<'a>(&'a mut dyn HasherWrite);

macro_rules! forward_writes {
    ($($method:ident: $ty:ty),* $(,)?) => {
        $(fn $method(&mut self, value: $ty) {
            self.0.$method(value);
        })*
    };
}

impl HasherWrite for DynWriter<'_> {
    forward_writes! {
        write: &[u8], write_str: &str, write_length_prefix: usize,
        write_u8: u8, write_u16: u16, write_u32: u32, write_u64: u64, write_u128: u128,
        write_usize: usize,
        write_i8: i8, write_i16: i16, write_i32: i32, write_i64: i64, write_i128: i128,
        write_isize: isize,
    }
}

/// A stage of a [`Pipeline`]. `Pass<Arc<RootExpr>>` (the default) is an
/// Expr-to-Expr transformation that must preserve the closure, input, and
/// output metadata. A pass producing another `T` (HLO, a serialized module,
/// a wire program, ...) can only be the final stage, installed with
/// [`Pipeline::finalize`].
///
/// A pass is a function of the expression and of what it hashes
/// (`anyhash::Hash`): its results are cached under its name, its hash, and
/// the passes before it. Hash everything that configures the pass.
pub trait Pass<T = Arc<RootExpr>>: DynHash + Send + Sync {
    fn name(&self) -> &'static str;
    fn run(&self, expr: Arc<RootExpr>) -> Result<T, PassError>;
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

/// Why a pipeline did not produce a result.
#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error(transparent)]
    Pass(#[from] PassError),
    #[error(transparent)]
    Cache(#[from] CacheError),
}

/// What the final stage of a pipeline can produce: anything that can be
/// archived, so that it can be cached.
pub trait Artifact:
    Archive<Archived: Portable>
    + for<'a> Serialize<HighSerializer<AlignedVec, ArenaHandle<'a>, rancor::Error>>
    + 'static
{
}

impl<T> Artifact for T where
    T: Archive<Archived: Portable>
        + for<'a> Serialize<HighSerializer<AlignedVec, ArenaHandle<'a>, rancor::Error>>
        + 'static
{
}

/// An ordered list of Expr-to-Expr passes, optionally followed by one final
/// stage producing something else (see [`Pipeline::finalize`]):
///
/// ```ignore
/// let pipeline = Pipeline::new()
///     .with(ExpandTransforms)
///     .with(Dce)
///     .finalize(LowerXla);
/// let program = pipeline.run(expr, &cache).await?;
/// ```
///
/// After each Expr-to-Expr pass the result is checked with `Expr::check` and
/// must keep the original signature. The result of every stage is cached,
/// under the expression the pipeline was given and the passes up to that
/// stage: a later run on the same expression starts after the last stage
/// whose result is cached. Set `GNX_DUMP_IR` to a comma-separated list of
/// pass names (or `all`) to print the expression after those passes.
#[derive(Default)]
pub struct Pipeline {
    stages: Vec<Stage<Arc<RootExpr>>>,
}

struct Stage<T> {
    pass: Box<dyn Pass<T>>,
    // Identifies the passes up to and including this one.
    prefix: Blake3Hash,
}

impl<T> Stage<T> {
    fn new(before: Option<Blake3Hash>, pass: Box<dyn Pass<T>>) -> Self {
        let mut hasher = StableHasher::new();
        if let Some(before) = before {
            hasher.write(before.as_bytes());
        }
        hasher.write_str(pass.name());
        pass.dyn_hash(&mut hasher);
        Self {
            pass,
            prefix: hasher.finish(),
        }
    }

    // What the result of this stage on the expression `input` is bound to.
    fn key(&self, input: Blake3Hash) -> ([u8; 32], [u8; 32]) {
        (*input.as_bytes(), *self.prefix.as_bytes())
    }
}

impl Pipeline {
    pub fn new() -> Self {
        Self::default()
    }

    fn prefix(&self) -> Option<Blake3Hash> {
        self.stages.last().map(|stage| stage.prefix)
    }

    pub fn with(mut self, pass: impl Pass + 'static) -> Self {
        (self.stages).push(Stage::new(self.prefix(), Box::new(pass)));
        self
    }

    /// Install the final stage, which may produce something other than an Expr.
    pub fn finalize<T: Artifact>(self, last: impl Pass<T> + 'static) -> FinalPipeline<T> {
        FinalPipeline {
            last: Stage::new(self.prefix(), Box::new(last)),
            exprs: self,
        }
    }

    pub fn passes(&self) -> impl ExactSizeIterator<Item = &dyn Pass> {
        self.stages.iter().map(|stage| stage.pass.as_ref())
    }

    /// Run the passes on `expr`, starting after the last one whose result
    /// is in `cache`.
    pub async fn run(
        &self,
        expr: Arc<RootExpr>,
        cache: &Cache,
    ) -> Result<Arc<RootExpr>, PipelineError> {
        let input = expr.hash();
        let signature = Signature::of(expr.view());
        let mut expr = expr;
        let mut start = 0;
        for (index, stage) in self.stages.iter().enumerate().rev() {
            if let Some(key) = cache.resolve::<RootExpr>(&stage.key(input)).await? {
                expr = cache.get_root(key).await?;
                start = index + 1;
                if dump_after(stage.pass.name()) {
                    let name = stage.pass.name();
                    eprintln!("// gnx: after pass `{name}` (cached)\n{}", expr.view());
                }
                break;
            }
        }
        for stage in &self.stages[start..] {
            let name = stage.pass.name();
            expr = stage.pass.run(expr)?;
            (expr.view().check()).map_err(|error| PassError::new(name, error))?;
            let actual = Signature::of(expr.view());
            if actual != signature {
                return Err(PassError::new(
                    name,
                    format!("changed the expression signature from {signature} to {actual}"),
                )
                .into());
            }
            let key = cache.put_root(&expr).await?;
            cache.bind(&stage.key(input), key).await?;
            if dump_after(name) {
                eprintln!("// gnx: after pass `{name}`\n{}", expr.view());
            }
        }
        Ok(expr)
    }
}

/// A [`Pipeline`] followed by a stage that produces a `T`.
pub struct FinalPipeline<T: Artifact> {
    exprs: Pipeline,
    last: Stage<T>,
}

impl<T: Artifact> FinalPipeline<T> {
    /// The Expr-to-Expr passes, excluding the final stage.
    pub fn passes(&self) -> impl ExactSizeIterator<Item = &dyn Pass> {
        self.exprs.passes()
    }

    /// Run the Expr-to-Expr passes only, skipping the final stage.
    pub async fn optimize(
        &self,
        expr: Arc<RootExpr>,
        cache: &Cache,
    ) -> Result<Arc<RootExpr>, PipelineError> {
        self.exprs.run(expr, cache).await
    }

    /// The result of the final stage, in its archived form: read from
    /// `cache` if the pipeline has run on `expr` before.
    pub async fn run(&self, expr: Arc<RootExpr>, cache: &Cache) -> Result<Blob<T>, PipelineError> {
        let key = self.last.key(expr.hash());
        if let Some(cached) = cache.resolve::<T>(&key).await? {
            return Ok(cache.get(cached).await?);
        }
        let expr = self.exprs.run(expr, cache).await?;
        let result = cache.put(&self.last.pass.run(expr)?).await?;
        cache.bind(&key, result.key()).await?;
        Ok(result)
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

impl<I: ?Sized + 'static> anyhash::Hash for CheckSupported<I> {
    fn hash<H: HasherWrite>(&self, state: &mut H) {
        state.write_str(std::any::type_name::<I>());
    }
}

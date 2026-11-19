//! Pure SSA annotations for mapping array dimensions onto named mesh axes.

use std::collections::HashSet;
use std::fmt;

use gnx_expr::{Dialect, Eqn, Invocation, LoweringError, Op, Operation};

use gnx_array::{Array, ArrayInfo};

pub struct ShardDialect;
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Constraint {
    spec: Sharding,
}
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Replicate;

static DIALECT: ShardDialect = ShardDialect;

type Result<T> = std::result::Result<T, LoweringError>;

impl Dialect for ShardDialect {
    fn name(&self) -> &'static str {
        "shard"
    }
}

/// A dimension maps to zero or more mesh axes. An empty list leaves that
/// dimension unpartitioned. Each mesh axis may appear at most once.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Sharding(Vec<Vec<String>>);

pub fn valid_axis(name: &str) -> bool {
    !name.is_empty() && name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
}

impl Sharding {
    pub fn new(axes: Vec<Vec<String>>) -> Result<Self> {
        let mut seen = HashSet::new();
        if axes
            .iter()
            .flatten()
            .any(|axis| !valid_axis(axis) || !seen.insert(axis))
        {
            return Err(LoweringError::RegionSignature(
                "invalid or repeated sharding axis",
            ));
        }
        Ok(Self(axes))
    }

    pub fn axes(&self) -> &[Vec<String>] {
        &self.0
    }
    pub fn is_replicated(&self) -> bool {
        self.0.iter().all(Vec::is_empty)
    }
    pub fn replicated(rank: usize) -> Self {
        Self(vec![vec![]; rank])
    }

    pub fn from_op(op: &Op, rank: usize) -> Result<Self> {
        let spec = if op.downcast_ref::<Replicate>().is_some() {
            Self::replicated(rank)
        } else if let Some(constraint) = op.downcast_ref::<Constraint>() {
            constraint.spec.clone()
        } else {
            return Err(signature());
        };
        if spec.axes().len() != rank {
            return Err(signature());
        }
        Ok(spec)
    }
}

fn signature() -> LoweringError {
    LoweringError::RegionSignature("invalid sharding annotation signature")
}

fn check(eqn: &Eqn) -> Result<()> {
    let [(_, info)] = eqn.outputs() else {
        return Err(signature());
    };
    let info = info.downcast::<ArrayInfo>().map_err(|_| signature())?;
    if eqn.inputs().len() != 1 || !eqn.subexprs().is_empty() {
        return Err(signature());
    }
    Sharding::from_op(eqn.op(), info.shape().dims().len())?;
    Ok(())
}

impl fmt::Display for Constraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "shard::constraint<axes: {:?}>", self.spec.axes())
    }
}

impl Operation for Constraint {
    fn name(&self) -> &'static str {
        "constraint"
    }
    fn dialect(&self) -> &'static dyn Dialect {
        &DIALECT
    }
    fn check_expr(&self, eqn: &Eqn) -> Result<()> {
        check(eqn)
    }
}

impl fmt::Display for Replicate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("shard::replicate")
    }
}

impl Operation for Replicate {
    fn name(&self) -> &'static str {
        "replicate"
    }
    fn dialect(&self) -> &'static dyn Dialect {
        &DIALECT
    }
    fn check_expr(&self, eqn: &Eqn) -> Result<()> {
        check(eqn)
    }
}

pub fn constraint_op(spec: Sharding) -> Op {
    Op::new(Constraint { spec })
}

pub fn replicate_op() -> Op {
    Op::new(Replicate)
}

pub fn constrain(array: &Array, spec: Sharding) -> Result<Array> {
    if spec.axes().len() != array.shape().dims().len() {
        return Err(signature());
    }
    Ok(annotate(array, constraint_op(spec)))
}

pub fn replicate(array: &Array) -> Array {
    annotate(array, replicate_op())
}

fn annotate(array: &Array, op: Op) -> Array {
    let tracer = array.tracer();
    Invocation::invoke(
        op,
        vec![],
        vec![tracer.clone()],
        vec![tracer.trace_ref().info().clone()],
    )
    .pop()
    .expect("one annotation result")
    .try_cast::<Array>()
    .ok()
    .expect("array result")
    .into()
}

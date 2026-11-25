//! Pure SSA annotations for mapping array dimensions onto named mesh axes.

use std::collections::HashSet;
use std::fmt;

use gnx_expr::{Eqn, Invocation, LoweringError, Op, OpRef, Operation};

use gnx_common::{Array, ArrayInfo};

/// Builds a `shard::constraint`, stored as a [`Constraint`].
#[derive(rkyv::Archive, rkyv::Serialize)]
#[rkyv(archived = Constraint)]
pub struct ConstraintBuilder {
    axes: Vec<Vec<String>>,
}

impl Constraint {
    /// The mesh axes of every dimension.
    pub fn spec(&self) -> Sharding {
        Sharding(
            (self.axes.iter())
                .map(|axes| axes.iter().map(|axis| axis.as_str().to_owned()).collect())
                .collect(),
        )
    }
}

gnx_common::unit_operations! {
    pub struct Replicate;
}

gnx_expr::dialect! {
    dyn Operation, "shard" => { "constraint" => Constraint, "replicate" => Replicate }
}

type Result<T> = std::result::Result<T, LoweringError>;

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
        Self::from_operation(op.get(), rank)
    }

    /// Like [`Sharding::from_op`], for an operation read from an expression.
    pub fn from_op_ref(op: OpRef<'_>, rank: usize) -> Result<Self> {
        Self::from_operation(op.get(), rank)
    }

    fn from_operation(op: &dyn Operation, rank: usize) -> Result<Self> {
        let spec = if op.downcast_ref::<Replicate>().is_some() {
            Self::replicated(rank)
        } else if let Some(constraint) = op.downcast_ref::<Constraint>() {
            constraint.spec()
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

fn check(eqn: Eqn<'_>) -> Result<()> {
    let mut outputs = eqn.outputs();
    let (Some(output), None) = (outputs.next(), outputs.next()) else {
        return Err(signature());
    };
    let info = ArrayInfo::from_info(&output.info_ref().to_owned()).ok_or_else(signature)?;
    if eqn.inputs().len() != 1 || eqn.subexprs().len() != 0 {
        return Err(signature());
    }
    Sharding::from_op_ref(eqn.op(), info.shape().dims().len())?;
    Ok(())
}

impl Operation for Constraint {
    fn display(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "shard::constraint<axes: {:?}>", self.spec().axes())
    }
    fn check_expr(&self, eqn: Eqn<'_>) -> Result<()> {
        check(eqn)
    }
}

impl Operation for Replicate {
    fn display(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("shard::replicate")
    }
    fn check_expr(&self, eqn: Eqn<'_>) -> Result<()> {
        check(eqn)
    }
}

pub fn constraint_op(spec: Sharding) -> Op {
    Op::new(ConstraintBuilder { axes: spec.0 })
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

use super::{Dialect, Eqn, Op, Operation, Result, arity, fmt, result_rank, signature};

pub use super::{eye_op as eye, ones_op as ones, zeros_op as zeros};

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct IotaOperation {
    axis: usize,
}
impl IotaOperation {
    pub fn axis(&self) -> usize {
        self.axis
    }
}
gnx_common::array_operation!(
    IotaOperation,
    "iota",
    |op, eqn| {
        arity(eqn, 0)?;
        if op.axis >= result_rank(eqn)? {
            return Err(signature("iota axis is out of range"));
        }
        Ok(())
    },
    axis
);
pub fn iota(axis: usize) -> Op {
    Op::new(IotaOperation { axis })
}

simple_operations! {
    FullOperation, full_op, full, 1;
}
pub use zeros as empty;

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct ConstantOperation {
    bits: u64,
}
impl ConstantOperation {
    pub fn value(&self) -> f64 {
        f64::from_bits(self.bits)
    }
}
gnx_common::array_operation!(
    ConstantOperation,
    "constant",
    |_op, eqn| {
        arity(eqn, 0)?;
        if result_rank(eqn)? != 0 {
            return Err(signature("constant must produce a scalar"));
        }
        Ok(())
    },
    bits
);
/// A floating-point scalar, converted to the result's dtype.
pub fn constant(value: f64) -> Op {
    Op::new(ConstantOperation {
        bits: value.to_bits(),
    })
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct ScalarBoundOperation {
    lower: bool,
}
impl ScalarBoundOperation {
    pub fn lower(&self) -> bool {
        self.lower
    }
}
gnx_common::array_operation!(
    ScalarBoundOperation,
    "scalar_bound",
    |_op, eqn| {
        arity(eqn, 0)?;
        if result_rank(eqn)? != 0 {
            return Err(signature("scalar bound must produce a scalar"));
        }
        Ok(())
    },
    lower
);
pub fn scalar_bound(lower: bool) -> Op {
    Op::new(ScalarBoundOperation { lower })
}

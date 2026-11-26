use super::{Op, arity, count, result_rank, signature};

pub use super::{eye_op as eye, ones_op as ones, zeros_op as zeros};

#[derive(rkyv::Archive, rkyv::Serialize)]
#[rkyv(archived = IotaOperation)]
pub struct Iota {
    axis: u64,
}
impl IotaOperation {
    pub fn axis(&self) -> usize {
        count(self.axis)
    }
}
gnx_base::array_operation!(
    IotaOperation,
    "iota",
    |op, eqn| {
        arity(eqn, 0)?;
        if op.axis() >= result_rank(eqn)? {
            return Err(signature("iota axis is out of range"));
        }
        Ok(())
    },
    axis
);
pub fn iota(axis: usize) -> Op {
    Op::new(Iota { axis: axis as u64 })
}

simple_operations! {
    FullOperation, full_op, full, "full", 1;
}
pub use zeros as empty;

#[derive(rkyv::Archive, rkyv::Serialize)]
#[rkyv(archived = ConstantOperation)]
pub struct Constant {
    bits: u64,
}
impl ConstantOperation {
    pub fn value(&self) -> f64 {
        f64::from_bits(self.bits())
    }
    fn bits(&self) -> u64 {
        self.bits.to_native()
    }
}
gnx_base::array_operation!(
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
    Op::new(Constant {
        bits: value.to_bits(),
    })
}

#[derive(rkyv::Archive, rkyv::Serialize)]
#[rkyv(archived = ScalarBoundOperation)]
pub struct ScalarBound {
    lower: bool,
}
impl ScalarBoundOperation {
    pub fn lower(&self) -> bool {
        self.lower
    }
}
gnx_base::array_operation!(
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
    Op::new(ScalarBound { lower })
}

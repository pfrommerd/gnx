use std::fmt;

use gnx_common::Array;
use gnx_expr::{Eqn, Invocation, LoweringError, Op, Operation};

gnx_common::unit_operations! {
    pub struct StopGradientOperation;
}

impl Operation for StopGradientOperation {
    fn display(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ad::stop_gradient")
    }

    fn check_expr(&self, eqn: Eqn<'_>) -> Result<(), LoweringError> {
        if eqn.inputs().len() != 1 || eqn.outputs().len() != 1 || eqn.subexprs().len() != 0 {
            return Err(LoweringError::RegionSignature("stop_gradient signature"));
        }
        Ok(())
    }
}

pub fn stop_gradient_op() -> Op {
    Op::new(StopGradientOperation)
}

/// Return the input array with its contribution to differentiation suppressed.
pub fn stop_gradient(array: &Array) -> Array {
    let tracer = array.tracer();
    Invocation::invoke(
        stop_gradient_op(),
        vec![],
        vec![tracer.clone()],
        vec![tracer.trace_ref().info().clone()],
    )
    .pop()
    .expect("stop_gradient has one result")
    .try_cast::<Array>()
    .ok()
    .expect("stop_gradient produces an array")
    .into()
}

crate::defjvp!(StopGradientOperation, None);
crate::defbroadcasting!(StopGradientOperation);

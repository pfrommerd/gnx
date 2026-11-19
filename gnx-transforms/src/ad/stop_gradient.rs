use std::fmt;

use gnx_array::Array;
use gnx_expr::{Dialect, Eqn, Invocation, LoweringError, Op, Operation};

use super::AdDialect;

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct StopGradientOperation;

impl fmt::Display for StopGradientOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ad::stop_gradient")
    }
}

impl Operation for StopGradientOperation {
    fn name(&self) -> &'static str {
        "stop_gradient"
    }

    fn dialect(&self) -> &'static dyn Dialect {
        &AdDialect
    }

    fn check_expr(&self, eqn: &Eqn) -> Result<(), LoweringError> {
        if eqn.inputs().len() != 1 || eqn.outputs().len() != 1 || !eqn.subexprs().is_empty() {
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

//! Automatic differentiation.
//!
//! Builders wrap a function with JAX's defaults; builder methods override them,
//! with `wrt` specs given as [`Dag`] graphs over the arguments:
//!
//! ```ignore
//! let df = grad(|x: Array, y: Array| f(x, y)).wrt(dag!(true, false));
//! let (y, ty) = jvp(f).invoke((primals, tangents));
//! let (y, dx) = vjp(f).invoke((primals, cotangents));
//! let h = hessian(f);
//! ```
//!
//! Calls trace the function once per input signature (through `jit`) and
//! record region operations:
//!
//! | op | operands | results | parameters |
//! |---|---|---|---|
//! | `ad.jvp` | primals ++ tangents of `wrt` | results ++ tangents of `of` | `wrt`, `of` |
//! | `ad.vjp` | primals ++ cotangents of `of` | results ++ cotangents of `wrt` | `wrt`, `of` |
//!
//! Non-differentiated values are captures of the body. Backends with native
//! support (JAX) lower them directly; others run
//! [`ExpandTransforms`](crate::ExpandTransforms), which uses [`JvpRule`]s for
//! forward mode and, for reverse mode, linearizes with [`JvpRule`]s and
//! transposes the linear part with [`TransposeRule`]s.
//!
//! [`Dag`]: gnx_graph::util::Dag

mod api;
mod stop_gradient;
pub use stop_gradient::{StopGradientOperation, stop_gradient, stop_gradient_op};
pub mod def;
pub mod jvp;
pub mod region;
pub(crate) mod rules;
pub mod transpose;

pub use api::{Grad, Hessian, Jvp, ValueAndGrad, Vjp, grad, hessian, jvp, value_and_grad, vjp};
pub use jvp::{Dual, JvpRule, Tangent};
pub use transpose::{Operand, TransposeRule};

use std::fmt;

use gnx_expr::lower::LoweringContext;
use gnx_expr::pass::Builder;
use gnx_expr::{Dialect, Eqn, HEqn, LoweringError, Op, Operation, Var, register_op_interface};

use crate::body::{
    BodySignature, check_expr_with, lower_transform, signature, strictly_increasing,
};
use crate::expand::Transform;
use crate::{TransformResult, util};

pub struct AdDialect;
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct JvpOperation {
    parameters: DiffParameters,
}

impl JvpOperation {
    pub fn parameters(&self) -> &DiffParameters {
        &self.parameters
    }
}
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct VjpOperation {
    parameters: DiffParameters,
}

impl VjpOperation {
    pub fn parameters(&self) -> &DiffParameters {
        &self.parameters
    }
}

static AD: AdDialect = AdDialect;

impl Dialect for AdDialect {
    fn name(&self) -> &'static str {
        "ad"
    }
}

/// Parameters of `ad.jvp` and `ad.vjp`: differentiate the body's results
/// `of` with respect to its inputs `wrt` (both strictly increasing).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DiffParameters {
    pub wrt: Vec<usize>,
    pub of: Vec<usize>,
}

impl DiffParameters {
    pub fn from_op(op: &Op) -> Result<Self, LoweringError> {
        if let Some(op) = op.downcast_ref::<JvpOperation>() {
            Ok(op.parameters().clone())
        } else if let Some(op) = op.downcast_ref::<VjpOperation>() {
            Ok(op.parameters().clone())
        } else {
            Err(signature("expected a differentiation operation"))
        }
    }

    pub fn jvp_op(&self) -> Op {
        Op::new(JvpOperation {
            parameters: self.clone(),
        })
    }

    pub fn vjp_op(&self) -> Op {
        Op::new(VjpOperation {
            parameters: self.clone(),
        })
    }

    /// Check the operand (if known) and result metadata against the body.
    /// `forward` selects the jvp layout; otherwise the vjp layout.
    pub fn check(
        &self,
        body: &BodySignature,
        operands: Option<&[gnx_expr::ValueInfo]>,
        operand_count: usize,
        outputs: &[gnx_expr::ValueInfo],
        forward: bool,
    ) -> Result<(), LoweringError> {
        strictly_increasing(&self.wrt, body.inputs.len())?;
        strictly_increasing(&self.of, body.outputs.len())?;
        let wrt: Vec<_> = self.wrt.iter().map(|&i| body.inputs[i].clone()).collect();
        let of: Vec<_> = self.of.iter().map(|&i| body.outputs[i].clone()).collect();
        if wrt.iter().chain(&of).any(|info| !util::is_inexact(info)) {
            return Err(signature(
                "differentiated inputs and results must be floating-point arrays",
            ));
        }
        let (extra_inputs, extra_outputs) = if forward { (wrt, of) } else { (of, wrt) };
        let expected_inputs: Vec<_> = body.inputs.iter().cloned().chain(extra_inputs).collect();
        let expected_outputs: Vec<_> = body.outputs.iter().cloned().chain(extra_outputs).collect();
        if operand_count != expected_inputs.len()
            || operands.is_some_and(|operands| operands != expected_inputs)
            || outputs != expected_outputs
        {
            return Err(signature("differentiation signature"));
        }
        Ok(())
    }
}

macro_rules! diff_operation {
    ($ty:ty, $name:literal, $forward:expr, $expand:path) => {
        impl fmt::Display for $ty {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(
                    f,
                    concat!("ad::", $name, "<of: {:?}, wrt: {:?}>"),
                    self.parameters.of, self.parameters.wrt
                )
            }
        }

        impl Operation for $ty {
            fn name(&self) -> &'static str {
                $name
            }
            fn dialect(&self) -> &'static dyn Dialect {
                &AD
            }
            fn lower(
                &self,
                eqn: &HEqn,
                ctx: &mut LoweringContext<'_>,
            ) -> Result<(), LoweringError> {
                lower_transform(eqn, ctx, |body, operands, outputs| {
                    self.parameters
                        .check(body, Some(operands), operands.len(), outputs, $forward)
                })
            }
            fn check_expr(&self, eqn: &Eqn) -> Result<(), LoweringError> {
                check_expr_with(eqn, |body, count, outputs| {
                    self.parameters.check(body, None, count, outputs, $forward)
                })
            }
        }

        impl Transform for $ty {
            fn expand(&self, b: &mut Builder, eqn: &Eqn) -> TransformResult<Vec<Var>> {
                $expand(b, eqn, self.parameters())
            }
        }
    };
}
diff_operation!(JvpOperation, "jvp", true, jvp::expand_jvp);
diff_operation!(VjpOperation, "vjp", false, transpose::expand_vjp);
register_op_interface!(dyn Transform: JvpOperation, VjpOperation);

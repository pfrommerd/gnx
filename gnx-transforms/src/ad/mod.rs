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

use gnx_common::ops::{indices, stored_indices};
use gnx_expr::lower::LoweringContext;
use gnx_common::pass::Builder;
use gnx_expr::{
    Eqn, EqnSource, HEqn, LoweringError, Op, OpRef, Operation, Var, register_op_interface,
};
use rkyv::{Archive, Serialize};

use crate::expand::Transform;
use crate::{TransformResult, util};
use gnx_common::ops::{signature, strictly_increasing};

/// Builds an `ad::jvp`, stored as a [`JvpOperation`].
#[derive(Archive, Serialize)]
#[rkyv(archived = JvpOperation)]
pub struct JvpBuilder {
    wrt: Vec<u32>,
    of: Vec<u32>,
}

/// Builds an `ad::vjp`, stored as a [`VjpOperation`].
#[derive(Archive, Serialize)]
#[rkyv(archived = VjpOperation)]
pub struct VjpBuilder {
    wrt: Vec<u32>,
    of: Vec<u32>,
}

impl JvpOperation {
    pub fn parameters(&self) -> DiffParameters {
        DiffParameters {
            wrt: indices(&self.wrt),
            of: indices(&self.of),
        }
    }
}

impl VjpOperation {
    pub fn parameters(&self) -> DiffParameters {
        DiffParameters {
            wrt: indices(&self.wrt),
            of: indices(&self.of),
        }
    }
}

gnx_expr::dialect! {
    dyn Operation, "ad" => {
        "jvp" => JvpOperation,
        "vjp" => VjpOperation,
        "stop_gradient" => StopGradientOperation,
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
        Self::from_operation(op.get())
    }

    /// Like [`DiffParameters::from_op`], for an operation read from an expression.
    pub fn from_op_ref(op: OpRef<'_>) -> Result<Self, LoweringError> {
        Self::from_operation(op.get())
    }

    fn from_operation(op: &dyn Operation) -> Result<Self, LoweringError> {
        if let Some(op) = op.downcast_ref::<JvpOperation>() {
            Ok(op.parameters())
        } else if let Some(op) = op.downcast_ref::<VjpOperation>() {
            Ok(op.parameters())
        } else {
            Err(signature("expected a differentiation operation"))
        }
    }

    pub fn jvp_op(&self) -> Op {
        Op::new(JvpBuilder {
            wrt: stored_indices(&self.wrt),
            of: stored_indices(&self.of),
        })
    }

    pub fn vjp_op(&self) -> Op {
        Op::new(VjpBuilder {
            wrt: stored_indices(&self.wrt),
            of: stored_indices(&self.of),
        })
    }

    /// Check the operand (if known) and result metadata against the body.
    /// `forward` selects the jvp layout; otherwise the vjp layout.
    pub fn check(
        &self,
        body_inputs: &[gnx_expr::VarInfo],
        body_outputs: &[gnx_expr::VarInfo],
        operands: Option<&[gnx_expr::VarInfo]>,
        operand_count: usize,
        outputs: &[gnx_expr::VarInfo],
        forward: bool,
    ) -> Result<(), LoweringError> {
        strictly_increasing(&self.wrt, body_inputs.len())?;
        strictly_increasing(&self.of, body_outputs.len())?;
        let wrt: Vec<_> = self.wrt.iter().map(|&i| body_inputs[i].clone()).collect();
        let of: Vec<_> = self.of.iter().map(|&i| body_outputs[i].clone()).collect();
        if wrt.iter().chain(&of).any(|info| !util::is_inexact(info)) {
            return Err(signature(
                "differentiated inputs and results must be floating-point arrays",
            ));
        }
        let (extra_inputs, extra_outputs) = if forward { (wrt, of) } else { (of, wrt) };
        let expected_inputs: Vec<_> = body_inputs.iter().cloned().chain(extra_inputs).collect();
        let expected_outputs: Vec<_> = body_outputs.iter().cloned().chain(extra_outputs).collect();
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
        impl Operation for $ty {
            fn display(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                let parameters = self.parameters();
                write!(
                    f,
                    concat!("ad::", $name, "<of: {:?}, wrt: {:?}>"),
                    parameters.of, parameters.wrt
                )
            }
            fn lower(
                &self,
                eqn: HEqn<'_>,
                ctx: &mut LoweringContext<'_>,
            ) -> Result<(), LoweringError> {
                ctx.lower_value_region_apply(eqn, |inputs, results, operands, outputs| {
                    self.parameters().check(
                        inputs,
                        results,
                        Some(operands),
                        operands.len(),
                        outputs,
                        $forward,
                    )
                })
            }
            fn check_expr(&self, eqn: Eqn<'_>) -> Result<(), LoweringError> {
                let (inputs, results) = util::region_signature(eqn)?;
                self.parameters().check(
                    &inputs,
                    &results,
                    None,
                    eqn.inputs().len(),
                    &util::output_infos(eqn),
                    $forward,
                )
            }
        }

        impl Transform for $ty {
            fn expand(&self, b: &mut Builder, eqn: &EqnSource) -> TransformResult<Vec<Var>> {
                $expand(b, eqn, &self.parameters())
            }
        }
    };
}
diff_operation!(JvpOperation, "jvp", true, jvp::expand_jvp);
diff_operation!(VjpOperation, "vjp", false, transpose::expand_vjp);
register_op_interface!(dyn Transform: JvpOperation, VjpOperation);

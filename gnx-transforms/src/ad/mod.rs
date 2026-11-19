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
//! | op | operands | results | attributes |
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
pub mod def;
pub mod jvp;
pub mod region;
mod rules;
pub mod transpose;

pub use api::{Grad, Hessian, Jvp, ValueAndGrad, Vjp, grad, hessian, jvp, value_and_grad, vjp};
pub use jvp::{Dual, JvpRule, Tangent};
pub use transpose::{Operand, TransposeRule};

use gnx_expr::lower::LoweringContext;
use gnx_expr::pass::Builder;
use gnx_expr::{
    AttrMap, Dialect, Eqn, HEqn, LoweringError, Op, Operation, Var, register_op_interface,
};
use gnx_math::ops::{index_attr, index_list};

use crate::body::{
    BodySignature, check_expr_with, exact_attrs, lower_transform, signature, strictly_increasing,
};
use crate::expand::Transform;
use crate::{TransformResult, util};

pub struct AdDialect;
pub struct JvpOperation;
pub struct VjpOperation;

static AD: AdDialect = AdDialect;
gnx_expr::register_dialect!(&AD);

impl Dialect for AdDialect {
    fn name(&self) -> &'static str {
        "ad"
    }

    fn create(&self, name: &'static str) -> Option<&'static dyn Operation> {
        Some(match name {
            "jvp" => &JvpOperation,
            "vjp" => &VjpOperation,
            _ => return None,
        })
    }
}

/// Parameters of `ad.jvp` and `ad.vjp`: differentiate the body's results
/// `of` with respect to its inputs `wrt` (both strictly increasing).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiffParameters {
    pub wrt: Vec<usize>,
    pub of: Vec<usize>,
}

impl DiffParameters {
    pub fn from_op(op: &Op) -> Result<Self, LoweringError> {
        exact_attrs(op, &["wrt", "of"])?;
        Ok(Self {
            wrt: index_list(op, "wrt")?,
            of: index_list(op, "of")?,
        })
    }

    fn attrs(&self) -> AttrMap {
        [
            ("wrt".into(), index_attr(&self.wrt)),
            ("of".into(), index_attr(&self.of)),
        ]
        .into()
    }

    pub fn jvp_op(&self) -> Op {
        Op::new_with_attrs(&JvpOperation, self.attrs())
    }

    pub fn vjp_op(&self) -> Op {
        Op::new_with_attrs(&VjpOperation, self.attrs())
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
        if wrt.iter().chain(&of).any(|info| !util::is_float(info)) {
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
                let parameters = DiffParameters::from_op(eqn.op())?;
                lower_transform(eqn, ctx, |body, operands, outputs| {
                    parameters.check(body, Some(operands), operands.len(), outputs, $forward)
                })
            }
            fn check_expr(&self, eqn: &Eqn) -> Result<(), LoweringError> {
                let parameters = DiffParameters::from_op(eqn.op())?;
                check_expr_with(eqn, |body, count, outputs| {
                    parameters.check(body, None, count, outputs, $forward)
                })
            }
        }

        impl Transform for $ty {
            fn expand(&self, b: &mut Builder, eqn: &Eqn) -> TransformResult<Vec<Var>> {
                $expand(b, eqn)
            }
        }
    };
}
diff_operation!(JvpOperation, "jvp", true, jvp::expand_jvp);
diff_operation!(VjpOperation, "vjp", false, transpose::expand_vjp);
register_op_interface!(dyn Transform: JvpOperation, VjpOperation);

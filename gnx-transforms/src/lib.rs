extern crate self as gnx_transforms;

pub mod ad;
pub use ad::stop_gradient;
mod cache;
pub mod callable;
pub mod cf;
mod error;
pub mod expand;
pub mod jit;
pub mod map;
mod spec;
mod traced;
pub mod util;

pub use ad::{grad, hessian, jvp, value_and_grad, vjp};
pub use callable::{Callable, Function, Jitable, TryCallable, TryJitable};
pub use cf::{call, call_immutable, call_lowered, cond, scan, switch, while_loop};
pub use error::{ApplyError, TransformError, TransformResult};
pub use expand::{ExpandTransforms, Transform, expand_transforms};
pub use jit::{Jit, jit, jit_fn, jittable, try_jit_fn};
pub use map::map;

// Macros: `map!`, `jvp!`, ... wrap a closure like `jit_fn!` and take builder
// options in `dag!` syntax.
pub use gnx_derive::{grad, hessian, jvp, map, value_and_grad, vjp};

#[doc(hidden)]
pub mod __private {
    pub use gnx_expr;
}

/// Everything needed to give an operation transformation rules.
///
/// Most operations need one declaration per transformation, in the style of
/// JAX's rule helpers:
///
/// ```ignore
/// use gnx_transforms::rules::*;
///
/// defbilinear!(MulOperation, |_, ct, [_, y]| Ok(ct * y.val()?), |_, ct, [x, _]| Ok(x.val()? * ct));
/// defbroadcasting!(MulOperation);
/// ```
///
/// | Macro | Defines | For |
/// |---|---|---|
/// | [`defjvp!`] | forward mode, one tangent closure per operand | nonlinear operations |
/// | [`deftranspose!`] | transposition, one cotangent closure per operand | linear operations with a custom JVP |
/// | [`deflinear!`] | forward mode and transposition | operations linear in all operands |
/// | [`defbilinear!`] | forward mode and transposition | products |
/// | [`defbroadcasting!`] | batching | elementwise operations |
/// | [`defbatch!`] | batching along a new leading axis | structural operations |
///
/// Reverse mode needs no rule of its own: it linearizes with the JVP rules
/// and transposes the linear part. Operations without operands need no rules
/// at all. Operations that own regions implement [`JvpRule`],
/// [`TransposeRule`] and [`BatchRule`] directly.
pub mod rules {
    pub use crate::ad::def::{Arg, Ctx, Val};
    pub use crate::ad::rules::reduction::{
        ReduceJvpRule, additive_reducer, copy_reduction_primal, reducer_body,
    };
    pub use crate::ad::{Dual, JvpRule, Operand, Tangent, TransposeRule};
    pub use crate::map::batch::materialize;
    pub use crate::map::def::{leading, shifted};
    pub use crate::map::rules::array::batch_reduce;
    pub use crate::map::{BatchRule, Batched};
    pub use crate::{TransformError, TransformResult};
    pub use crate::{defbatch, defbilinear, defbroadcasting, defjvp, deflinear, deftranspose};
}

/// Constructors for the transform and control-flow dialects in expression macros.
pub mod ops {
    pub mod ad {
        use crate::ad::DiffParameters;
        pub use crate::ad::stop_gradient_op as stop_gradient;
        use gnx_expr::Op;

        pub fn jvp(wrt: &[usize], of: &[usize]) -> Op {
            DiffParameters {
                wrt: wrt.to_vec(),
                of: of.to_vec(),
            }
            .jvp_op()
        }

        pub fn vjp(wrt: &[usize], of: &[usize]) -> Op {
            DiffParameters {
                wrt: wrt.to_vec(),
                of: of.to_vec(),
            }
            .vjp_op()
        }
    }

    pub mod cf {
        pub use crate::cf::{
            call_op as call, cond_op as cond, scan_op as scan, switch_op as switch,
            while_op as while_loop,
        };
    }

    pub mod map {
        use crate::map::MapParameters;
        use gnx_expr::Op;

        pub fn map(size: usize, in_axes: &[usize], out_axes: &[usize]) -> Op {
            MapParameters {
                size,
                in_axes: in_axes.to_vec(),
                out_axes: out_axes.to_vec(),
            }
            .to_op()
        }
    }
}

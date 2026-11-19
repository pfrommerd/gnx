extern crate self as gnx_transforms;

pub mod ad;
pub mod body;
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
pub use body::{Body, BodySignature};
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
    pub use crate::ad::{Dual, JvpRule, Operand, Tangent, TransposeRule};
    pub use crate::map::batch::materialize;
    pub use crate::map::def::{leading, shifted};
    pub use crate::map::{BatchRule, Batched};
    pub use crate::{TransformError, TransformResult};
    pub use crate::{defbatch, defbilinear, defbroadcasting, defjvp, deflinear, deftranspose};
}

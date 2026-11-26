pub use gnx_base::{AsDType, IntoScalar, Scalar, half};
pub use gnx_transforms as transforms;
pub use gnx_transforms::callable;

// Re-export gnx-graph as graph
// and gnx-io as io
pub use gnx_expr as expr;
pub use gnx_graph as graph;
/// Neural-network layers compose as ordinary graph fields. Select their
/// trainable arrays by type when differentiating a loss:
///
/// ```
/// use gnx::{Array, Graph, array::DType, graph::Of, lax, nn, np, value_and_grad};
///
/// #[derive(Clone, Graph)]
/// struct Model { head: nn::Linear }
///
/// let head = nn::Linear::new(lax::zeros([2, 3], DType::F32), Some(lax::zeros([2], DType::F32)));
/// let x = lax::zeros([5, 3], DType::F32);
/// let model = Model { head };
/// let loss = value_and_grad!(
///     |m: &Model, x: Array| np::sum(&m.head.forward(x), np::ALL),
///     wrt = Of::<nn::Param>::filter(),
/// );
/// let (value, (grads, _)) = loss(&model, x);
/// // Apply an update outside the loss. The gradient cells are independent.
/// let bias = model.head.bias().unwrap();
/// bias.set(bias.value() - grads.head.bias().unwrap().value() * 0.01);
/// ```
pub use gnx_nn as nn;

pub use gnx_base::{cache, pass, store};

pub use gnx_math::{array, buffer, device, lax, np, random, shard};
/// Constructors for operations in expression macros.
///
/// ```
/// use gnx::array::{ArrayInfo, DType, Shape};
/// use gnx::expr::{VarInfo, expr};
/// use gnx::ops;
///
/// let input = VarInfo::new(ArrayInfo::new(Shape::fixed([2, 3]), DType::F32));
/// let output = VarInfo::new(ArrayInfo::new(Shape::fixed([3, 2]), DType::F32));
/// let body = expr!(|x: input| => {
///     y: output = array::transpose<&[1, 0]> x;
///     in (y)
/// });
/// let body = body.build().unwrap();
/// body.view().check().unwrap();
/// ```
pub mod ops {
    pub use gnx_math::ops::*;
    pub use gnx_transforms::ops::{ad, cf, map};
}

pub mod eval;
pub use eval::{CaptureError, Eval, EvalError, EvalPlanError, expr};

// Include all util modules from gnx-graph and gnx-io
pub mod util {
    pub use gnx_graph::util::*;
    pub use gnx_io::util::*;
}

// for convenience, re-export a handful of top-level namespace items
// - Array and Shape from gnx-math
// - Graph and GraphId from gnx-graph
pub use gnx_graph::{Graph, GraphId};
pub use gnx_math::array::{Array, IntoShape, Shape};
pub use gnx_math::device::{Device, ExecOpts};

pub use gnx_transforms::{
    Callable, ExpandTransforms, Function, Jit, Jitable, TryCallable, TryJitable, grad, hessian,
    jit, jit_fn, jittable, jvp, map, on_axis, scan_graph, stop_gradient, try_jit_fn,
    value_and_grad, vjp,
};

extern crate self as gnx;

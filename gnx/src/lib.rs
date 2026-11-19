pub use gnx_transforms as transforms;
pub use gnx_transforms::callable;

// Re-export gnx-graph as graph
// and gnx-io as io
pub use gnx_expr as expr;
pub use gnx_graph as graph;

pub use gnx_math::{array, buffer, device, shard};
/// Constructors for operations in expression macros.
///
/// ```
/// use gnx::array::{ArrayInfo, DType, Shape};
/// use gnx::expr::{ValueInfo, expr};
/// use gnx::ops;
///
/// let input = ValueInfo::new(ArrayInfo::new(Shape::fixed([2, 3]), DType::F32));
/// let output = ValueInfo::new(ArrayInfo::new(Shape::fixed([3, 2]), DType::F32));
/// let body = expr!(|x: input| => {
///     y: output = array::transpose<&[1, 0]> x;
///     in (y)
/// });
/// body.check().unwrap();
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
pub use gnx_math::array::{Array, Shape};
pub use gnx_math::device::{Device, ExecOpts};

pub use gnx_transforms::{
    Callable, ExpandTransforms, Function, Jit, Jitable, TryCallable, TryJitable, grad, hessian,
    jit, jit_fn, jittable, jvp, map, try_jit_fn, value_and_grad, vjp,
};

extern crate self as gnx;

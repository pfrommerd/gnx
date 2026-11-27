//! Elementwise activations and numerically stable last-axis softmaxes.
pub use gnx_math::np::{log_softmax, softmax};
use gnx_math::{Array, lax};
pub use lax::{logistic as sigmoid, tanh};
pub fn relu(x: &Array) -> Array {
    lax::max(x, &lax::zeros(x.shape(), x.dtype()))
}
pub fn silu(x: &Array) -> Array {
    x * sigmoid(x)
}
/// Gate a value projection with the SiLU of its gate projection.
pub fn swiglu(gate: &Array, value: &Array) -> Array {
    silu(gate) * value
}
/// Tanh approximation to GELU.
pub fn gelu(x: &Array) -> Array {
    x * 0.5
        * (1.0_f64 + lax::tanh(&((x + x * x * x * 0.044715) * (2. / std::f64::consts::PI).sqrt())))
}

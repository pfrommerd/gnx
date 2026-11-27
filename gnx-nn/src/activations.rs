//! Elementwise activations and numerically stable last-axis softmaxes.
use gnx_graph::Leaf;
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

pub type Activation = fn(&Array) -> Array;

#[derive(Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord, Leaf)]
pub enum ActivationConfig {
    Relu,
    Gelu,
    Silu,
    Tanh,
    Sigmoid,
}

impl From<ActivationConfig> for Activation {
    fn from(value: ActivationConfig) -> Self {
        use ActivationConfig::*;
        match value {
            Relu => relu,
            Gelu => gelu,
            Silu => silu,
            Tanh => tanh,
            Sigmoid => sigmoid,
        }
    }
}

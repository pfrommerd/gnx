//! Neural-network layers are ordinary owned graph values. Mark trainable
//! arrays with [`Param`]; use `Of::<Param>::filter()` with GNX transforms.

use gnx_graph::{Graph, Leaf, cast, util::LifetimeFree};
use gnx_math::{Array, ArrayRef};

pub mod activations;
pub mod attention;
pub mod embedding;
pub mod linear;
pub mod mlp;
pub mod norm;
pub mod rngs;
mod tensor;
pub use attention::*;
pub use embedding::*;
pub use linear::*;
pub use mlp::*;
pub use norm::*;
pub use rngs::*;

/// A trainable array cell. Clones share the same parameter; scalar-loss
/// gradients contain fresh cells independent of the model.
#[derive(Clone, Graph, LifetimeFree)]
pub struct Param {
    value: ArrayRef,
}

impl Param {
    pub fn new(value: Array) -> Self {
        Self {
            value: ArrayRef::new(value),
        }
    }

    /// Read the current value, recording a read when tracing.
    pub fn value(&self) -> Array {
        self.value.get()
    }

    /// Replace the contents without breaking ties with other handles.
    pub fn set(&self, value: Array) {
        self.value.set(value);
    }

    pub fn into_value(self) -> Array {
        self.value.get()
    }
}

impl Leaf for Param {
    type Ref<'a> = &'a Self;
    fn as_ref(&self) -> &Self {
        self
    }
    fn clone_ref(value: &Self) -> Self {
        value.clone()
    }
    fn try_from_value<V>(value: V) -> Result<Self, V> {
        cast!(value, Self)
    }
    fn try_from_ref<V>(value: &V) -> Result<&Self, &V> {
        cast!(value, &Self)
    }
    fn try_into_value<V: 'static>(self) -> Result<V, Self> {
        cast!(self, V)
    }
}

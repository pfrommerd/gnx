//! Neural-network layers are ordinary owned graph values. Mark trainable
//! arrays with [`Param`]; use `Of::<Param>::filter()` with GNX transforms.

use gnx_graph::{Graph, MaybeLeaf, util::LifetimeFree};
use gnx_io::{Deserialize, Serialize};
use gnx_math::{Array, ArrayRef};

pub mod activations;
pub mod attention;
pub mod cache;
pub mod embedding;
mod init;
pub mod linear;
pub mod mlp;
pub mod norm;
pub mod optim;
pub mod rngs;
mod tensor;
pub use attention::*;
pub use cache::{CacheKey, KVCache, KvPrefix};
pub use embedding::*;
pub use init::*;
pub use linear::*;
pub use mlp::*;
pub use norm::*;
pub use rngs::*;

/// A trainable array cell. Clones share the same parameter; scalar-loss
/// gradients contain fresh cells independent of the model.
#[derive(Clone, Graph, LifetimeFree, MaybeLeaf)]
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

pub trait Model: Graph + InitializeWith<Self::DefaultInit> {
    type Config: ModelConfig<Model = Self>;
    type DefaultInit: Default;

    /// Build the parameter and buffer layout with shape/dtype placeholders, without drawing values.
    fn empty(config: Self::Config) -> Self;
}

/// Construct a model using a particular initializer or bundle of initializers.
pub trait InitializeWith<I>: Sized {
    fn init(config: <Self as Model>::Config, init: I, rngs: &Rngs) -> Self
    where
        Self: Model;
}

pub trait ModelConfig: Graph + Serialize + for<'de> Deserialize<'de> {
    type Model: Model<Config = Self>;
}

use crate::{
    Rngs, RngsExt,
    tensor::{constant, work},
};
use gnx_math::{Array, ArrayInfo, DType, IntoShape, Shape, random};

/// An NNX-style initializer for one array. `key` is a scalar random key.
pub trait Initializer {
    fn initialize(&self, key: &Array, shape: impl IntoShape, dtype: DType) -> Array;

    /// Initialize from the model's parameter stream. Keyless initializers skip the draw.
    fn initialize_from(&self, rngs: &Rngs, shape: impl IntoShape, dtype: DType) -> Array {
        self.initialize(&rngs.params(), shape, dtype)
    }
}

impl<F: Fn(&Array, Shape, DType) -> Array> Initializer for F {
    fn initialize(&self, key: &Array, shape: impl IntoShape, dtype: DType) -> Array {
        self(key, shape.into_shape(), dtype)
    }
}

#[derive(Clone, Copy, Default, Debug)]
pub struct Empty;
impl Initializer for Empty {
    fn initialize(&self, _: &Array, shape: impl IntoShape, dtype: DType) -> Array {
        Array::placeholder(ArrayInfo::new(shape, dtype))
    }
    fn initialize_from(&self, _: &Rngs, shape: impl IntoShape, dtype: DType) -> Array {
        Array::placeholder(ArrayInfo::new(shape, dtype))
    }
}

#[derive(Clone, Copy, Default, Debug)]
pub struct Fill(f64);
impl Initializer for Fill {
    fn initialize(&self, _: &Array, shape: impl IntoShape, dtype: DType) -> Array {
        constant(self.0, shape, dtype)
    }
    fn initialize_from(&self, _: &Rngs, shape: impl IntoShape, dtype: DType) -> Array {
        constant(self.0, shape, dtype)
    }
}

/// Default linear and convolution policy. A bias's own shape does not encode its fan-in.
#[derive(Clone, Copy, Default, Debug)]
pub struct FanInUniform;

/// A weight initializer gets the fan_in
pub trait WeightInitializer {
    fn initialize_weight(
        &self,
        rngs: &Rngs,
        shape: impl IntoShape,
        dtype: DType,
        fan_in: usize,
    ) -> Array;
}
impl<I: Initializer> WeightInitializer for I {
    fn initialize_weight(
        &self,
        rngs: &Rngs,
        shape: impl IntoShape,
        dtype: DType,
        _: usize,
    ) -> Array {
        self.initialize_from(rngs, shape, dtype)
    }
}

impl WeightInitializer for FanInUniform {
    fn initialize_weight(
        &self,
        rngs: &Rngs,
        shape: impl IntoShape,
        dtype: DType,
        fan_in: usize,
    ) -> Array {
        let bound = if fan_in == 0 {
            0.
        } else {
            (fan_in as f64).sqrt().recip()
        };
        let u = random::uniform(&rngs.params(), shape, work(dtype));
        (u * (2. * bound) - bound).astype(dtype)
    }
}

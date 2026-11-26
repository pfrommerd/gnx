//! Neural-network layers are ordinary owned graph values. Mark trainable
//! arrays with [`Param`]; use `Of::<Param>::filter()` with GNX transforms.

use gnx_graph::{Graph, Leaf, cast, util::LifetimeFree};
use gnx_math::{Array, ArrayRef, lax};

/// Conventional neural-network stream names on the generic random stream collection.
pub trait RngsExt {
    fn params(&self) -> Array;
    fn dropout(&self) -> Array;
}
impl RngsExt for gnx_math::random::Rngs {
    fn params(&self) -> Array {
        self.next("params")
    }
    fn dropout(&self) -> Array {
        self.next("dropout")
    }
}

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

/// An affine projection over the last input axis: `x @ weight.T + bias`.
#[derive(Clone, Graph)]
pub struct Linear {
    pub weight: Param,
    pub bias: Param,
}

impl Linear {
    /// PyTorch-style initialization: independently uniform weight and bias in
    /// ±1/sqrt(fan_in). Each array consumes one key from the `params` stream.
    pub fn init(
        in_features: usize,
        out_features: usize,
        dtype: gnx_math::DType,
        rngs: &gnx_math::random::Rngs,
    ) -> Self {
        use gnx_math::{DType, random};
        assert!(matches!(
            dtype,
            DType::BF16 | DType::F16 | DType::F32 | DType::F64
        ));
        let work = if dtype == DType::F64 {
            dtype
        } else {
            DType::F32
        };
        let bound = if in_features == 0 {
            0.
        } else {
            (in_features as f64).sqrt().recip()
        };
        let init = |shape: gnx_math::Shape| {
            let u = random::uniform(&rngs.params(), shape, work);
            lax::convert_element_type(
                &(u * lax::scalar(2. * bound, work) - lax::scalar(bound, work)),
                dtype,
            )
        };
        use gnx_math::IntoShape;
        Self::new(
            init((out_features, in_features).into_shape()),
            init((out_features,).into_shape()),
        )
    }
    /// Construct from a `[out_features, in_features]` weight matrix and
    /// `[out_features]` bias of the same dtype.
    pub fn new(weight: Array, bias: Array) -> Self {
        let shape = weight.shape();
        assert_eq!(shape.dims().len(), 2, "Linear weight must be a matrix");
        assert_eq!(
            bias.shape().dims(),
            &shape.dims()[..1],
            "Linear bias must match the output axis"
        );
        assert_eq!(
            weight.dtype(),
            bias.dtype(),
            "Linear weight and bias must have the same dtype"
        );
        Self {
            weight: Param::new(weight),
            bias: Param::new(bias),
        }
    }

    /// Map `[..., in_features]` to `[..., out_features]`.
    pub fn forward(&self, input: Array) -> Array {
        let axis = input
            .shape()
            .dims()
            .len()
            .checked_sub(1)
            .expect("Linear input needs a feature axis");
        let y = lax::dot_general(
            &input,
            &self.weight.value(),
            lax::DotDimensions {
                lhs_contracting: vec![axis],
                rhs_contracting: vec![1],
                lhs_batch: vec![],
                rhs_batch: vec![],
            },
        );
        y + self.bias.value()
    }
}

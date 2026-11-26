//! Neural-network layers are ordinary owned graph values. Mark trainable
//! arrays with [`Param`]; use `Of::<Param>::filter()` with GNX transforms.

use gnx_graph::{Graph, Leaf, cast, util::LifetimeFree};
use gnx_math::{Array, lax};

/// A trainable array. Role filters match the wrapper; tracer and array
/// filters descend into its `value` field.
#[derive(Clone, Graph, LifetimeFree)]
pub struct Param {
    value: Array,
}

impl Param {
    pub fn new(value: Array) -> Self {
        Self { value }
    }

    pub fn value(&self) -> &Array {
        &self.value
    }

    pub fn into_value(self) -> Array {
        self.value
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
            self.weight.value(),
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

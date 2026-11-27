use crate::{Empty, FanInUniform, InitializeWith, Model, ModelConfig, Rngs, WeightInitializer};
use crate::{Param, tensor::*};
use gnx_graph::Graph;
use gnx_io::{Deserialize, Serialize};
use gnx_math::{
    Array, DType,
    lax::{self, ConvAxes, ConvKernelAxes, ConvParameters},
};

/// N-dimensional channels-last convolution, kernel `[spatial..., in/groups, out]`.
#[derive(Clone, Graph)]
pub struct Conv {
    pub kernel: Param,
    pub bias: Option<Param>,
    pub strides: Vec<usize>,
    pub padding: Vec<(i64, i64)>,
    pub dilation: Vec<usize>,
    pub groups: usize,
}
#[derive(Clone, Graph, Serialize, Deserialize)]
pub struct ConvConfig {
    pub input: usize,
    pub output: usize,
    pub kernel: Vec<usize>,
    pub groups: usize,
    pub bias: bool,
    pub dtype: DType,
}
impl ModelConfig for ConvConfig {
    type Model = Conv;
}

/// Initializers for the fields of [`Conv`].
#[derive(Default)]
pub struct ConvInit<Kernel = FanInUniform, Bias = FanInUniform> {
    pub kernel: Kernel,
    pub bias: Bias,
}
impl Model for Conv {
    type Config = ConvConfig;
    type DefaultInit = ConvInit;

    fn empty(config: Self::Config) -> Self {
        Self::init(
            config,
            ConvInit {
                kernel: Empty,
                bias: Empty,
            },
            &Rngs::default(),
        )
    }
}
impl<Kernel, Bias> InitializeWith<ConvInit<Kernel, Bias>> for Conv
where
    Kernel: WeightInitializer,
    Bias: WeightInitializer,
{
    fn init(config: ConvConfig, init: ConvInit<Kernel, Bias>, rngs: &Rngs) -> Self {
        let ConvConfig {
            input,
            output,
            kernel,
            groups,
            bias,
            dtype: t,
        } = config;
        assert!(
            groups > 0
                && input > 0
                && output > 0
                && input % groups == 0
                && output % groups == 0
                && !kernel.is_empty()
                && !kernel.contains(&0)
        );
        let fan = kernel.iter().product::<usize>() * (input / groups);
        let mut shape = kernel;
        shape.extend([input / groups, output]);
        Conv::new(
            init.kernel.initialize_weight(rngs, shape, t, fan),
            bias.then(|| init.bias.initialize_weight(rngs, [output], t, fan)),
            groups,
        )
    }
}

impl Conv {
    /// Construct from a `[spatial..., in/groups, out]` kernel and optional `[out]` bias.
    pub fn new(kernel: Array, bias: Option<Array>, groups: usize) -> Self {
        Self::from_params(Param::new(kernel), bias.map(Param::new), groups)
    }

    pub fn from_params(kernel: Param, bias: Option<Param>, groups: usize) -> Self {
        let w = kernel.value();
        let shape = w
            .fixed_dims()
            .expect("Conv kernel requires fixed dimensions");
        assert!(
            shape.len() >= 3 && !shape.contains(&0),
            "invalid Conv kernel shape"
        );
        let n = shape.len() - 2;
        let output = shape[n + 1];
        assert!(groups > 0 && output % groups == 0, "invalid Conv groups");
        if let Some(b) = &bias {
            let b = b.value();
            assert_eq!(
                b.shape().dims(),
                &w.shape().dims()[n + 1..],
                "Conv bias must match the output axis"
            );
            assert_eq!(
                w.dtype(),
                b.dtype(),
                "Conv kernel and bias must have the same dtype"
            );
        }
        Self {
            kernel,
            bias,
            strides: vec![1; n],
            padding: vec![(0, 0); n],
            dilation: vec![1; n],
            groups,
        }
    }
    pub fn bias(&self) -> Option<&Param> {
        self.bias.as_ref()
    }
    pub fn forward(&self, x: Array) -> Array {
        let w = self.kernel.value();
        let n = self.strides.len();
        let layout = ConvAxes {
            batch: 0,
            feature: n + 1,
            spatial: (1..=n).collect(),
        };
        let p = ConvParameters {
            input: layout.clone(),
            output: layout,
            kernel: ConvKernelAxes {
                input_feature: n,
                output_feature: n + 1,
                spatial: (0..n).collect(),
            },
            strides: self.strides.clone(),
            padding: self.padding.clone(),
            rhs_dilation: self.dilation.clone(),
            lhs_dilation: vec![1; n],
            feature_groups: self.groups,
            batch_groups: 1,
        };
        let y = lax::conv_general_dilated(&x, &w, p);
        match self.bias() {
            Some(b) => y + b.value(),
            None => y,
        }
    }
}

/// Project the last input axis with `x @ weight.T`, optionally adding a bias.
#[derive(Clone, Graph)]
pub struct Linear {
    pub weight: Param,
    pub bias: Option<Param>,
}

/// Each initialized array consumes one key and is uniform in ±1/sqrt(in_features).
#[derive(Clone, Graph, Serialize, Deserialize)]
pub struct LinearConfig {
    pub in_features: usize,
    pub out_features: usize,
    pub bias: bool,
    pub dtype: DType,
}
impl ModelConfig for LinearConfig {
    type Model = Linear;
}

impl LinearConfig {
    pub(crate) fn projection(in_features: usize, out_features: usize, dtype: DType) -> Self {
        Self {
            in_features,
            out_features,
            bias: false,
            dtype,
        }
    }
}

/// Initializers for the fields of [`Linear`].
#[derive(Default)]
pub struct LinearInit<Weight = FanInUniform, Bias = FanInUniform> {
    pub weight: Weight,
    pub bias: Bias,
}
impl Linear {
    pub(crate) fn empty_bundle() -> LinearInit<Empty, Empty> {
        LinearInit {
            weight: Empty,
            bias: Empty,
        }
    }
}

impl Model for Linear {
    type Config = LinearConfig;
    type DefaultInit = LinearInit;

    fn empty(config: Self::Config) -> Self {
        Self::init(config, Self::empty_bundle(), &Rngs::default())
    }
}
impl<Weight, Bias> InitializeWith<LinearInit<Weight, Bias>> for Linear
where
    Weight: WeightInitializer,
    Bias: WeightInitializer,
{
    fn init(config: LinearConfig, init: LinearInit<Weight, Bias>, rngs: &Rngs) -> Self {
        let LinearConfig {
            in_features,
            out_features,
            bias,
            dtype,
        } = config;
        Linear::new(
            init.weight
                .initialize_weight(rngs, [out_features, in_features], dtype, in_features),
            bias.then(|| {
                init.bias
                    .initialize_weight(rngs, [out_features], dtype, in_features)
            }),
        )
    }
}
impl Linear {
    /// Construct from an `[out, in]` matrix and an optional `[out]` bias.
    pub fn new(weight: Array, bias: Option<Array>) -> Self {
        Self::from_params(Param::new(weight), bias.map(Param::new))
    }

    /// Reuse existing parameter cells (for example, a tied embedding weight).
    pub fn from_params(weight: Param, bias: Option<Param>) -> Self {
        let w = weight.value();
        let shape = w.shape();
        assert_eq!(shape.dims().len(), 2, "Linear weight must be a matrix");
        if let Some(b) = &bias {
            let b = b.value();
            assert_eq!(
                b.shape().dims(),
                &shape.dims()[..1],
                "Linear bias must match the output axis"
            );
            assert_eq!(
                w.dtype(),
                b.dtype(),
                "Linear weight and bias must have the same dtype"
            );
        }
        Self { weight, bias }
    }

    pub fn bias(&self) -> Option<&Param> {
        self.bias.as_ref()
    }

    /// Map `[..., in_features]` to `[..., out_features]`.
    pub fn forward(&self, input: Array) -> Array {
        let y = project(&input, &self.weight.value());
        match self.bias() {
            Some(bias) => y + bias.value(),
            None => y,
        }
    }
}

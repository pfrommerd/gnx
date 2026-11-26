use crate::{Param, Rngs, RngsExt, tensor::*};
use gnx_graph::Graph;
use gnx_math::{
    Array, DType,
    lax::{self, ConvAxes, ConvKernelAxes, ConvParameters},
    random,
};

fn uniform(shape: Vec<usize>, fan: usize, t: DType, r: &Rngs) -> Array {
    let w = work(t);
    let bound = if fan == 0 {
        0.
    } else {
        (fan as f64).sqrt().recip()
    };
    let u = random::uniform(&r.params(), shape, w);
    (u * (2. * bound) - bound).astype(t)
}

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
impl Conv {
    /// Initialize both kernel and bias. Use `init_with` to omit the bias.
    pub fn init(
        input: usize,
        output: usize,
        kernel: &[usize],
        groups: usize,
        t: DType,
        r: &Rngs,
    ) -> Self {
        Self::init_with(input, output, kernel, groups, true, t, r)
    }

    pub fn init_with(
        input: usize,
        output: usize,
        kernel: &[usize],
        groups: usize,
        bias: bool,
        t: DType,
        r: &Rngs,
    ) -> Self {
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
        let mut shape = kernel.to_vec();
        shape.extend([input / groups, output]);
        Self::new(
            uniform(shape, fan, t, r),
            bias.then(|| uniform(vec![output], fan, t, r)),
            groups,
        )
    }

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

impl Linear {
    /// PyTorch-style initialization: independently uniform weight and bias in
    /// ±1/sqrt(fan_in). Each array consumes one key from the `params` stream.
    pub fn init(
        in_features: usize,
        out_features: usize,
        dtype: gnx_math::DType,
        rngs: &Rngs,
    ) -> Self {
        Self::init_with(in_features, out_features, true, dtype, rngs)
    }

    /// When `bias` is false, initialize only the weight, consuming one key.
    pub fn init_with(
        in_features: usize,
        out_features: usize,
        bias: bool,
        dtype: DType,
        rngs: &Rngs,
    ) -> Self {
        Self::new(
            uniform(vec![out_features, in_features], in_features, dtype, rngs),
            bias.then(|| uniform(vec![out_features], in_features, dtype, rngs)),
        )
    }

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

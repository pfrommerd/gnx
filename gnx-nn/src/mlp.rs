//! Bias-free feed-forward blocks and token-wise sparse expert routing.
use crate::{Empty, Fill, InitializeWith, Initializer, LinearInit, Model, ModelConfig, Rngs};
use crate::{Linear, LinearConfig, Param, RngsExt, activations, tensor::*};
use gnx_graph::Graph;
use gnx_io::{Deserialize, Serialize};
use gnx_math::{Array, ArrayRef, DType, lax, random};

#[derive(Clone, Graph)]
pub struct SwiGlu {
    pub gate: Linear,
    pub up: Linear,
    pub down: Linear,
}
#[derive(Clone, Graph, Serialize, Deserialize)]
pub struct SwiGluConfig {
    pub hidden: usize,
    pub width: usize,
    pub dtype: DType,
}
impl ModelConfig for SwiGluConfig {
    type Model = SwiGlu;
}

/// Initializers for the fields of [`SwiGlu`].
#[derive(Default)]
pub struct SwiGluInit<Gate = LinearInit, Up = LinearInit, Down = LinearInit> {
    pub gate: Gate,
    pub up: Up,
    pub down: Down,
}
impl SwiGlu {
    pub(crate) fn empty_bundle()
    -> SwiGluInit<LinearInit<Empty, Empty>, LinearInit<Empty, Empty>, LinearInit<Empty, Empty>>
    {
        SwiGluInit {
            gate: Linear::empty_bundle(),
            up: Linear::empty_bundle(),
            down: Linear::empty_bundle(),
        }
    }
}

impl Model for SwiGlu {
    type Config = SwiGluConfig;
    type DefaultInit = SwiGluInit;

    fn empty(config: Self::Config) -> Self {
        Self::init(config, Self::empty_bundle(), &Rngs::default())
    }
}
impl<Gate, Up, Down> InitializeWith<SwiGluInit<Gate, Up, Down>> for SwiGlu
where
    Linear: InitializeWith<Gate>,
    Linear: InitializeWith<Up>,
    Linear: InitializeWith<Down>,
{
    fn init(config: SwiGluConfig, init: SwiGluInit<Gate, Up, Down>, rngs: &Rngs) -> Self {
        let SwiGluConfig {
            hidden,
            width,
            dtype: t,
        } = config;
        SwiGlu {
            gate: Linear::init(LinearConfig::projection(hidden, width, t), init.gate, rngs),
            up: Linear::init(LinearConfig::projection(hidden, width, t), init.up, rngs),
            down: Linear::init(LinearConfig::projection(width, hidden, t), init.down, rngs),
        }
    }
}
impl SwiGlu {
    pub fn forward(&self, x: Array) -> Array {
        self.forward_with(&x, |layer, x| layer.forward(x.clone()))
    }
    /// Apply a projection policy (for example, checkpoint-precision matmuls with F32 gating).
    pub fn forward_with(&self, x: &Array, project: impl Fn(&Linear, &Array) -> Array) -> Array {
        let gated = activations::swiglu(&project(&self.gate, x), &project(&self.up, x));
        project(&self.down, &gated)
    }
}

/// Uniform expert banks in `[experts, out, in]` layout. Split large banks so
/// the flattened iota used by a random draw fits GNX's 32-bit axis limit.
#[derive(Default)]
pub struct ExpertUniform;
impl ExpertUniform {
    fn draw(
        shape: impl gnx_math::IntoShape,
        dtype: DType,
        mut key: impl FnMut() -> Array,
    ) -> Array {
        let info = gnx_math::ArrayInfo::new(shape, dtype);
        let [experts, out, input]: [usize; 3] = gnx_math::fixed_dims(&info)
            .expect("fixed expert bank shape")
            .try_into()
            .expect("expected [experts, out, in]");
        let chunk = u32::MAX as usize / input.checked_mul(out).expect("expert size overflow");
        assert!(chunk > 0, "one expert matrix exceeds the random draw limit");
        let mut draw = |n| random::uniform(&key(), [n, out, input], work(dtype));
        let z = if experts <= chunk {
            draw(experts)
        } else {
            let parts: Vec<_> = (0..experts)
                .step_by(chunk)
                .map(|start| draw(chunk.min(experts - start)))
                .collect();
            lax::concatenate(&parts.iter().collect::<Vec<_>>(), 0)
        };
        ((z - 0.5) * (2. / (input as f64).sqrt())).astype(dtype)
    }
}
impl Initializer for ExpertUniform {
    fn initialize(&self, key: &Array, shape: impl gnx_math::IntoShape, dtype: DType) -> Array {
        let mut keys = std::iter::once(key.clone())
            .chain((1u32..).map(|i| random::fold_in(key, &Array::scalar(i))));
        Self::draw(shape, dtype, || keys.next().unwrap())
    }
    fn initialize_from(&self, rngs: &Rngs, shape: impl gnx_math::IntoShape, dtype: DType) -> Array {
        Self::draw(shape, dtype, || rngs.params())
    }
}

#[derive(Clone, Graph)]
pub struct RoutedExperts {
    pub router: Linear,
    /// Per-expert correction used only for sigmoid top-k selection, not combination weights.
    pub correction: ArrayRef,
    pub gate: Param,
    pub up: Param,
    pub down: Param,
    pub shared: Vec<SwiGlu>,
    pub top_k: usize,
    pub sigmoid: bool,
    pub normalize: bool,
    pub scale: f64,
}
#[derive(Clone, Graph, Serialize, Deserialize)]
pub struct RoutedExpertsConfig {
    pub hidden: usize,
    pub width: usize,
    pub experts: usize,
    pub top_k: usize,
    pub shared: usize,
    pub sigmoid: bool,
    pub dtype: DType,
}
impl ModelConfig for RoutedExpertsConfig {
    type Model = RoutedExperts;
}

/// Initializers for the fields of [`RoutedExperts`].
#[derive(Default)]
pub struct RoutedExpertsInit<
    Router = LinearInit,
    Correction = Fill,
    Gate = ExpertUniform,
    Up = ExpertUniform,
    Down = ExpertUniform,
    Shared = SwiGluInit,
> {
    pub router: Router,
    pub correction: Correction,
    pub gate: Gate,
    pub up: Up,
    pub down: Down,
    pub shared: Shared,
}
impl Model for RoutedExperts {
    type Config = RoutedExpertsConfig;
    type DefaultInit = RoutedExpertsInit;

    fn empty(config: Self::Config) -> Self {
        Self::init(
            config,
            RoutedExpertsInit {
                router: Linear::empty_bundle(),
                correction: Empty,
                gate: Empty,
                up: Empty,
                down: Empty,
                shared: SwiGlu::empty_bundle(),
            },
            &Rngs::default(),
        )
    }
}
impl<Router, Correction, Gate, Up, Down, Shared>
    InitializeWith<RoutedExpertsInit<Router, Correction, Gate, Up, Down, Shared>> for RoutedExperts
where
    Linear: InitializeWith<Router>,
    Correction: Initializer,
    Gate: Initializer,
    Up: Initializer,
    Down: Initializer,
    SwiGlu: InitializeWith<Shared>,
{
    fn init(
        config: RoutedExpertsConfig,
        init: RoutedExpertsInit<Router, Correction, Gate, Up, Down, Shared>,
        rngs: &Rngs,
    ) -> Self {
        let RoutedExpertsConfig {
            hidden,
            width,
            experts,
            top_k,
            shared,
            sigmoid,
            dtype: t,
        } = config;
        assert!(top_k > 0 && top_k <= experts && hidden > 0 && width > 0);
        RoutedExperts {
            router: Linear::init(
                LinearConfig::projection(hidden, experts, t),
                init.router,
                rngs,
            ),
            correction: ArrayRef::new(init.correction.initialize_from(rngs, [experts], work(t))),
            gate: Param::new(init.gate.initialize_from(rngs, [experts, width, hidden], t)),
            up: Param::new(init.up.initialize_from(rngs, [experts, width, hidden], t)),
            down: Param::new(init.down.initialize_from(rngs, [experts, hidden, width], t)),
            shared: (shared > 0)
                .then(|| {
                    SwiGlu::init(
                        SwiGluConfig {
                            hidden,
                            width: shared * width,
                            dtype: t,
                        },
                        init.shared,
                        rngs,
                    )
                })
                .into_iter()
                .collect(),
            top_k,
            sigmoid,
            normalize: true,
            scale: 1.,
        }
    }
}
impl RoutedExperts {
    /// Returns `[tokens,k]` expert indices and combination weights.
    pub fn route(&self, x: &Array) -> (Array, Array) {
        let d = dims(x);
        assert_eq!(d.len(), 2);
        let f = work(x.dtype());
        let logits = project(&x.astype(f), &self.router.weight.value().astype(f));
        let logits = match self.router.bias() {
            Some(bias) => logits + bias.value().astype(f),
            None => logits,
        };
        let scores = if self.sigmoid {
            activations::sigmoid(&logits)
        } else {
            activations::softmax(&logits)
        };
        let choice = if self.sigmoid {
            &scores + self.correction.get()
        } else {
            scores.clone()
        };
        let (_, idx) = lax::top_k(&choice, self.top_k);
        let n = dims(&scores)[1];
        let row = lax::iota([d[0], 1], 0, DType::I32)
            * i32::try_from(n).expect("expert count exceeds I32");
        let weights = scores.reshape([d[0] * n]).at(&idx + row);
        let weights = if self.normalize {
            &weights / sum_keep(&weights, &[1])
        } else {
            weights
        };
        (idx, weights * self.scale)
    }
    pub fn forward(&self, x: Array) -> Array {
        let d = dims(&x);
        let hidden = *d.last().unwrap();
        let tokens = d[..d.len() - 1].iter().product();
        let flat = x.reshape([tokens, hidden]);
        let (ids, weights) = self.route(&flat);
        let k = self.top_k;
        let input = lax::broadcast_in_dim(&flat, [tokens, k, hidden], &[0, 2]);
        let apply = |z: &Array, w: &Param| {
            let w = w.value().at(&ids);
            lax::dot_general(
                z,
                &w,
                lax::DotDimensions {
                    lhs_batch: vec![0, 1],
                    rhs_batch: vec![0, 1],
                    lhs_contracting: vec![2],
                    rhs_contracting: vec![3],
                },
            )
        };
        let a = activations::silu(&apply(&input, &self.gate)) * apply(&input, &self.up);
        let y = apply(&a, &self.down) * weights.reshape([tokens, k, 1]).astype(x.dtype());
        let mut y = lax::reduce_sum(&y, &[1]).reshape(x.shape());
        for expert in &self.shared {
            y = y + expert.forward(x.clone());
        }
        y
    }
}

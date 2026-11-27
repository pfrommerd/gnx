//! Channels-last normalization. Half-precision inputs accumulate statistics in F32.
use crate::{Empty, Fill, InitializeWith, Initializer, Model, ModelConfig, Rngs};
use crate::{Param, tensor::*};
use gnx_graph::Graph;
use gnx_io::{Deserialize, Serialize};
use gnx_math::{Array, ArrayRef, DType, lax};
fn mean(x: &Array, axes: &[usize]) -> Array {
    let d = dims(x);
    let n = axes.iter().map(|&a| d[a]).product::<usize>();
    assert!(n > 0);
    sum_keep(x, axes) * (1. / n as f64)
}
fn normalize(x: &Array, axes: &[usize], eps: f64, center: bool) -> Array {
    let z = if center { x - mean(x, axes) } else { x.clone() };
    let var = mean(&(&z * &z), axes);
    z * lax::rsqrt(&(var + eps))
}
fn input(x: &Array, p: &Param) -> Array {
    assert_eq!(dims(x).last(), dims(&p.value()).first());
    x.astype(work(x.dtype()))
}
#[derive(Clone, Graph)]
pub struct RmsNorm {
    pub scale: Param,
    pub eps: f64,
}
#[derive(Clone, Graph, Serialize, Deserialize)]
pub struct RmsNormConfig {
    pub features: usize,
    pub eps: f64,
    pub dtype: DType,
}
impl ModelConfig for RmsNormConfig {
    type Model = RmsNorm;
}

/// Initializers for the fields of [`RmsNorm`].
#[derive(Default)]
pub struct RmsNormInit<Scale = Fill> {
    pub scale: Scale,
}
impl RmsNorm {
    pub(crate) fn empty_bundle() -> RmsNormInit<Empty> {
        RmsNormInit { scale: Empty }
    }
}

impl Model for RmsNorm {
    type Config = RmsNormConfig;
    type DefaultInit = RmsNormInit;

    fn empty(config: Self::Config) -> Self {
        Self::init(config, Self::empty_bundle(), &Rngs::default())
    }
}
impl<Scale> InitializeWith<RmsNormInit<Scale>> for RmsNorm
where
    Scale: Initializer,
{
    fn init(config: RmsNormConfig, init: RmsNormInit<Scale>, rngs: &Rngs) -> Self {
        let RmsNormConfig {
            features,
            eps,
            dtype: t,
        } = config;
        assert!(features > 0);
        RmsNorm {
            scale: Param::new(init.scale.initialize_from(rngs, [features], t)),
            eps,
        }
    }
}
impl RmsNorm {
    pub fn forward(&self, x: Array) -> Array {
        let z = input(&x, &self.scale);
        let z = normalize(&z, &[dims(&z).len() - 1], self.eps, false);
        z.astype(x.dtype()) * self.scale.value()
    }
}
#[derive(Clone, Graph)]
pub struct LayerNorm {
    pub scale: Param,
    pub bias: Param,
    pub eps: f64,
}
#[derive(Clone, Graph, Serialize, Deserialize)]
pub struct LayerNormConfig {
    pub features: usize,
    pub eps: f64,
    pub dtype: DType,
}
impl ModelConfig for LayerNormConfig {
    type Model = LayerNorm;
}

/// Initializers for the fields of [`LayerNorm`].
#[derive(Default)]
pub struct LayerNormInit<Scale = Fill, Bias = Fill> {
    pub scale: Scale,
    pub bias: Bias,
}
impl Model for LayerNorm {
    type Config = LayerNormConfig;
    type DefaultInit = LayerNormInit;

    fn empty(config: Self::Config) -> Self {
        Self::init(
            config,
            LayerNormInit {
                scale: Empty,
                bias: Empty,
            },
            &Rngs::default(),
        )
    }
}
impl<Scale, Bias> InitializeWith<LayerNormInit<Scale, Bias>> for LayerNorm
where
    Scale: Initializer,
    Bias: Initializer,
{
    fn init(config: LayerNormConfig, init: LayerNormInit<Scale, Bias>, rngs: &Rngs) -> Self {
        let LayerNormConfig {
            features,
            eps,
            dtype: t,
        } = config;
        assert!(features > 0);
        LayerNorm {
            scale: Param::new(init.scale.initialize_from(rngs, [features], t)),
            bias: Param::new(init.bias.initialize_from(rngs, [features], t)),
            eps,
        }
    }
}
impl LayerNorm {
    pub fn forward(&self, x: Array) -> Array {
        let z = input(&x, &self.scale);
        let z = normalize(&z, &[dims(&z).len() - 1], self.eps, true);
        z.astype(x.dtype()) * self.scale.value() + self.bias.value()
    }
}
#[derive(Clone, Graph)]
pub struct GroupNorm {
    pub scale: Param,
    pub bias: Param,
    pub groups: usize,
    pub eps: f64,
}
#[derive(Clone, Graph, Serialize, Deserialize)]
pub struct GroupNormConfig {
    pub features: usize,
    pub groups: usize,
    pub eps: f64,
    pub dtype: DType,
}
impl ModelConfig for GroupNormConfig {
    type Model = GroupNorm;
}

/// Initializers for the fields of [`GroupNorm`].
#[derive(Default)]
pub struct GroupNormInit<Scale = Fill, Bias = Fill> {
    pub scale: Scale,
    pub bias: Bias,
}
impl Model for GroupNorm {
    type Config = GroupNormConfig;
    type DefaultInit = GroupNormInit;

    fn empty(config: Self::Config) -> Self {
        Self::init(
            config,
            GroupNormInit {
                scale: Empty,
                bias: Empty,
            },
            &Rngs::default(),
        )
    }
}
impl<Scale, Bias> InitializeWith<GroupNormInit<Scale, Bias>> for GroupNorm
where
    Scale: Initializer,
    Bias: Initializer,
{
    fn init(config: GroupNormConfig, init: GroupNormInit<Scale, Bias>, rngs: &Rngs) -> Self {
        let GroupNormConfig {
            features,
            groups,
            eps,
            dtype: t,
        } = config;
        assert!(features > 0);
        assert!(groups > 0 && features % groups == 0);
        GroupNorm {
            scale: Param::new(init.scale.initialize_from(rngs, [features], t)),
            bias: Param::new(init.bias.initialize_from(rngs, [features], t)),
            groups,
            eps,
        }
    }
}
impl GroupNorm {
    pub fn forward(&self, x: Array) -> Array {
        let z = input(&x, &self.scale);
        let mut d = dims(&z);
        assert!(d.len() >= 2);
        let c = d.pop().unwrap();
        let mut axes = (1..d.len()).collect::<Vec<_>>();
        d.extend([self.groups, c / self.groups]);
        axes.push(d.len() - 1);
        let z = normalize(&z.reshape(d), &axes, self.eps, true).reshape(x.shape());
        z.astype(x.dtype()) * self.scale.value() + self.bias.value()
    }
}
#[derive(Clone, Graph)]
pub struct BatchNorm {
    pub scale: Param,
    pub bias: Param,
    pub running_mean: ArrayRef,
    pub running_var: ArrayRef,
    pub momentum: f64,
    pub eps: f64,
}
/// `momentum` is the fraction of the previous running statistic retained.
#[derive(Clone, Graph, Serialize, Deserialize)]
pub struct BatchNormConfig {
    pub features: usize,
    pub momentum: f64,
    pub eps: f64,
    pub dtype: DType,
}
impl ModelConfig for BatchNormConfig {
    type Model = BatchNorm;
}

/// Initializers for the fields of [`BatchNorm`].
#[derive(Default)]
pub struct BatchNormInit<Scale = Fill, Bias = Fill, RunningMean = Fill, RunningVar = Fill> {
    pub scale: Scale,
    pub bias: Bias,
    pub running_mean: RunningMean,
    pub running_var: RunningVar,
}

impl Model for BatchNorm {
    type Config = BatchNormConfig;
    type DefaultInit = BatchNormInit;

    fn empty(config: Self::Config) -> Self {
        Self::init(
            config,
            BatchNormInit {
                scale: Empty,
                bias: Empty,
                running_mean: Empty,
                running_var: Empty,
            },
            &Rngs::default(),
        )
    }
}
impl<Scale, Bias, RunningMean, RunningVar>
    InitializeWith<BatchNormInit<Scale, Bias, RunningMean, RunningVar>> for BatchNorm
where
    Scale: Initializer,
    Bias: Initializer,
    RunningMean: Initializer,
    RunningVar: Initializer,
{
    fn init(
        config: BatchNormConfig,
        init: BatchNormInit<Scale, Bias, RunningMean, RunningVar>,
        rngs: &Rngs,
    ) -> Self {
        let BatchNormConfig {
            features,
            momentum,
            eps,
            dtype: t,
        } = config;
        assert!(features > 0);
        assert!((0.0..=1.).contains(&momentum));
        BatchNorm {
            scale: Param::new(init.scale.initialize_from(rngs, [features], t)),
            bias: Param::new(init.bias.initialize_from(rngs, [features], t)),
            running_mean: ArrayRef::new(init.running_mean.initialize_from(
                rngs,
                [features],
                work(t),
            )),
            running_var: ArrayRef::new(init.running_var.initialize_from(rngs, [features], work(t))),
            momentum,
            eps,
        }
    }
}
impl BatchNorm {
    pub fn forward(&self, x: Array, training: bool) -> Array {
        let z = input(&x, &self.scale);
        let (m, v) = if training {
            let d = dims(&z);
            assert!(d.len() >= 2);
            let axes = (0..d.len() - 1).collect::<Vec<_>>();
            let m = mean(&z, &axes);
            let delta = &z - &m;
            let v = mean(&(&delta * &delta), &axes);
            let update = |cell: &ArrayRef, new: &Array| {
                let old = cell.get();
                cell.set(&old * self.momentum + new.reshape(old.shape()) * (1. - self.momentum))
            };
            update(&self.running_mean, &m);
            update(&self.running_var, &v);
            (m, v)
        } else {
            (self.running_mean.get(), self.running_var.get())
        };
        let y = (&z - m) * lax::rsqrt(&(v + self.eps));
        y.astype(x.dtype()) * self.scale.value() + self.bias.value()
    }
}

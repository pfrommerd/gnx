//! Vision transformer from [Image Generation with a Sphere Encoder](https://github.com/facebookresearch/sphere-encoder).
//! Rust/GNX adaptation of the PyTorch model, copyright Meta Platforms, Inc. and affiliates,
//! licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
//!
//! Images are NCHW in `[-1, 1]`; latent patches are in raster order. With a configured
//! [`Manifold`], pass the encoder output directly to the matching decoder. With `None`,
//! the encoder emits raw latents; use [`spherify`] for upstream-style reconstruction.
//!
//! ```no_run
//! use gnx_models::sphere::{BackboneConfig, Encoder, EncoderConfig, Decoder, DecoderConfig, Size, spherify};
//! use gnx_nn::{InitializeWith, Rngs};
//! use gnx_math::Array;
//! # fn example(image: Array) {
//! let rngs = Rngs::default();
//! let backbone = BackboneConfig::preset(Size::Base);
//! let encoder = Encoder::init(EncoderConfig { backbone: backbone.clone() }, (), &rngs);
//! let decoder = Decoder::init(DecoderConfig { backbone, ..Default::default() }, (), &rngs);
//! // Load learned parameters before reconstruction; fresh prediction heads start at zero.
//! let reconstruction = decoder.forward(spherify(&encoder.forward(image, None)), None);
//! # }
//! ```
use gnx_graph::Graph;
use gnx_io::{Deserialize, Serialize};
use gnx_math::{Array, DType, lax, random};
use gnx_nn::{
    Conv, Embedding, InitializeWith, Linear, Model, ModelConfig, Param, RmsNorm, Rngs, RngsExt,
    activations::{silu, swiglu},
    scaled_dot_product_attention,
};

#[derive(
    Clone,
    Copy,
    gnx_graph::GraphHash,
    gnx_graph::GraphPartialEq,
    gnx_graph::GraphEq,
    gnx_graph::Leaf,
    Serialize,
    Deserialize,
)]
pub enum PixelHead {
    Linear,
    Conv,
}

/// Geometry of the encoder output and the corresponding decoder input.
#[derive(
    Clone,
    Copy,
    gnx_graph::GraphHash,
    gnx_graph::GraphPartialEq,
    gnx_graph::GraphEq,
    gnx_graph::Leaf,
    Serialize,
    Deserialize,
)]
pub enum Manifold {
    /// One L2-normalized vector per patch: `[batch, patches, token_channels]`.
    Sphere,
    /// One normalized (sin, cos) pair per channel: `[batch, patches, token_channels, 2]`.
    Torus,
}

#[derive(Clone, Graph, Serialize, Deserialize)]
pub struct BackboneConfig {
    pub image_size: usize,
    pub patch_size: usize,
    pub token_channels: usize,
    #[serde(default)]
    pub manifold: Option<Manifold>,
    pub width: usize,
    pub layers: usize,
    pub heads: usize,
    pub classes: usize,
    pub context_start: usize,
    pub context_tokens: usize,
    pub spherify_blocks: bool,
    pub mixer_depth: usize,
    pub affine_mixer: bool,
    pub dtype: DType,
}
impl Default for BackboneConfig {
    fn default() -> Self {
        Self {
            image_size: 256,
            patch_size: 16,
            token_channels: 16,
            manifold: None,
            width: 768,
            layers: 12,
            heads: 12,
            classes: 0,
            context_start: 4,
            context_tokens: 0,
            spherify_blocks: false,
            mixer_depth: 0,
            affine_mixer: true,
            dtype: DType::F32,
        }
    }
}
/// Upstream backbone presets. Other configuration fields can be overridden independently.
pub enum Size {
    Small,
    Base,
    Large,
    XLarge,
    Huge,
    Giant,
}
impl BackboneConfig {
    fn latent_width(&self) -> usize {
        match self.manifold {
            Some(Manifold::Torus) => 2 * self.token_channels,
            _ => self.token_channels,
        }
    }
    pub fn preset(size: Size) -> Self {
        let (width, layers, heads, context_start) = match size {
            Size::Small => (512, 8, 8, 2),
            Size::Base => (768, 12, 12, 4),
            Size::Large => (1024, 24, 16, 8),
            Size::XLarge => (1152, 28, 16, 8),
            Size::Huge => (1280, 32, 16, 10),
            Size::Giant => (1664, 40, 16, 12),
        };
        Self {
            width,
            layers,
            heads,
            context_start,
            ..Self::default()
        }
    }
}
#[derive(Clone, Default, Graph, Serialize, Deserialize)]
pub struct EncoderConfig {
    pub backbone: BackboneConfig,
}
#[derive(Clone, Graph, Serialize, Deserialize)]
pub struct DecoderConfig {
    pub backbone: BackboneConfig,
    pub pixel_head: PixelHead,
}
impl Default for DecoderConfig {
    fn default() -> Self {
        Self {
            backbone: Default::default(),
            pixel_head: PixelHead::Linear,
        }
    }
}

#[derive(Clone, Graph)]
pub struct Block {
    pub norm1: RmsNorm,
    pub norm2: RmsNorm,
    pub qkv: Linear,
    pub proj: Linear,
    pub w12: Linear,
    pub w3: Linear,
    pub modulation: Option<Linear>,
}
#[derive(Clone, Graph)]
pub struct MixerBlock {
    pub norm1: RmsNorm,
    pub norm2: RmsNorm,
    pub token_in: Linear,
    pub token_out: Linear,
    pub channel_in: Linear,
    pub channel_out: Linear,
    pub alpha1: Param,
    pub alpha2: Param,
}
#[derive(Clone, Graph)]
pub struct Mixer {
    pub blocks: Vec<MixerBlock>,
    pub scale: Option<Param>,
}
#[derive(Clone, Graph)]
pub struct Encoder {
    pub backbone: Backbone,
    pub embed: Linear,
    pub pointwise: Linear,
    pub head: Linear,
    pub mixer: Option<Mixer>,
}
#[derive(Clone, Graph)]
pub struct Decoder {
    pub backbone: Backbone,
    pub embed: Linear,
    pub head: Linear,
    pub mixer: Option<Mixer>,
    pub refine: Conv,
    pub upsample: Option<Conv>,
}
#[derive(Clone, Graph)]
pub struct Backbone {
    pub config: BackboneConfig,
    pub blocks: Vec<Block>,
    pub labels: Option<Embedding>,
    pub context_pos: Option<Param>,
    pub context_embed: Option<Param>,
    pub head_modulation: Option<(RmsNorm, Linear)>,
}

enum Init {
    Uniform(f64),
    Normal(f64),
    Constant(f64),
}
struct Builder<'a> {
    dtype: DType,
    rngs: Option<&'a Rngs>,
}
impl Builder<'_> {
    fn array(&self, shape: impl gnx_math::IntoShape, init: Init) -> Array {
        let Some(r) = self.rngs else {
            return Array::placeholder(gnx_math::ArrayInfo::new(shape, self.dtype));
        };
        match init {
            Init::Uniform(bound) => {
                (random::uniform(&r.params(), shape, DType::F32) * (2. * bound) - bound)
                    .astype(self.dtype)
            }
            Init::Normal(std) => {
                (random::normal(&r.params(), shape, DType::F32) * std).astype(self.dtype)
            }
            Init::Constant(v) => lax::broadcast_in_dim(&lax::scalar(v, self.dtype), shape, &[]),
        }
    }
    fn linear(&self, i: usize, o: usize, bias: bool, init: Init) -> Linear {
        Linear::new(
            self.array([o, i], init),
            bias.then(|| self.array([o], Init::Constant(0.))),
        )
    }
    fn norm(&self, d: usize) -> RmsNorm {
        RmsNorm {
            scale: Param::new(self.array([d], Init::Constant(1.))),
            eps: 1e-6,
        }
    }
    fn conv(&self, i: usize, o: usize, k: usize, bound: f64) -> Conv {
        Conv::new(
            self.array([k, k, i, o], Init::Uniform(bound)),
            Some(self.array([o], Init::Constant(0.))),
            1,
        )
    }
}
fn xavier(i: usize, o: usize) -> Init {
    Init::Uniform((6. / (i + o) as f64).sqrt())
}
macro_rules! sphere_model {
    ($model:ident, $config:ident) => {
        impl ModelConfig for $config {
            type Model = $model;
        }
        impl Model for $model {
            type Config = $config;
            type DefaultInit = ();
            fn empty(config: $config) -> Self {
                Self::build(config, None)
            }
        }
        impl InitializeWith<()> for $model {
            fn init(config: $config, _: (), rngs: &Rngs) -> Self {
                Self::build(config, Some(rngs))
            }
        }
        impl $model {
            /// Evaluation forward. Omitted labels select the null class in a conditioned model.
            pub fn forward(&self, input: Array, labels: Option<&Array>) -> Array {
                let cond = self.backbone.condition(&input, labels, None);
                self.forward_conditioned(input, cond.as_ref())
            }
            /// Training forward with 10% classifier-free label dropout.
            pub fn forward_train(&self, input: Array, labels: &Array, rngs: &Rngs) -> Array {
                let cond = self.backbone.condition(&input, Some(labels), Some(rngs));
                self.forward_conditioned(input, cond.as_ref())
            }
        }
    };
}
sphere_model!(Encoder, EncoderConfig);
sphere_model!(Decoder, DecoderConfig);
impl Backbone {
    fn build(config: BackboneConfig, b: &Builder<'_>) -> Self {
        let c = &config;
        let (d, p, h, z, k) = (
            c.width,
            c.patch_size,
            c.heads,
            c.token_channels,
            c.context_tokens,
        );
        assert!(
            p > 0
                && c.image_size > 1
                && c.image_size % p == 0
                && h > 0
                && d > 0
                && d % h == 0
                && (d / h) % 4 == 0
                && z > 0
                && c.layers > 0
        );
        assert!(matches!(
            c.dtype,
            DType::F32 | DType::BF16 | DType::F16 | DType::F64
        ));
        let linear = |i, o| b.linear(i, o, true, xavier(i, o));
        let zero = |i, o| b.linear(i, o, true, Init::Constant(0.));
        let mlp = d * 8 / 3;
        let cond = c.classes > 0;
        let blocks = (0..c.layers)
            .map(|_| Block {
                norm1: b.norm(d),
                norm2: b.norm(d),
                qkv: linear(d, 3 * d),
                proj: linear(d, d),
                w12: linear(d, 2 * mlp),
                w3: linear(mlp, d),
                modulation: cond.then(|| zero(d, 6 * d)),
            })
            .collect();
        Self {
            blocks,
            labels: cond.then(|| Embedding {
                weight: Param::new(b.array([c.classes + 1, d], Init::Normal(0.02))),
            }),
            context_pos: (k > 0).then(|| Param::new(b.array([1, k, d], Init::Normal(0.02)))),
            context_embed: (k > 0 && !cond)
                .then(|| Param::new(b.array([1, k, d], Init::Normal(0.02)))),
            head_modulation: cond.then(|| (b.norm(d), zero(d, 2 * d))),
            config,
        }
    }
}
impl Mixer {
    fn build(
        c: &BackboneConfig,
        b: &Builder<'_>,
        m: usize,
        init: impl Fn(usize, usize) -> Init,
    ) -> Option<Self> {
        let n = (c.image_size / c.patch_size).pow(2);
        (c.mixer_depth > 0).then(|| {
            let lin = |i, o| b.linear(i, o, true, init(i, o));
            Mixer {
                blocks: (0..c.mixer_depth)
                    .map(|_| MixerBlock {
                        norm1: b.norm(m),
                        norm2: b.norm(m),
                        token_in: lin(n, n),
                        token_out: lin(n, n),
                        channel_in: lin(m, 2 * m),
                        channel_out: lin(2 * m, m),
                        alpha1: Param::new(b.array([1], Init::Constant(0.))),
                        alpha2: Param::new(b.array([1], Init::Constant(0.))),
                    })
                    .collect(),
                scale: c.affine_mixer.then(|| b.norm(m).scale),
            }
        })
    }
}

fn dims<const N: usize>(x: &Array) -> [usize; N] {
    x.fixed_dims()
        .unwrap()
        .try_into()
        .expect("unexpected tensor rank")
}
fn cat(a: &Array, b: &Array, axis: usize) -> Array {
    lax::concatenate(&[a, b], axis)
}
fn swap(x: &Array) -> Array {
    lax::transpose(x, &[0, 2, 1])
}
/// Normalize only the coordinate axis, retaining the token (and torus channel) axes.
fn unit_vector(x: Array) -> Array {
    let mut shape = x.fixed_dims().unwrap();
    let axis = shape.len() - 1;
    shape[axis] = 1;
    let dtype = x.dtype();
    let f = x.astype(if dtype == DType::F64 {
        DType::F64
    } else {
        DType::F32
    });
    let norm_sq = lax::reduce_sum(&(&f * &f), &[axis]).reshape(shape);
    (f * lax::rsqrt(&(norm_sq + 1e-6))).astype(dtype)
}
fn normalized(x: &Array, axes: &[usize]) -> Array {
    let mut shape = x.fixed_dims().unwrap();
    let mut count = 1;
    for &i in axes {
        count *= shape[i];
        shape[i] = 1;
    }
    let t = if x.dtype() == DType::F64 {
        DType::F64
    } else {
        DType::F32
    };
    let f = x.astype(t);
    let ms = lax::reduce_sum(&(&f * &f), axes).reshape(shape) * (1. / count as f64);
    (f * lax::rsqrt(&(ms + 1e-6))).astype(x.dtype())
}
/// Project each example to unit RMS across the entire patch sequence.
pub fn spherify(latents: &Array) -> Array {
    let [_, _, _] = dims(latents);
    normalized(latents, &[1, 2])
}
impl Mixer {
    fn forward(&self, mut x: Array) -> Array {
        for m in &self.blocks {
            let t = swap(&m.norm1.forward(x.clone()));
            let t = m.token_out.forward(silu(&m.token_in.forward(t)));
            x = x + swap(&t) * m.alpha1.value();
            let t = m.channel_in.forward(m.norm2.forward(x.clone()));
            x = x + m.channel_out.forward(silu(&t)) * m.alpha2.value();
        }
        let x = normalized(&x, &[2]);
        match &self.scale {
            Some(p) => x * p.value(),
            None => x,
        }
    }
}
// Absolute positions encode column first, axial rotary phases encode row first.
fn angles(grid: usize, width: usize) -> (Array, Array) {
    let t = DType::F32;
    let q = width / 4;
    let freq = lax::exp(&(lax::iota([q], 0, t) * (-10000f64.ln() / q as f64)));
    let axis = |i| (lax::iota([grid, grid, 1], i, t) * &freq).reshape([grid * grid, q]);
    (axis(0), axis(1))
}
struct Rotary {
    cos: Array,
    sin: Array,
}
impl Rotary {
    fn new(grid: usize, d: usize, prefix: usize, t: DType) -> Self {
        let (y, x) = angles(grid, d);
        let mut phase = cat(&y, &x, 1);
        if prefix > 0 {
            phase = cat(&lax::zeros([prefix, d / 2], DType::F32), &phase, 0)
        }
        let phase = phase.reshape([1, prefix + grid * grid, 1, d / 2, 1]);
        Self {
            cos: lax::cos(&phase).astype(t),
            sin: lax::sin(&phase).astype(t),
        }
    }
    fn apply(&self, x: Array) -> Array {
        let [b, n, h, d] = dims(&x);
        let pairs = x.reshape([b, n, h, d / 2, 2]);
        let rot = cat(
            &(pairs.at((.., .., .., .., 1..2)) * -1.),
            &pairs.at((.., .., .., .., 0..1)),
            4,
        );
        (pairs * &self.cos + rot * &self.sin).reshape([b, n, h, d])
    }
}
fn chunk(x: &Array, i: usize, width: usize) -> Array {
    x.at((.., .., i * width..(i + 1) * width))
}
fn modulate(x: Array, m: Option<&Array>, offset: usize, d: usize) -> Array {
    match m {
        Some(m) => x * (chunk(m, offset + 1, d) + 1.) + chunk(m, offset, d),
        None => x,
    }
}
fn gate(x: Array, m: Option<&Array>, i: usize, d: usize) -> Array {
    match m {
        Some(m) => x * chunk(m, i, d),
        None => x,
    }
}
impl Block {
    fn forward(&self, x: Array, cond: Option<&Array>, rope: &Rotary, h: usize) -> Array {
        let [b, n, d] = dims(&x);
        let hd = d / h;
        let m = self
            .modulation
            .as_ref()
            .map(|l| l.forward(silu(cond.unwrap())));
        let m = m.as_ref();
        let qkv = self
            .qkv
            .forward(modulate(self.norm1.forward(x.clone()), m, 0, d));
        let heads = |i| chunk(&qkv, i, d).reshape([b, n, h, hd]);
        let a = scaled_dot_product_attention(
            &rope.apply(heads(0)),
            &rope.apply(heads(1)),
            &heads(2),
            (hd as f64).sqrt().recip(),
            false,
            None,
        );
        let x = x + gate(self.proj.forward(a.reshape([b, n, d])), m, 2, d);
        let uv = self
            .w12
            .forward(modulate(self.norm2.forward(x.clone()), m, 3, d));
        let w = d * 8 / 3;
        x + gate(
            self.w3
                .forward(swiglu(&chunk(&uv, 0, w), &chunk(&uv, 1, w))),
            m,
            5,
            d,
        )
    }
}
fn reflect(x: Array) -> Array {
    let [_, h, w, _] = dims(&x);
    let y = lax::concatenate(&[&x.at((.., 1..2)), &x, &x.at((.., h - 2..h - 1))], 1);
    lax::concatenate(
        &[&y.at((.., .., 1..2)), &y, &y.at((.., .., w - 2..w - 1))],
        2,
    )
}
// Transposed convolution in NHWC, stored as the spatially reversed HWIO kernel.
fn upsample(x: Array, conv: &Conv, p: usize) -> Array {
    let axes = lax::ConvAxes {
        batch: 0,
        feature: 3,
        spatial: vec![1, 2],
    };
    let pad = p as i64;
    lax::conv_general_dilated(
        &x,
        &conv.kernel.value(),
        lax::ConvParameters {
            input: axes.clone(),
            output: axes,
            kernel: lax::ConvKernelAxes {
                spatial: vec![0, 1],
                input_feature: 2,
                output_feature: 3,
            },
            strides: vec![1; 2],
            padding: vec![(pad, pad); 2],
            lhs_dilation: vec![p; 2],
            rhs_dilation: vec![1; 2],
            feature_groups: 1,
            batch_groups: 1,
        },
    ) + conv.bias().unwrap().value()
}
impl Backbone {
    fn condition(
        &self,
        input: &Array,
        labels: Option<&Array>,
        rngs: Option<&Rngs>,
    ) -> Option<Array> {
        let c = &self.config;
        let b = input.fixed_dims().unwrap()[0];
        self.labels.as_ref().map(|embed| {
            let null;
            let ids = match labels {
                Some(ids) => ids,
                None => {
                    null = lax::broadcast_in_dim(&Array::scalar(c.classes as i32), [b], &[]);
                    &null
                }
            };
            assert_eq!(dims(ids), [b]);
            let dropped;
            let ids = if let Some(r) = rngs {
                let draw = random::uniform(&r.dropout(), [b], DType::F32);
                let drop = lax::lt(
                    &draw,
                    &lax::broadcast_in_dim(&Array::scalar(0.1f32), [b], &[]),
                );
                dropped = lax::select(
                    &drop,
                    &lax::broadcast_in_dim(&lax::scalar(c.classes as f64, ids.dtype()), [b], &[]),
                    ids,
                );
                &dropped
            } else {
                ids
            };
            embed.forward(ids).reshape([b, 1, c.width])
        })
    }

    fn forward(&self, mut x: Array, cond: Option<&Array>) -> Array {
        let c = &self.config;
        let b = x.fixed_dims().unwrap()[0];
        let d = c.width;
        let p = c.patch_size;
        let g = c.image_size / p;
        assert_eq!(cond.is_some(), self.labels.is_some());
        if let Some(y) = cond {
            assert_eq!(dims(y), [b, 1, d]);
        }
        let (row, col) = angles(g, d);
        let pos = lax::concatenate(
            &[
                &lax::sin(&col),
                &lax::cos(&col),
                &lax::sin(&row),
                &lax::cos(&row),
            ],
            1,
        );
        let t = x.dtype();
        x = x + pos.astype(t);
        let mut rope = Rotary::new(g, d / c.heads, 0, t);
        let mut k = 0;
        for (i, block) in self.blocks.iter().enumerate() {
            if i == c.context_start && c.context_tokens > 0 {
                k = c.context_tokens;
                let context = match cond {
                    Some(y) => y.clone(),
                    None => self.context_embed.as_ref().unwrap().value(),
                };
                let context = context + self.context_pos.as_ref().unwrap().value();
                x = cat(
                    &lax::broadcast_in_dim(&context, [b, k, d], &[0, 1, 2]),
                    &x,
                    1,
                );
                rope = Rotary::new(g, d / c.heads, k, t)
            }
            x = block.forward(x, cond, &rope, c.heads);
            if c.spherify_blocks && i + 1 < self.blocks.len() {
                x = if k == 0 {
                    spherify(&x)
                } else {
                    cat(&x.at((.., ..k)), &spherify(&x.at((.., k..))), 1)
                }
            }
        }
        if k > 0 {
            x = x.at((.., k..))
        }
        if let Some((norm, l)) = &self.head_modulation {
            x = modulate(norm.forward(x), Some(&l.forward(silu(cond.unwrap()))), 0, d)
        }
        x
    }
}

impl Encoder {
    fn build(config: EncoderConfig, rngs: Option<&Rngs>) -> Self {
        let c = config.backbone;
        let b = Builder {
            dtype: c.dtype,
            rngs,
        };
        let backbone = Backbone::build(c, &b);
        let c = &backbone.config;
        let d = c.width;
        let pix = 3 * c.patch_size.pow(2);
        Self {
            embed: b.linear(pix, d, false, xavier(pix, d)),
            pointwise: b.linear(d, d, true, xavier(d, d)),
            head: b.linear(d, c.latent_width(), true, Init::Constant(0.)),
            mixer: Mixer::build(c, &b, c.latent_width(), xavier),
            backbone,
        }
    }
    /// Encode NCHW images with an optional precomputed `[batch, 1, width]` class embedding.
    pub fn forward_conditioned(&self, input: Array, cond: Option<&Array>) -> Array {
        let c = &self.backbone.config;
        let [b, channels, h, w] = dims(&input);
        assert_eq!([channels, h, w], [3, c.image_size, c.image_size]);
        let p = c.patch_size;
        let g = h / p;
        let patches = lax::transpose(&input.reshape([b, 3, g, p, g, p]), &[0, 2, 4, 1, 3, 5])
            .reshape([b, g * g, 3 * p * p]);
        let x = self.pointwise.forward(self.embed.forward(patches));
        let x = self.head.forward(self.backbone.forward(x, cond));
        let x = match &self.mixer {
            Some(m) => m.forward(x),
            None => x,
        };
        match c.manifold {
            None => x,
            Some(Manifold::Sphere) => unit_vector(x),
            Some(Manifold::Torus) => unit_vector(x.reshape([b, g * g, c.token_channels, 2])),
        }
    }
}

impl Decoder {
    fn build(config: DecoderConfig, rngs: Option<&Rngs>) -> Self {
        let c = config.backbone;
        let b = Builder {
            dtype: c.dtype,
            rngs,
        };
        let backbone = Backbone::build(c, &b);
        let c = &backbone.config;
        let p = c.patch_size;
        let d = c.width;
        let conv = matches!(config.pixel_head, PixelHead::Conv);
        assert!(!conv || c.image_size / p > 1);
        let ch = if conv { 32 } else { 3 };
        let up = p + 2;
        Self {
            embed: b.linear(c.latent_width(), d, true, Init::Normal(1.)),
            mixer: Mixer::build(c, &b, d, |_, _| Init::Normal(1.)),
            head: b.linear(
                d,
                if conv { ch } else { 3 * p * p },
                true,
                Init::Constant(0.),
            ),
            refine: b.conv(ch, ch, 3, 0.01 * (6. / (10 * ch) as f64).sqrt()),
            upsample: conv
                .then(|| b.conv(ch, 3, up, 0.01 * (6. / (ch + 3 * up * up) as f64).sqrt())),
            backbone,
        }
    }
    /// Decode patch latents with an optional precomputed `[batch, 1, width]` class embedding.
    pub fn forward_conditioned(&self, input: Array, cond: Option<&Array>) -> Array {
        let c = &self.backbone.config;
        let p = c.patch_size;
        let g = c.image_size / p;
        let input = match c.manifold {
            Some(Manifold::Torus) => {
                let [b, n, d, pair] = dims(&input);
                assert_eq!([n, d, pair], [g * g, c.token_channels, 2]);
                input.reshape([b, n, c.latent_width()])
            }
            _ => input,
        };
        let [b, n, z] = dims(&input);
        assert_eq!([n, z], [g * g, c.latent_width()]);
        let x = self.embed.forward(input);
        let x = match &self.mixer {
            Some(m) => m.forward(x),
            None => x,
        };
        let x = self.head.forward(self.backbone.forward(x, cond));
        let image = if let Some(up) = &self.upsample {
            upsample(
                silu(&self.refine.forward(reflect(x.reshape([b, g, g, 32])))),
                up,
                p,
            )
        } else {
            let pixels = lax::transpose(&x.reshape([b, g, g, 3, p, p]), &[0, 1, 4, 2, 5, 3])
                .reshape([b, g * p, g * p, 3]);
            self.refine.forward(reflect(pixels))
        };
        lax::transpose(&lax::tanh(&image), &[0, 3, 1, 2])
    }
}

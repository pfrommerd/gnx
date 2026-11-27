//! Channels-last decoder attention. Positions count tokens, not cache entries.
use crate::{
    CacheKey, KVCache, KvPrefix, Linear, LinearConfig, RmsNorm, RmsNormConfig, activations,
    tensor::*,
};
use crate::{InitializeWith, LinearInit, Model, ModelConfig, RmsNormInit, Rngs};
use gnx_graph::Graph;
use gnx_io::{Deserialize, Serialize};
use gnx_math::{
    Array, DType, Dim, lax,
    ops::bounded::{axis_length, pad_to_bound},
};

fn lane(x: &Array) -> Array {
    let shape = dims(x);
    assert_eq!(shape[0], 1, "map cached attention over sequences");
    x.reshape(&shape[1..])
}

fn singleton_batch(x: &Array) -> Array {
    let mut shape = vec![1];
    shape.extend(dims(x));
    x.reshape(shape)
}

impl KvPrefix {
    /// Attend to this logical history. Queries are `[time, heads, width]`;
    /// causal chunks align to the end of the *prefix*, not the storage bound.
    /// Optional user padding is a one-dimensional keep mask at cache capacity.
    pub fn attend(&self, q: &Array, scale: f64, causal: bool, padding: Option<&Array>) -> Array {
        let padding = padding.map(singleton_batch);
        lane(&attention(
            &singleton_batch(q),
            &singleton_batch(&self.key),
            &singleton_batch(&self.value),
            scale,
            causal,
            Some(&self.len),
            padding.as_ref(),
        ))
    }
}

/// Rotary frequencies, optionally interpolated (linear scaling or YaRN).
#[derive(Clone, Graph, Serialize, Deserialize)]
pub struct Rotary {
    pub theta: f64,
    pub factor: f64,
    pub original_context: usize,
    pub beta_fast: f64,
    pub beta_slow: f64,
    pub magnitude: f64,
}
impl Rotary {
    pub fn new(theta: f64) -> Self {
        Self {
            theta,
            factor: 1.,
            original_context: 0,
            beta_fast: 32.,
            beta_slow: 1.,
            magnitude: 1.,
        }
    }
    /// Input `[batch,time,heads,rotary_dim]`; interleaved pairs for Kimi, split halves for Qwen.
    pub fn apply(&self, x: &Array, position: usize, interleaved: bool) -> Array {
        self.apply_at(x, &lax::scalar(position, work(x.dtype())), interleaved)
    }
    /// Like `apply`, with a scalar start position supplied by the traced computation.
    pub fn apply_at(&self, x: &Array, position: &Array, interleaved: bool) -> Array {
        assert!(position.shape().dims().is_empty());
        let [b, t, h, d] = dims(x).try_into().expect("rotary expects rank four");
        assert!(d > 0 && d % 2 == 0 && self.theta > 1. && self.factor >= 1.);
        let half = d / 2;
        let f = work(x.dtype());
        let i = lax::iota([half], 0, f);
        let freq = lax::exp(&(&i * (-self.theta.ln() / half as f64)));
        let freq = if self.original_context == 0 {
            freq * self.factor.recip()
        } else {
            let cutoff = |beta: f64| {
                half as f64 * (self.original_context as f64 / (std::f64::consts::TAU * beta)).ln()
                    / self.theta.ln()
            };
            let lo = cutoff(self.beta_fast).floor().max(0.);
            let hi = cutoff(self.beta_slow).ceil().min((d - 1) as f64);
            let ramp = (i - lo) * (1. / (hi - lo).max(0.001));
            let zero = constant(0., [half], f);
            let one = constant(1., [half], f);
            let ramp = &one - lax::max(&(&one - lax::max(&ramp, &zero)), &zero);
            freq * (1.0_f64 + ramp * (self.factor.recip() - 1.))
        };
        let angle = (lax::iota([1, t, 1, 1], 1, f) + position.astype(f)) * freq;
        let cos = (lax::cos(&angle) * self.magnitude).astype(x.dtype());
        let sin = (lax::sin(&angle) * self.magnitude).astype(x.dtype());
        let x = if interleaved {
            lax::transpose(&x.reshape([b, t, h, half, 2]), &[0, 1, 2, 4, 3]).reshape([b, t, h, d])
        } else {
            x.clone()
        };
        let a = x.at((.., .., .., ..half));
        let b = x.at((.., .., .., half..));
        lax::concatenate(&[&(&a * &cos - &b * &sin), &(a * sin + b * cos)], 3)
    }
}

/// Scaled attention on `[batch,time,heads,width]`. `padding` is a boolean `[batch,key_time]`
/// keep mask at storage capacity. K and V may share a bounded time axis;
/// softmax and causal alignment use its runtime length.
pub fn scaled_dot_product_attention(
    q: &Array,
    k: &Array,
    v: &Array,
    scale: f64,
    causal: bool,
    padding: Option<&Array>,
) -> Array {
    let ks = k.shape();
    let vs = v.shape();
    assert_eq!(ks.dims()[1], vs.dims()[1], "K/V time axes must agree");
    if matches!(ks.dims()[1], Dim::Bounded { .. }) {
        for shape in [&ks, &vs] {
            assert!(
                shape
                    .dims()
                    .iter()
                    .enumerate()
                    .all(|(i, d)| i == 1 || matches!(d, Dim::Fixed(_))),
                "only K/V time may be bounded"
            );
        }
        let len = axis_length(k, 1);
        attention(
            q,
            &pad_to_bound(k),
            &pad_to_bound(v),
            scale,
            causal,
            Some(&len),
            padding,
        )
    } else {
        attention(q, k, v, scale, causal, None, padding)
    }
}

// Dense lowering for both symbolic inputs and cursor-backed views. The
// contractions use capacity storage; no padding lane participates in softmax.
fn attention(
    q: &Array,
    k: &Array,
    v: &Array,
    scale: f64,
    causal: bool,
    len: Option<&Array>,
    padding: Option<&Array>,
) -> Array {
    let [b, t, h, d] = dims(q).try_into().expect("rank four queries");
    let [bk, s, kv, dk] = dims(k).try_into().expect("rank four keys");
    let [bv, sv, hv, dv] = dims(v).try_into().expect("rank four values");
    assert_eq!((b, s, kv), (bv, sv, hv));
    assert!(b == bk && dk == d && kv > 0 && h % kv == 0 && s > 0);
    let g = h / kv;
    let f = work(q.dtype());
    let dot = |a: &Array, b: &Array, i, j| {
        lax::dot_general(
            a,
            b,
            lax::DotDimensions {
                lhs_batch: vec![0, 2],
                rhs_batch: vec![0, 2],
                lhs_contracting: vec![i],
                rhs_contracting: vec![j],
            },
        )
    };
    let q = q.astype(f).reshape([b, t, kv, g, d]);
    let logits = dot(&q, &k.astype(f), 4, 3) * scale;
    let shape = [b, kv, t, g, s];
    let mut keep = padding.map(|p| {
        assert_eq!(dims(p), [b, s]);
        assert_eq!(p.dtype(), DType::Bool);
        p.clone()
    });
    if let Some(n) = len {
        let occupied = lax::lt(
            &lax::iota([b, s], 1, DType::I32),
            &lax::broadcast_in_dim(n, [b, s], &[]),
        );
        keep = Some(keep.map_or_else(|| occupied.clone(), |p| lax::bitwise_and(&occupied, &p)));
    }
    // A zero probability alone is insufficient when unwritten V contains NaNs.
    let v = keep.as_ref().map_or_else(
        || v.clone(),
        |m| {
            lax::select(
                &lax::broadcast_in_dim(m, v.shape(), &[0, 1]),
                v,
                &lax::zeros(v.shape(), v.dtype()),
            )
        },
    );
    let mut mask = keep.map(|m| lax::broadcast_in_dim(&m, shape, &[0, 4]));
    if causal && !(len.is_some() && t == 1) {
        assert!(t <= s);
        let end = len.map_or_else(|| lax::scalar(s, DType::I64), |n| n.astype(DType::I64));
        let i = lax::iota([t, s], 0, DType::I64) + (end - t as i64 + 1);
        let j = lax::iota([t, s], 1, DType::I64);
        let m = lax::broadcast_in_dim(&lax::lt(&j, &i), shape, &[2, 4]);
        mask = Some(mask.map_or_else(|| m.clone(), |p| lax::bitwise_and(&m, &p)));
    }
    let prob = if let Some(m) = mask {
        // Use a finite floor for the maximum, but exclude masked terms from
        // the sum itself (even a valid logit equal to the floor must work).
        let floor = if f == DType::F64 {
            f64::MIN
        } else {
            f32::MIN as f64
        };
        let masked = lax::select(&m, &logits, &constant(floor, shape, f));
        let max = lax::top_k(&masked, 1).0;
        let weights = lax::select(&m, &lax::exp(&(&masked - &max)), &lax::zeros(shape, f));
        let sum = lax::reduce_sum(&weights, &[4]).reshape(max.shape());
        // A nonempty row has at least one unit weight after subtracting its
        // maximum. Only an empty row needs the denominator of one.
        weights / lax::max(&sum, &constant(1., max.shape(), f))
    } else {
        activations::softmax(&logits)
    };
    let y = lax::dot_general(
        &prob.astype(v.dtype()),
        &v,
        lax::DotDimensions {
            lhs_batch: vec![0, 1],
            rhs_batch: vec![0, 2],
            lhs_contracting: vec![4],
            rhs_contracting: vec![1],
        },
    );
    lax::transpose(&y, &[0, 2, 1, 3, 4]).reshape([b, t, h, dv])
}

#[derive(Clone, Graph)]
pub struct GroupedQueryAttention {
    cache_keys: (CacheKey, CacheKey),
    pub query: Linear,
    pub key: Linear,
    pub value: Linear,
    pub output: Linear,
    pub query_norm: Vec<RmsNorm>,
    pub key_norm: Vec<RmsNorm>,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rotary: Rotary,
}
#[derive(Clone, Graph, Serialize, Deserialize)]
pub struct GroupedQueryAttentionConfig {
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub norm_eps: Option<f64>,
    pub rotary: Rotary,
    pub dtype: DType,
    pub cache_key: CacheKey,
}
impl ModelConfig for GroupedQueryAttentionConfig {
    type Model = GroupedQueryAttention;
}

/// Initializers for the fields of [`GroupedQueryAttention`].
#[derive(Default)]
pub struct GroupedQueryAttentionInit<
    Query = LinearInit,
    Key = LinearInit,
    Value = LinearInit,
    Output = LinearInit,
    QueryNorm = RmsNormInit,
    KeyNorm = RmsNormInit,
> {
    pub query: Query,
    pub key: Key,
    pub value: Value,
    pub output: Output,
    pub query_norm: QueryNorm,
    pub key_norm: KeyNorm,
}
impl Model for GroupedQueryAttention {
    type Config = GroupedQueryAttentionConfig;
    type DefaultInit = GroupedQueryAttentionInit;

    fn empty(config: Self::Config) -> Self {
        Self::init(
            config,
            GroupedQueryAttentionInit {
                query: Linear::empty_bundle(),
                key: Linear::empty_bundle(),
                value: Linear::empty_bundle(),
                output: Linear::empty_bundle(),
                query_norm: RmsNorm::empty_bundle(),
                key_norm: RmsNorm::empty_bundle(),
            },
            &Rngs::default(),
        )
    }
}
impl<Query, Key, Value, Output, QueryNorm, KeyNorm>
    InitializeWith<GroupedQueryAttentionInit<Query, Key, Value, Output, QueryNorm, KeyNorm>>
    for GroupedQueryAttention
where
    Linear: InitializeWith<Query>,
    Linear: InitializeWith<Key>,
    Linear: InitializeWith<Value>,
    Linear: InitializeWith<Output>,
    RmsNorm: InitializeWith<QueryNorm>,
    RmsNorm: InitializeWith<KeyNorm>,
{
    fn init(
        config: GroupedQueryAttentionConfig,
        init: GroupedQueryAttentionInit<Query, Key, Value, Output, QueryNorm, KeyNorm>,
        rngs: &Rngs,
    ) -> Self {
        let GroupedQueryAttentionConfig {
            hidden,
            heads,
            kv_heads: kv,
            head_dim: d,
            norm_eps,
            rotary,
            dtype: t,
            cache_key: key,
        } = config;
        assert!(heads > 0 && kv > 0 && heads % kv == 0 && d > 0 && d % 2 == 0);
        let norm_config = |eps| RmsNormConfig {
            features: d,
            eps,
            dtype: t,
        };
        GroupedQueryAttention {
            query: Linear::init(
                LinearConfig::projection(hidden, heads * d, t),
                init.query,
                rngs,
            ),
            cache_keys: key.split(),
            key: Linear::init(LinearConfig::projection(hidden, kv * d, t), init.key, rngs),
            value: Linear::init(
                LinearConfig::projection(hidden, kv * d, t),
                init.value,
                rngs,
            ),
            output: Linear::init(
                LinearConfig::projection(heads * d, hidden, t),
                init.output,
                rngs,
            ),
            query_norm: norm_eps
                .map(|eps| RmsNorm::init(norm_config(eps), init.query_norm, rngs))
                .into_iter()
                .collect(),
            key_norm: norm_eps
                .map(|eps| RmsNorm::init(norm_config(eps), init.key_norm, rngs))
                .into_iter()
                .collect(),
            heads,
            kv_heads: kv,
            head_dim: d,
            rotary,
        }
    }
}
impl GroupedQueryAttention {
    pub fn init_cache(&self, cache: &mut KVCache) {
        for key in [&self.cache_keys.0, &self.cache_keys.1] {
            cache.init_time(
                key.clone(),
                &[self.kv_heads, self.head_dim],
                self.key.weight.value().dtype(),
            );
        }
    }

    pub fn forward(
        &self,
        x: &Array,
        pos: &Array,
        cache: Option<&KVCache>,
        mask: Option<&Array>,
    ) -> Array {
        let [b, t, _] = dims(x).try_into().expect("expected [batch,time,hidden]");
        let proj = |p: &Linear, h| p.forward(x.clone()).reshape([b, t, h, self.head_dim]);
        let rot = |z: Array, n: &[RmsNorm]| {
            self.rotary.apply_at(
                &n.first()
                    .map_or_else(|| z.clone(), |a| a.forward(z.clone())),
                pos,
                false,
            )
        };
        let q = rot(proj(&self.query, self.heads), &self.query_norm);
        let k = rot(proj(&self.key, self.kv_heads), &self.key_norm);
        let v = proj(&self.value, self.kv_heads);
        let scale = (self.head_dim as f64).sqrt().recip();
        let y = match cache {
            Some(c) => c.write_kv(&self.cache_keys, &lane(&k), &lane(&v)).attend(
                &lane(&q),
                scale,
                true,
                mask.map(lane).as_ref(),
            ),
            None => scaled_dot_product_attention(&q, &k, &v, scale, true, mask),
        };
        self.output
            .forward(y.reshape([b, t, self.heads * self.head_dim]))
    }
}

/// Multi-head latent attention with a shared rotary key and compressed KV cache.
#[derive(Clone, Graph)]
pub struct LatentAttention {
    cache_keys: (CacheKey, CacheKey),
    pub query_down: Linear,
    pub query_norm: RmsNorm,
    pub query: Linear,
    pub kv_down: Linear,
    pub kv_norm: RmsNorm,
    pub kv_up: Linear,
    pub output: Linear,
    pub heads: usize,
    pub rank: usize,
    pub nope: usize,
    pub rope: usize,
    pub value_dim: usize,
    pub rotary: Rotary,
    pub score_scale: f64,
}
#[derive(Clone, Graph, Serialize, Deserialize)]
pub struct LatentAttentionConfig {
    pub hidden: usize,
    pub heads: usize,
    pub query_rank: usize,
    pub rank: usize,
    pub nope: usize,
    pub rope: usize,
    pub value_dim: usize,
    pub eps: f64,
    pub rotary: Rotary,
    pub dtype: DType,
    pub cache_key: CacheKey,
}
impl ModelConfig for LatentAttentionConfig {
    type Model = LatentAttention;
}

/// Initializers for the fields of [`LatentAttention`].
#[derive(Default)]
pub struct LatentAttentionInit<
    QueryDown = LinearInit,
    QueryNorm = RmsNormInit,
    Query = LinearInit,
    KvDown = LinearInit,
    KvNorm = RmsNormInit,
    KvUp = LinearInit,
    Output = LinearInit,
> {
    pub query_down: QueryDown,
    pub query_norm: QueryNorm,
    pub query: Query,
    pub kv_down: KvDown,
    pub kv_norm: KvNorm,
    pub kv_up: KvUp,
    pub output: Output,
}
impl Model for LatentAttention {
    type Config = LatentAttentionConfig;
    type DefaultInit = LatentAttentionInit;

    fn empty(config: Self::Config) -> Self {
        Self::init(
            config,
            LatentAttentionInit {
                query_down: Linear::empty_bundle(),
                query_norm: RmsNorm::empty_bundle(),
                query: Linear::empty_bundle(),
                kv_down: Linear::empty_bundle(),
                kv_norm: RmsNorm::empty_bundle(),
                kv_up: Linear::empty_bundle(),
                output: Linear::empty_bundle(),
            },
            &Rngs::default(),
        )
    }
}
impl<QueryDown, QueryNorm, Query, KvDown, KvNorm, KvUp, Output>
    InitializeWith<LatentAttentionInit<QueryDown, QueryNorm, Query, KvDown, KvNorm, KvUp, Output>>
    for LatentAttention
where
    Linear: InitializeWith<QueryDown>,
    RmsNorm: InitializeWith<QueryNorm>,
    Linear: InitializeWith<Query>,
    Linear: InitializeWith<KvDown>,
    RmsNorm: InitializeWith<KvNorm>,
    Linear: InitializeWith<KvUp>,
    Linear: InitializeWith<Output>,
{
    fn init(
        config: LatentAttentionConfig,
        init: LatentAttentionInit<QueryDown, QueryNorm, Query, KvDown, KvNorm, KvUp, Output>,
        rngs: &Rngs,
    ) -> Self {
        let LatentAttentionConfig {
            hidden,
            heads,
            query_rank,
            rank,
            nope,
            rope,
            value_dim,
            eps,
            rotary,
            dtype: t,
            cache_key: key,
        } = config;
        assert!(rope > 0 && rope % 2 == 0 && nope > 0 && rank > 0);
        LatentAttention {
            query_down: Linear::init(
                LinearConfig::projection(hidden, query_rank, t),
                init.query_down,
                rngs,
            ),
            cache_keys: key.split(),
            query_norm: RmsNorm::init(
                RmsNormConfig {
                    features: query_rank,
                    eps,
                    dtype: t,
                },
                init.query_norm,
                rngs,
            ),
            query: Linear::init(
                LinearConfig::projection(query_rank, heads * (nope + rope), t),
                init.query,
                rngs,
            ),
            kv_down: Linear::init(
                LinearConfig::projection(hidden, rank + rope, t),
                init.kv_down,
                rngs,
            ),
            kv_norm: RmsNorm::init(
                RmsNormConfig {
                    features: rank,
                    eps,
                    dtype: t,
                },
                init.kv_norm,
                rngs,
            ),
            kv_up: Linear::init(
                LinearConfig::projection(rank, heads * (nope + value_dim), t),
                init.kv_up,
                rngs,
            ),
            output: Linear::init(
                LinearConfig::projection(heads * value_dim, hidden, t),
                init.output,
                rngs,
            ),
            heads,
            rank,
            nope,
            rope,
            value_dim,
            rotary,
            score_scale: ((nope + rope) as f64).sqrt().recip(),
        }
    }
}
impl LatentAttention {
    pub fn init_cache(&self, cache: &mut KVCache) {
        let dtype = self.kv_down.weight.value().dtype();
        cache.init_time(self.cache_keys.0.clone(), &[self.rank], dtype);
        cache.init_time(self.cache_keys.1.clone(), &[1, self.rope], dtype);
    }

    pub fn forward(
        &self,
        x: &Array,
        pos: &Array,
        cache: Option<&KVCache>,
        mask: Option<&Array>,
    ) -> Array {
        let [b, t, _] = dims(x).try_into().expect("expected [batch,time,hidden]");
        let h = self.heads;
        let q = self
            .query
            .forward(self.query_norm.forward(self.query_down.forward(x.clone())))
            .reshape([b, t, h, self.nope + self.rope]);
        let q = lax::concatenate(
            &[
                &q.at((.., .., .., ..self.nope)),
                &self
                    .rotary
                    .apply_at(&q.at((.., .., .., self.nope..)), pos, true),
            ],
            3,
        );
        let raw = self.kv_down.forward(x.clone());
        let latent = self.kv_norm.forward(raw.at((.., .., ..self.rank)));
        let pe = self.rotary.apply_at(
            &raw.at((.., .., self.rank..)).reshape([b, t, 1, self.rope]),
            pos,
            true,
        );
        let (latent, pe, len) = match cache {
            Some(c) => {
                let prefix = c.write_kv(&self.cache_keys, &lane(&latent), &lane(&pe));
                (
                    singleton_batch(&prefix.key),
                    singleton_batch(&prefix.value),
                    Some(prefix.len),
                )
            }
            None => (latent, pe, None),
        };
        let s = dims(&latent)[1];
        let kv = self
            .kv_up
            .forward(latent)
            .reshape([b, s, h, self.nope + self.value_dim]);
        let k = lax::concatenate(
            &[
                &kv.at((.., .., .., ..self.nope)),
                &lax::broadcast_in_dim(&pe, [b, s, h, self.rope], &[0, 1, 2, 3]),
            ],
            3,
        );
        let v = kv.at((.., .., .., self.nope..));
        self.output.forward(
            attention(&q, &k, &v, self.score_scale, true, len.as_ref(), mask).reshape([
                b,
                t,
                h * self.value_dim,
            ]),
        )
    }
}

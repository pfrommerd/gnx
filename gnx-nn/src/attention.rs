//! Channels-last decoder attention. Positions count tokens, not cache entries.
use crate::{Linear, RmsNorm, Rngs, activations, tensor::*};
use gnx_graph::Graph;
use gnx_math::{Array, DType, lax};

/// Caller-owned history for one layer. MLA stores compressed latents and rotary keys.
#[derive(Default, Clone)]
pub struct AttentionCache {
    history: Option<(Array, Array)>,
    next_position: usize,
}
impl AttentionCache {
    pub fn len(&self) -> usize {
        self.history.as_ref().map_or(0, |(a, _)| dims(a)[1])
    }
    pub fn is_empty(&self) -> bool {
        self.history.is_none()
    }
    fn append(&mut self, a: Array, b: Array, pos: usize) -> (Array, Array) {
        let t = dims(&a)[1];
        let pair = if let Some((x, y)) = &self.history {
            assert_eq!(pos, self.next_position, "discontinuous cache positions");
            (lax::concatenate(&[x, &a], 1), lax::concatenate(&[y, &b], 1))
        } else {
            (a, b)
        };
        self.next_position = pos + t;
        self.history = Some(pair.clone());
        pair
    }
}

/// Rotary frequencies, optionally interpolated (linear scaling or YaRN).
#[derive(Clone, Graph)]
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
/// keep mask. A causal query chunk is aligned to the end of the key sequence.
pub fn scaled_dot_product_attention(
    q: &Array,
    k: &Array,
    v: &Array,
    scale: f64,
    causal: bool,
    padding: Option<&Array>,
) -> Array {
    let [b, t, h, d] = dims(q).try_into().expect("rank four queries");
    let [bk, s, kv, dk] = dims(k).try_into().expect("rank four keys");
    let [bv, sv, hv, dv] = dims(v).try_into().expect("rank four values");
    assert_eq!((b, s, kv), (bv, sv, hv));
    assert!(b == bk && dk == d && kv > 0 && h % kv == 0 && s > 0 && (!causal || t <= s));
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
    let mut mask = padding.map(|p| {
        assert_eq!(dims(p), [b, s]);
        assert_eq!(p.dtype(), DType::Bool);
        lax::broadcast_in_dim(p, shape, &[0, 4])
    });
    if causal {
        let i = lax::iota([t, s], 0, DType::I64) + (s - t + 1) as u64;
        let j = lax::iota([t, s], 1, DType::I64);
        let m = lax::broadcast_in_dim(&lax::lt(&j, &i), shape, &[2, 4]);
        mask = Some(mask.map_or_else(|| m.clone(), |p| lax::bitwise_and(&m, &p)));
    }
    let prob = if let Some(m) = mask {
        // A finite floor keeps even fully masked rows well-defined before zeroing them.
        let floor = if f == DType::F64 {
            f64::MIN
        } else {
            f32::MIN as f64
        };
        let masked = lax::select(&m, &logits, &constant(floor, shape, f));
        let p = activations::softmax(&masked);
        lax::select(&m, &p, &lax::zeros(shape, f))
    } else {
        activations::softmax(&logits)
    };
    let y = lax::dot_general(
        &prob.astype(v.dtype()),
        v,
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
impl GroupedQueryAttention {
    pub fn init(
        hidden: usize,
        heads: usize,
        kv: usize,
        d: usize,
        norm_eps: Option<f64>,
        rotary: Rotary,
        t: DType,
        r: &Rngs,
    ) -> Self {
        assert!(heads > 0 && kv > 0 && heads % kv == 0 && d > 0 && d % 2 == 0);
        let p = |o| Linear::init_with(hidden, o, false, t, r);
        let norm = || {
            norm_eps
                .map(|e| RmsNorm::init(d, e, t))
                .into_iter()
                .collect()
        };
        Self {
            query: p(heads * d),
            key: p(kv * d),
            value: p(kv * d),
            output: Linear::init_with(heads * d, hidden, false, t, r),
            query_norm: norm(),
            key_norm: norm(),
            heads,
            kv_heads: kv,
            head_dim: d,
            rotary,
        }
    }
    pub fn forward(
        &self,
        x: &Array,
        pos: usize,
        cache: Option<&mut AttentionCache>,
        mask: Option<&Array>,
    ) -> Array {
        let [b, t, _] = dims(x).try_into().expect("expected [batch,time,hidden]");
        let proj = |p: &Linear, h| p.forward(x.clone()).reshape([b, t, h, self.head_dim]);
        let rot = |z: Array, n: &[RmsNorm]| {
            self.rotary.apply(
                &n.first()
                    .map_or_else(|| z.clone(), |a| a.forward(z.clone())),
                pos,
                false,
            )
        };
        let q = rot(proj(&self.query, self.heads), &self.query_norm);
        let k = rot(proj(&self.key, self.kv_heads), &self.key_norm);
        let v = proj(&self.value, self.kv_heads);
        let (k, v) = match cache {
            Some(c) => c.append(k, v, pos),
            None => (k, v),
        };
        self.output.forward(
            scaled_dot_product_attention(
                &q,
                &k,
                &v,
                (self.head_dim as f64).sqrt().recip(),
                true,
                mask,
            )
            .reshape([b, t, self.heads * self.head_dim]),
        )
    }
}

/// Multi-head latent attention with a shared rotary key and compressed KV cache.
#[derive(Clone, Graph)]
pub struct LatentAttention {
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
impl LatentAttention {
    pub fn init(
        hidden: usize,
        heads: usize,
        query_rank: usize,
        rank: usize,
        nope: usize,
        rope: usize,
        value_dim: usize,
        eps: f64,
        rotary: Rotary,
        t: DType,
        r: &Rngs,
    ) -> Self {
        assert!(rope > 0 && rope % 2 == 0 && nope > 0 && rank > 0);
        let p = |i, o| Linear::init_with(i, o, false, t, r);
        Self {
            query_down: p(hidden, query_rank),
            query_norm: RmsNorm::init(query_rank, eps, t),
            query: p(query_rank, heads * (nope + rope)),
            kv_down: p(hidden, rank + rope),
            kv_norm: RmsNorm::init(rank, eps, t),
            kv_up: p(rank, heads * (nope + value_dim)),
            output: p(heads * value_dim, hidden),
            heads,
            rank,
            nope,
            rope,
            value_dim,
            rotary,
            score_scale: ((nope + rope) as f64).sqrt().recip(),
        }
    }
    pub fn forward(
        &self,
        x: &Array,
        pos: usize,
        cache: Option<&mut AttentionCache>,
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
                    .apply(&q.at((.., .., .., self.nope..)), pos, true),
            ],
            3,
        );
        let raw = self.kv_down.forward(x.clone());
        let latent = self.kv_norm.forward(raw.at((.., .., ..self.rank)));
        let pe = self.rotary.apply(
            &raw.at((.., .., self.rank..)).reshape([b, t, 1, self.rope]),
            pos,
            true,
        );
        let (latent, pe) = match cache {
            Some(c) => c.append(latent, pe, pos),
            None => (latent, pe),
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
            scaled_dot_product_attention(&q, &k, &v, self.score_scale, true, mask).reshape([
                b,
                t,
                h * self.value_dim,
            ]),
        )
    }
}

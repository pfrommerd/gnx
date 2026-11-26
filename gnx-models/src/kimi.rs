//! Kimi K2 text decoder with low-rank Q/KV projections, YaRN and sigmoid-routed experts.
use crate::{DecoderCache, FeedForward};
use gnx_graph::Graph;
use gnx_math::{Array, DType};
use gnx_nn::*;
#[derive(Clone)]
pub struct KimiK2Config {
    pub vocab: usize,
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub query_rank: usize,
    pub kv_rank: usize,
    pub nope: usize,
    pub rope: usize,
    pub value_dim: usize,
    pub intermediate: usize,
    pub dense_layers: usize,
    pub experts: usize,
    pub expert_width: usize,
    pub shared: usize,
    pub top_k: usize,
    pub routed_scale: f64,
    pub eps: f64,
    pub rotary: Rotary,
    /// Multiplier on the usual inverse square-root attention scale.
    pub attention_scale: f64,
    pub tied: bool,
}
impl Default for KimiK2Config {
    /// Kimi-K2-Base dimensions. Weights are initialized in the caller's floating dtype.
    fn default() -> Self {
        Self {
            vocab: 163840,
            hidden: 7168,
            layers: 61,
            heads: 64,
            query_rank: 1536,
            kv_rank: 512,
            nope: 128,
            rope: 64,
            value_dim: 128,
            intermediate: 18432,
            dense_layers: 1,
            experts: 384,
            expert_width: 2048,
            shared: 1,
            top_k: 8,
            routed_scale: 2.827,
            eps: 1e-6,
            rotary: Rotary {
                theta: 50000.,
                factor: 32.,
                original_context: 4096,
                beta_fast: 1.,
                beta_slow: 1.,
                magnitude: 1.,
            },
            attention_scale: (1. + 0.1 * 32_f64.ln()).powi(2),
            tied: false,
        }
    }
}
#[derive(Clone, Graph)]
pub struct KimiK2Block {
    pub attention_norm: RmsNorm,
    pub attention: LatentAttention,
    pub mlp_norm: RmsNorm,
    pub mlp: FeedForward,
}
#[derive(Clone, Graph)]
pub struct KimiK2LM {
    pub embedding: Embedding,
    pub blocks: Vec<KimiK2Block>,
    pub norm: RmsNorm,
    pub head: Linear,
}
impl KimiK2LM {
    pub fn init(c: &KimiK2Config, t: DType, r: &Rngs) -> Self {
        assert!(c.layers > 0 && c.dense_layers <= c.layers);
        let embedding = Embedding::init(c.vocab, c.hidden, t, r);
        let norm = || RmsNorm::init(c.hidden, c.eps, t);
        let blocks = (0..c.layers)
            .map(|i| {
                let mut attention = LatentAttention::init(
                    c.hidden,
                    c.heads,
                    c.query_rank,
                    c.kv_rank,
                    c.nope,
                    c.rope,
                    c.value_dim,
                    c.eps,
                    c.rotary.clone(),
                    t,
                    r,
                );
                attention.score_scale *= c.attention_scale;
                let mlp = if i < c.dense_layers {
                    FeedForward::dense(SwiGlu::init(c.hidden, c.intermediate, t, r))
                } else {
                    let mut m = RoutedExperts::init(
                        c.hidden,
                        c.expert_width,
                        c.experts,
                        c.top_k,
                        c.shared,
                        true,
                        t,
                        r,
                    );
                    m.scale = c.routed_scale;
                    FeedForward::routed(m)
                };
                KimiK2Block {
                    attention_norm: norm(),
                    mlp_norm: norm(),
                    attention,
                    mlp,
                }
            })
            .collect();
        let head = if c.tied {
            embedding.tied_head()
        } else {
            Linear::init_with(c.hidden, c.vocab, false, t, r)
        };
        Self {
            embedding,
            blocks,
            norm: norm(),
            head,
        }
    }
    pub fn forward(&self, ids: &Array) -> Array {
        self.forward_at(ids, 0, None, None)
    }
    pub fn forward_at(
        &self,
        ids: &Array,
        position: usize,
        cache: Option<&mut DecoderCache>,
        padding: Option<&Array>,
    ) -> Array {
        let mut x = self.embedding.forward(ids);
        let mut layers = cache.map(|c| c.prepare(self.blocks.len()));
        for (i, b) in self.blocks.iter().enumerate() {
            let a = b.attention.forward(
                &b.attention_norm.forward(x.clone()),
                position,
                layers.as_mut().map(|l| &mut l[i]),
                padding,
            );
            x = x + a;
            let y = b.mlp.forward(b.mlp_norm.forward(x.clone()));
            x = x + y;
        }
        self.head.forward(self.norm.forward(x))
    }
}

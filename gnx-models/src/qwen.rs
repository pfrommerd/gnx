//! Qwen3 dense and Qwen3-MoE causal language models.
use crate::{DecoderCache, FeedForward};
use gnx_graph::Graph;
use gnx_math::{Array, DType};
use gnx_nn::*;
#[derive(Clone)]
pub struct Qwen3Config {
    pub vocab: usize,
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub eps: f64,
    pub rotary: Rotary,
    pub tied: bool,
    pub experts: usize,
    pub expert_width: usize,
    pub top_k: usize,
    pub sparse_step: usize,
    pub mlp_only_layers: Vec<usize>,
    pub normalize_top_k: bool,
}
impl Default for Qwen3Config {
    /// Qwen3-0.6B.
    fn default() -> Self {
        Self {
            vocab: 151936,
            hidden: 1024,
            layers: 28,
            heads: 16,
            kv_heads: 8,
            head_dim: 128,
            intermediate: 3072,
            eps: 1e-6,
            rotary: Rotary::new(1e6),
            tied: true,
            experts: 0,
            expert_width: 0,
            top_k: 0,
            sparse_step: 1,
            mlp_only_layers: vec![],
            normalize_top_k: true,
        }
    }
}
impl Qwen3Config {
    pub fn moe_30b_a3b() -> Self {
        Self {
            hidden: 2048,
            layers: 48,
            heads: 32,
            kv_heads: 4,
            intermediate: 6144,
            tied: false,
            experts: 128,
            expert_width: 768,
            top_k: 8,
            ..Self::default()
        }
    }
}
#[derive(Clone, Graph)]
pub struct Qwen3Block {
    pub attention_norm: RmsNorm,
    pub attention: GroupedQueryAttention,
    pub mlp_norm: RmsNorm,
    pub mlp: FeedForward,
}
#[derive(Clone, Graph)]
pub struct Qwen3LM {
    pub embedding: Embedding,
    pub blocks: Vec<Qwen3Block>,
    pub norm: RmsNorm,
    pub head: Linear,
}
impl Qwen3LM {
    pub fn init(c: &Qwen3Config, t: DType, r: &Rngs) -> Self {
        assert!(c.layers > 0 && c.sparse_step > 0);
        let embedding = Embedding::init(c.vocab, c.hidden, t, r);
        let norm = || RmsNorm::init(c.hidden, c.eps, t);
        let blocks = (0..c.layers)
            .map(|i| {
                let attention = GroupedQueryAttention::init(
                    c.hidden,
                    c.heads,
                    c.kv_heads,
                    c.head_dim,
                    Some(c.eps),
                    c.rotary.clone(),
                    t,
                    r,
                );
                let mlp = if c.experts > 0
                    && (i + 1) % c.sparse_step == 0
                    && !c.mlp_only_layers.contains(&i)
                {
                    let mut m = RoutedExperts::init(
                        c.hidden,
                        c.expert_width,
                        c.experts,
                        c.top_k,
                        0,
                        false,
                        t,
                        r,
                    );
                    m.normalize = c.normalize_top_k;
                    FeedForward::routed(m)
                } else {
                    FeedForward::dense(SwiGlu::init(c.hidden, c.intermediate, t, r))
                };
                Qwen3Block {
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

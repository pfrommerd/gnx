//! Composable text decoders. Inputs are integer `[batch, time]` token IDs;
//! outputs are floating-point `[batch, time, vocab]` logits.
pub mod kimi;
pub mod qwen;
use gnx_graph::Graph;
use gnx_math::Array;
use gnx_nn::{AttentionCache, RoutedExperts, SwiGlu};
pub use kimi::{KimiK2Config, KimiK2LM};
pub use qwen::{Qwen3Config, Qwen3LM};

/// Start each independent sequence with a fresh cache. `position` in `forward_at`
/// is the absolute position of the first input token (also for chunked prefill).
#[derive(Default, Clone)]
pub struct DecoderCache {
    layers: Vec<AttentionCache>,
}
impl DecoderCache {
    pub fn len(&self) -> usize {
        self.layers.first().map_or(0, AttentionCache::len)
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn prepare(&mut self, n: usize) -> &mut [AttentionCache] {
        if self.layers.is_empty() {
            self.layers.resize_with(n, Default::default);
        }
        assert_eq!(self.layers.len(), n, "cache/model depth mismatch");
        &mut self.layers
    }
}

/// Exactly one branch is present. Both branches participate in GNX graph traversal.
#[derive(Clone, Graph)]
pub struct FeedForward {
    dense: Vec<SwiGlu>,
    routed: Vec<RoutedExperts>,
}
impl FeedForward {
    fn dense(m: SwiGlu) -> Self {
        Self {
            dense: vec![m],
            routed: vec![],
        }
    }
    fn routed(m: RoutedExperts) -> Self {
        Self {
            dense: vec![],
            routed: vec![m],
        }
    }
    pub fn dense_layer(&self) -> Option<&SwiGlu> {
        self.dense.first()
    }
    pub fn experts(&self) -> Option<&RoutedExperts> {
        self.routed.first()
    }
    pub fn forward(&self, x: Array) -> Array {
        match self.dense.first() {
            Some(m) => m.forward(x),
            None => self.routed[0].forward(x),
        }
    }
}

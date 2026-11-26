use crate::{Linear, Param, Rngs, RngsExt, tensor::*};
use gnx_graph::Graph;
use gnx_math::{Array, DType, random};
/// Token embedding with rows `[vocabulary, hidden]`. IDs must be in range.
#[derive(Clone, Graph)]
pub struct Embedding {
    pub weight: Param,
}
impl Embedding {
    pub fn init(vocab: usize, hidden: usize, t: DType, r: &Rngs) -> Self {
        assert!(vocab > 0 && hidden > 0);
        let w = random::normal(&r.params(), [vocab, hidden], work(t)) * 0.02;
        Self {
            weight: Param::new(w.astype(t)),
        }
    }
    pub fn forward(&self, ids: &Array) -> Array {
        self.weight.value().at(ids)
    }
    pub fn tied_head(&self) -> Linear {
        Linear::from_params(self.weight.clone(), None)
    }
}

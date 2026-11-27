use crate::{Empty, InitializeWith, Initializer, Model, ModelConfig, Rngs};
use crate::{Linear, Param, tensor::*};
use gnx_graph::Graph;
use gnx_io::{Deserialize, Serialize};
use gnx_math::{Array, DType, IntoShape, random};
/// Normal embedding weights with standard deviation 0.02.
#[derive(Default)]
pub struct EmbeddingNormal;
impl Initializer for EmbeddingNormal {
    fn initialize(&self, key: &Array, shape: impl IntoShape, dtype: DType) -> Array {
        (random::normal(key, shape, work(dtype)) * 0.02).astype(dtype)
    }
}

/// Token embedding with rows `[vocabulary, hidden]`. IDs must be in range.
#[derive(Clone, Graph)]
pub struct Embedding {
    pub weight: Param,
}
#[derive(Clone, Graph, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    pub vocab: usize,
    pub hidden: usize,
    pub dtype: DType,
}
impl ModelConfig for EmbeddingConfig {
    type Model = Embedding;
}

/// Initializers for the fields of [`Embedding`].
#[derive(Default)]
pub struct EmbeddingInit<Weight = EmbeddingNormal> {
    pub weight: Weight,
}
impl Model for Embedding {
    type Config = EmbeddingConfig;
    type DefaultInit = EmbeddingInit;

    fn empty(config: Self::Config) -> Self {
        Self::init(config, EmbeddingInit { weight: Empty }, &Rngs::default())
    }
}
impl<Weight> InitializeWith<EmbeddingInit<Weight>> for Embedding
where
    Weight: Initializer,
{
    fn init(config: EmbeddingConfig, init: EmbeddingInit<Weight>, rngs: &Rngs) -> Self {
        let EmbeddingConfig {
            vocab,
            hidden,
            dtype,
        } = config;
        assert!(vocab > 0 && hidden > 0);
        let weight = init.weight.initialize_from(rngs, [vocab, hidden], dtype);
        Embedding {
            weight: Param::new(weight),
        }
    }
}
impl Embedding {
    pub fn forward(&self, ids: &Array) -> Array {
        self.weight.value().at(ids)
    }
    pub fn tied_head(&self) -> Linear {
        Linear::from_params(self.weight.clone(), None)
    }
}

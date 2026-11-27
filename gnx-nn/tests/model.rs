use gnx_io::{
    Deserialize,
    json::{JsonParser, to_string},
    util::StrSource,
};
use gnx_math::DType;
use gnx_nn::*;

#[test]
fn load_attention_layout_from_json() {
    let config = GroupedQueryAttentionConfig {
        hidden: 8,
        heads: 4,
        kv_heads: 2,
        head_dim: 2,
        norm_eps: Some(1e-5),
        rotary: Rotary::new(10000.),
        dtype: DType::BF16,
        cache_key: CacheKey::root().split().1,
    };
    let json = to_string(&config).unwrap();
    let restored = GroupedQueryAttentionConfig::deserialize(&mut JsonParser::new(StrSource::from(
        json.as_str(),
    )))
    .unwrap();
    assert_eq!(to_string(&restored).unwrap(), json);
    let empty = GroupedQueryAttention::empty(restored);
    fn initialized<C: ModelConfig>(config: C, rngs: &Rngs) -> C::Model {
        C::Model::init(config, <C::Model as Model>::DefaultInit::default(), rngs)
    }
    let initialized = initialized(config, &Rngs::default());
    for (a, b) in [
        (&empty.query, &initialized.query),
        (&empty.key, &initialized.key),
        (&empty.value, &initialized.value),
        (&empty.output, &initialized.output),
    ] {
        assert_eq!(a.weight.value().shape(), b.weight.value().shape());
        assert_eq!(a.weight.value().dtype(), DType::BF16);
        assert!(a.weight.value().tracer().trace_ref().is_placeholder());
        assert!(!b.weight.value().tracer().trace_ref().is_placeholder());
        assert!(a.bias.is_none());
    }
    assert_eq!(empty.key.weight.value().fixed_dims().unwrap(), [4, 8]);
    assert_eq!(empty.query_norm.len(), 1);
    assert_eq!(empty.key_norm[0].eps, 1e-5);
}

#[test]
fn nested_initializer_accepts_distinct_capturing_closures() {
    use gnx_math::{Array, Shape, lax};
    use std::cell::Cell;

    let calls = Cell::new(0);
    let gate = LinearInit {
        weight: |key: &Array, shape: Shape, dtype: DType| {
            calls.set(calls.get() + 1);
            assert!(key.dtype().is_key());
            assert_eq!(shape, Shape::fixed([6, 4]));
            assert_eq!(dtype, DType::F32);
            lax::zeros(shape, dtype)
        },
        bias: |_: &Array, _: Shape, _: DType| -> Array {
            panic!("bias-free projections must skip the bias initializer")
        },
    };
    let model = SwiGlu::init(
        SwiGluConfig {
            hidden: 4,
            width: 6,
            dtype: DType::F32,
        },
        SwiGluInit {
            gate,
            up: LinearInit {
                weight: Empty,
                bias: Empty,
            },
            down: <Linear as Model>::DefaultInit::default(),
        },
        &Rngs::default(),
    );
    assert_eq!(calls.get(), 1);
    assert!(
        model
            .up
            .weight
            .value()
            .tracer()
            .trace_ref()
            .is_placeholder()
    );
    assert!(
        !model
            .down
            .weight
            .value()
            .tracer()
            .trace_ref()
            .is_placeholder()
    );
}

#[test]
fn empty_batch_norm_keeps_accumulation_precision() {
    let layer = BatchNorm::empty(BatchNormConfig {
        features: 3,
        eps: 1e-5,
        momentum: 0.9,
        dtype: DType::F16,
    });
    assert_eq!(layer.scale.value().dtype(), DType::F16);
    assert_eq!(layer.running_var.get().dtype(), DType::F32);
    assert_eq!(layer.running_mean.get().fixed_dims().unwrap(), [3]);
}

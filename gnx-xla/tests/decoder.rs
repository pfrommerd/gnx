use futures_executor::block_on;
use gnx::{Array, Callable, Eval, jit, lax, nn::*, random::Algorithm};
use gnx_base::Device as _;
use gnx_math::{DType::F32, ExecOpts, TransferPolicy};
use gnx_xla::XlaDevice;
fn device() -> XlaDevice {
    XlaDevice::cpu(gnx_base::cache::Cache::in_memory()).unwrap()
}
fn rng() -> Rngs {
    Rngs::key(42, Algorithm::ThreeFry2x32)
}
fn read(d: &XlaDevice, x: Array) -> Vec<f32> {
    let x = x.astype(F32);
    block_on(d.eval(
        &x,
        ExecOpts {
            transfers: TransferPolicy::AllowCopy,
        },
    ))
    .unwrap();
    block_on(d.download(x.data().unwrap()))
        .unwrap()
        .bytes()
        .chunks_exact(4)
        .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
        .collect()
}
fn close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!((x - y).abs() < 2e-5, "element {i}: {x} != {y}")
    }
}
fn c(v: f64) -> Array {
    lax::scalar(v, F32)
}
#[test]
fn grouped_conv_and_norms() {
    let d = device();
    let conv = Conv::init(
        ConvConfig {
            input: 2,
            output: 2,
            kernel: vec![2],
            groups: 2,
            bias: false,
            dtype: F32,
        },
        <Conv as Model>::DefaultInit::default(),
        &rng(),
    );
    conv.kernel
        .set(lax::broadcast_in_dim(&c(1.), [2, 1, 2], &[]));
    let x = lax::iota([6], 0, F32).reshape([1, 3, 2]);
    close(&read(&d, conv.forward(x.clone())), &[2., 4., 6., 8.]);
    let biased = Conv::new(conv.kernel.value(), Some(lax::iota([2], 0, F32)), 2);
    close(&read(&d, biased.forward(x.clone())), &[2., 5., 6., 9.]);
    close(
        &read(
            &d,
            LayerNorm::init(
                LayerNormConfig {
                    features: 2,
                    eps: 0.,
                    dtype: F32,
                },
                <LayerNorm as Model>::DefaultInit::default(),
                &rng(),
            )
            .forward(x.clone()),
        ),
        &[-1., 1., -1., 1., -1., 1.],
    );
    let group = GroupNorm::init(
        GroupNormConfig {
            features: 2,
            groups: 2,
            eps: 0.,
            dtype: F32,
        },
        <GroupNorm as Model>::DefaultInit::default(),
        &rng(),
    );
    let a = (1.5_f32).sqrt();
    close(&read(&d, group.forward(x.clone())), &[-a, -a, 0., 0., a, a]);
    let norm = BatchNorm::init(
        BatchNormConfig {
            features: 2,
            momentum: 0.5,
            eps: 0.,
            dtype: F32,
        },
        <BatchNorm as Model>::DefaultInit::default(),
        &rng(),
    );
    let train = jit({
        let norm = norm.clone();
        move |x: Array| norm.forward(x, true)
    });
    close(
        &read(&d, train.invoke((x.clone(),))),
        &[-a, -a, 0., 0., a, a],
    );
    close(&read(&d, norm.running_mean.get()), &[1., 1.5]);
    let _ = read(&d, train.invoke((x.clone(),)));
    close(&read(&d, norm.running_mean.get()), &[1.5, 2.25]);
    let _ = read(&d, norm.forward(x, false));
    close(&read(&d, norm.running_mean.get()), &[1.5, 2.25]);
}
#[test]
fn attention_masks_and_router_selection() {
    let d = device();
    let q = lax::zeros([1, 2, 2, 2], F32);
    let k = lax::zeros([1, 3, 1, 2], F32);
    let v = lax::iota([1, 3, 1, 1], 1, F32);
    close(
        &read(&d, scaled_dot_product_attention(&q, &k, &v, 1., true, None)),
        &[0.5, 0.5, 1., 1.],
    );
    let keep = lax::lt(&lax::iota([1, 3], 1, F32), &lax::zeros([1, 3], F32));
    close(
        &read(
            &d,
            scaled_dot_product_attention(&q, &k, &v, 1., true, Some(&keep)),
        ),
        &[0.; 4],
    );
    let mut m = RoutedExperts::init(
        RoutedExpertsConfig {
            hidden: 2,
            width: 3,
            experts: 3,
            top_k: 2,
            shared: 0,
            sigmoid: true,
            dtype: F32,
        },
        <RoutedExperts as Model>::DefaultInit::default(),
        &rng(),
    );
    m.normalize = false;
    m.router
        .weight
        .set(lax::broadcast_in_dim(&lax::iota([3], 0, F32), [3, 2], &[0]));
    m.correction.set(lax::iota([3], 0, F32) * c(-10.));
    let (ids, w) = m.route(&lax::broadcast_in_dim(&c(1.), [1, 2], &[]));
    close(&read(&d, ids), &[0., 1.]);
    close(&read(&d, w), &[0.5, 1. / (1. + (-2_f32).exp())]);
}
fn check_cache(
    init: impl Fn(&mut KVCache),
    forward: impl Fn(&Array, &Array, Option<&KVCache>) -> Array,
) {
    let d = device();
    let x =
        lax::iota([2, 3, 4], 1, F32) + lax::iota([2, 3, 4], 2, F32) + lax::iota([2, 3, 4], 0, F32);
    let reference = read(&d, forward(&x, &Array::scalar(7_i32), None));
    let cache = gnx_transforms::map(|| {
        let mut cache = KVCache::new(5);
        init(&mut cache);
        cache
    })
    .axis_size(2)
    .invoke(());
    let run = gnx_transforms::map(|chunk: Array, cache: KVCache| {
        let t = chunk.fixed_dims().unwrap()[0];
        let result = forward(
            &chunk.reshape([1, t, 4]),
            &(cache.position() + 7),
            Some(&cache),
        );
        cache.advance(t);
        result.reshape([t, 4])
    });
    let first = run.invoke((x.at((.., 0..1, ..)), cache.clone()));
    let rest = run.invoke((x.at((.., 1..3, ..)), cache.clone()));
    close(&read(&d, lax::concatenate(&[&first, &rest], 1)), &reference);
    close(&read(&d, cache.position()), &[3., 3.]);
}
#[test]
fn grouped_attention_chunked_cache() {
    let layer = GroupedQueryAttention::init(
        GroupedQueryAttentionConfig {
            hidden: 4,
            heads: 2,
            kv_heads: 1,
            head_dim: 2,
            norm_eps: Some(1e-6),
            rotary: Rotary::new(10000.),
            dtype: F32,
            cache_key: CacheKey::root(),
        },
        <GroupedQueryAttention as Model>::DefaultInit::default(),
        &rng(),
    );
    check_cache(
        |c| layer.init_cache(c),
        |x, p, c| layer.forward(x, p, c, None),
    );
}
#[test]
fn latent_attention_chunked_cache() {
    let layer = LatentAttention::init(
        LatentAttentionConfig {
            hidden: 4,
            heads: 2,
            query_rank: 3,
            rank: 2,
            nope: 2,
            rope: 2,
            value_dim: 2,
            eps: 1e-6,
            rotary: Rotary::new(10000.),
            dtype: F32,
            cache_key: CacheKey::root(),
        },
        <LatentAttention as Model>::DefaultInit::default(),
        &rng(),
    );
    check_cache(
        |c| layer.init_cache(c),
        |x, p, c| layer.forward(x, p, c, None),
    );
}

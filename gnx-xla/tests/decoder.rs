use futures_executor::block_on;
use gnx::{Array, Callable, Eval, jit, lax, nn::*, random::Algorithm};
use gnx_math::{
    DType::{F32, I32},
    ExecOpts, TransferPolicy,
};
use gnx_models::*;
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
    d.download(x.data().unwrap())
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
    let conv = Conv::init_with(2, 2, &[2], 2, false, F32, &rng());
    conv.kernel
        .set(lax::broadcast_in_dim(&c(1.), [2, 1, 2], &[]));
    let x = lax::iota([6], 0, F32).reshape([1, 3, 2]);
    close(&read(&d, conv.forward(x.clone())), &[2., 4., 6., 8.]);
    let biased = Conv::new(conv.kernel.value(), Some(lax::iota([2], 0, F32)), 2);
    close(&read(&d, biased.forward(x.clone())), &[2., 5., 6., 9.]);
    close(
        &read(&d, LayerNorm::init(2, 0., F32).forward(x.clone())),
        &[-1., 1., -1., 1., -1., 1.],
    );
    let group = GroupNorm::init(2, 2, 0., F32);
    let a = (1.5_f32).sqrt();
    close(&read(&d, group.forward(x.clone())), &[-a, -a, 0., 0., a, a]);
    let norm = BatchNorm::init(2, 0.5, 0., F32);
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
    let mut m = RoutedExperts::init(2, 3, 3, 2, 0, true, F32, &rng());
    m.normalize = false;
    m.router
        .weight
        .set(lax::broadcast_in_dim(&lax::iota([3], 0, F32), [3, 2], &[0]));
    m.correction.set(lax::iota([3], 0, F32) * c(-10.));
    let (ids, w) = m.route(&lax::broadcast_in_dim(&c(1.), [1, 2], &[]));
    close(&read(&d, ids), &[0., 1.]);
    close(&read(&d, w), &[0.5, 1. / (1. + (-2_f32).exp())]);
}
/// One token, identity embedding/head, unit value and output weights, zero MLP.
/// Attention then adds the projected value directly. Compute expected logits in scalar Rust.
#[test]
fn single_token_reference_logits() {
    let d = device();
    let ids = lax::zeros([1, 1], I32);
    let eps = 1e-6;
    let set = |p: &Param, v: f64| p.set(lax::broadcast_in_dim(&c(v), p.value().shape(), &[]));
    let eye = lax::ne(&lax::iota([2, 2], 0, I32), &lax::iota([2, 2], 1, I32)).astype(F32);
    let reference = |v: f64| {
        let norm = (((1. + v).powi(2) + v * v) / 2. + eps).sqrt();
        [((v / norm) as f32), (((1. + v) / norm) as f32)]
    };
    let a = (0.5_f64 + eps).sqrt().recip();
    let qc = Qwen3Config {
        vocab: 2,
        hidden: 2,
        layers: 1,
        heads: 1,
        kv_heads: 1,
        head_dim: 2,
        intermediate: 2,
        ..Default::default()
    };
    let q = Qwen3LM::init(&qc, F32, &rng());
    q.embedding.weight.set(eye.clone());
    let b = &q.blocks[0];
    set(&b.attention.value.weight, 1.);
    set(&b.attention.output.weight, 1.);
    set(&b.mlp.dense_layer().unwrap().down.weight, 0.);
    // Row zero is [0,1]. The tied head swaps the two normalized residual coordinates.
    let mut expected = reference(2. * a);
    expected.reverse();
    close(&read(&d, q.forward(&ids)), &expected);
    let kc = KimiK2Config {
        vocab: 2,
        hidden: 2,
        layers: 1,
        heads: 1,
        query_rank: 2,
        kv_rank: 2,
        nope: 2,
        rope: 2,
        value_dim: 2,
        intermediate: 2,
        tied: true,
        ..Default::default()
    };
    let k = KimiK2LM::init(&kc, F32, &rng());
    k.embedding.weight.set(eye);
    let b = &k.blocks[0];
    set(&b.attention.kv_down.weight, 1.);
    set(&b.attention.kv_up.weight, 1.);
    set(&b.attention.output.weight, 1.);
    set(&b.mlp.dense_layer().unwrap().down.weight, 0.);
    let mut expected = reference(4. * a / (a * a + eps).sqrt());
    expected.reverse();
    close(&read(&d, k.forward(&ids)), &expected);
}
fn check_cache(run: impl Fn(&Array, usize, Option<&mut DecoderCache>) -> Array) {
    let d = device();
    let ids = lax::iota([1, 3], 1, I32);
    let reference = read(&d, run(&ids, 7, None));
    let mut cache = DecoderCache::default();
    let mut result = vec![];
    for i in 0..3 {
        result.extend(read(
            &d,
            run(&ids.at((.., i..i + 1)), 7 + i, Some(&mut cache)),
        ));
        assert_eq!(cache.len(), i + 1);
    }
    close(&result, &reference);
}
#[test]
fn qwen_chunked_logits() {
    let c = Qwen3Config {
        vocab: 8,
        hidden: 4,
        layers: 2,
        heads: 2,
        kv_heads: 1,
        head_dim: 2,
        intermediate: 6,
        expert_width: 3,
        experts: 3,
        top_k: 2,
        mlp_only_layers: vec![0],
        ..Default::default()
    };
    let m = Qwen3LM::init(&c, F32, &rng());
    check_cache(|ids, p, c| m.forward_at(ids, p, c, None));
}
#[test]
fn kimi_chunked_logits() {
    let c = KimiK2Config {
        vocab: 8,
        hidden: 4,
        layers: 2,
        heads: 2,
        query_rank: 3,
        kv_rank: 2,
        nope: 2,
        rope: 2,
        value_dim: 2,
        intermediate: 6,
        experts: 3,
        expert_width: 3,
        top_k: 2,
        ..Default::default()
    };
    let m = KimiK2LM::init(&c, F32, &rng());
    check_cache(|ids, p, c| m.forward_at(ids, p, c, None));
}

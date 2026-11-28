use futures_executor::block_on;
use gnx::{
    Array, Eval, lax,
    nn::{InitializeWith, Rngs},
};
use gnx_base::Device as _;
use gnx_math::{DType::F32, ExecOpts, TransferPolicy};
use gnx_models::sphere::{BackboneConfig, Decoder, DecoderConfig, Encoder, EncoderConfig};
use gnx_xla::XlaDevice;

fn eval(x: Array) -> Vec<f32> {
    let device = XlaDevice::cpu(gnx_base::cache::Cache::in_memory()).unwrap();
    block_on(device.eval(
        &x,
        ExecOpts {
            transfers: TransferPolicy::AllowCopy,
        },
    ))
    .unwrap();
    block_on(device.download(x.data().unwrap()))
        .unwrap()
        .bytes()
        .chunks_exact(4)
        .map(|v| f32::from_ne_bytes(v.try_into().unwrap()))
        .collect()
}
fn equal(a: &Array, b: &Array) -> Array {
    lax::lt(
        &lax::abs(&(a - b)),
        &lax::broadcast_in_dim(&Array::scalar(0.5f32), a.shape(), &[]),
    )
}
fn eye(rows: usize, cols: usize) -> Array {
    let s = [rows, cols];
    equal(&lax::iota(s, 0, F32), &lax::iota(s, 1, F32)).astype(F32)
}
fn config() -> BackboneConfig {
    BackboneConfig {
        image_size: 2,
        patch_size: 1,
        width: 4,
        heads: 1,
        layers: 1,
        token_channels: 4,
        ..BackboneConfig::default()
    }
}
#[test]
fn manifold_projection_normalizes_each_coordinate_group() {
    use gnx_models::sphere::Manifold;
    let raw = [0.0f32, 0.0, 1e-4, 0.0, 3.0, -4.0];
    for (manifold, channels, group) in [(Manifold::Sphere, 6, 6), (Manifold::Torus, 3, 2)] {
        let m = Encoder::init(
            EncoderConfig {
                backbone: BackboneConfig {
                    manifold: Some(manifold),
                    token_channels: channels,
                    ..config()
                },
            },
            (),
            &Rngs::default(),
        );
        let values = raw.map(|v| Array::scalar(v).reshape([1]));
        m.head
            .bias
            .as_ref()
            .unwrap()
            .set(lax::concatenate(&values.iter().collect::<Vec<_>>(), 0));
        let actual = eval(m.forward(lax::zeros([2, 3, 2, 2], F32), None));
        let expected: Vec<_> = raw
            .chunks(group)
            .flat_map(|v| {
                let norm = (1e-6 + v.iter().map(|x| x * x).sum::<f32>()).sqrt();
                v.iter().map(move |x| x / norm)
            })
            .collect();
        assert_eq!(actual.len(), 8 * raw.len());
        for (a, e) in actual.iter().zip(expected.iter().cycle()) {
            assert!((a - e).abs() < 1e-6, "{a} vs {e}");
        }
    }
}
#[test]
fn axial_attention_on_a_two_by_two_image() {
    let m = Encoder::init(EncoderConfig { backbone: config() }, (), &Rngs::default());
    m.embed.weight.set(eye(4, 3));
    m.pointwise.weight.set(eye(4, 4));
    let block = &m.backbone.blocks[0];
    let id = eye(4, 4);
    block.qkv.weight.set(lax::concatenate(&[&id, &id, &id], 0));
    block.proj.weight.set(id.clone());
    block
        .w12
        .weight
        .set(lax::zeros(block.w12.weight.value().shape(), F32));
    m.head.weight.set(id);
    let output = eval(m.forward(
        (lax::iota([12], 0, F32) * (1. / 12.)).reshape([1, 3, 2, 2]),
        None,
    ));
    // Scalar reference: softmax(Rq (Rk)^T / 2) V + residual, identity projections.
    let expected = [
        0.4764486, 2.5383793, 1.7722296, 1.785686, 1.6250285, 2.0926978, 1.7424596, 1.9065965,
        0.4962732, 2.7455924, 2.909343, 1.2014158, 1.6049108, 2.3092258, 2.8771305, 1.3137636,
    ];
    assert_eq!(output.len(), expected.len());
    for (a, b) in output.into_iter().zip(expected) {
        assert!((a - b).abs() < 2e-5, "{a} vs {b}")
    }
}
#[test]
fn decoder_unpatchifies_channel_first_and_reflects_borders() {
    let m = Decoder::init(
        DecoderConfig {
            backbone: BackboneConfig {
                image_size: 4,
                patch_size: 2,
                ..config()
            },
            ..Default::default()
        },
        (),
        &Rngs::default(),
    );
    // Constant patch per channel, followed by a one-pixel leftward sampling offset.
    m.head
        .bias
        .as_ref()
        .unwrap()
        .set(lax::iota([12], 0, F32) * 0.1);
    let spatial = equal(&lax::iota([3, 3], 0, F32), &lax::iota([3, 3], 1, F32));
    let left = equal(&lax::iota([3, 3], 1, F32), &lax::zeros([3, 3], F32));
    // Top-left kernel tap samples (y-1,x-1) with reflection on both axes.
    m.refine.kernel.set(
        lax::bitwise_and(&spatial, &left)
            .astype(F32)
            .reshape([3, 3, 1, 1])
            * eye(3, 3),
    );
    let result = eval(m.forward(lax::zeros([1, 4, 4], F32), None));
    for c in 0..3 {
        for y in 0usize..4 {
            for x in 0usize..4 {
                let k = c * 4 + (y.abs_diff(1) % 2) * 2 + x.abs_diff(1) % 2;
                assert!((result[c * 16 + y * 4 + x] - (k as f32 * 0.1).tanh()).abs() < 2e-5)
            }
        }
    }
}

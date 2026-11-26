use futures_executor::block_on;
use gnx::{
    Array, Callable, Eval, jit,
    nn::Linear,
    random::{self, Algorithm::*, Rngs},
};
use gnx_math::{DType, ExecOpts, Shape, TransferPolicy};
use gnx_xla::XlaDevice;

fn bytes(device: &XlaDevice, x: Array) -> Vec<u8> {
    block_on(device.eval(
        &x,
        ExecOpts {
            transfers: TransferPolicy::AllowCopy,
        },
    ))
    .unwrap();
    device.download(x.data().unwrap()).unwrap().bytes().to_vec()
}
fn words(device: &XlaDevice, x: Array) -> Vec<u32> {
    bytes(device, x)
        .chunks_exact(4)
        .map(|b| u32::from_ne_bytes(b.try_into().unwrap()))
        .collect()
}

#[test]
fn reference_vectors() {
    let d = XlaDevice::cpu(gnx_base::cache::Cache::in_memory()).unwrap();
    // JAX 0.4.38, jax_threefry_partitionable=True, seed 0, key_data(split(key, 2)).
    let k = random::key(0, ThreeFry2x32);
    let split = random::split(&k, 2);
    assert_eq!(split.shape(), Shape::fixed([2]));
    let expected = [1797259609, 2579123966, 928981903, 3453687069];
    assert_eq!(words(&d, random::key_data(&split)), expected);
    let mapped = gnx::map(|k: Array| random::bits(&k, (), DType::U32)).invoke((split,));
    let (a, b) = random::split2(&k);
    let expected_bits = [a, b].map(|k| words(&d, random::bits(&k, (), DType::U32))[0]);
    assert_eq!(words(&d, mapped), expected_bits);
    assert_eq!(
        words(&d, random::bits(&k, (2,), DType::U32)),
        [expected[0] ^ expected[1], expected[2] ^ expected[3]]
    );
    assert_eq!(
        words(
            &d,
            random::key_data(&random::fold_in(&k, &gnx::lax::scalar(1., DType::U32)))
        ),
        expected[2..]
    );
    let k = random::wrap_key_data(&random::key_data(&k), Philox4x32);
    // First two lanes of the Random123 Philox4x32-10 zero-key/zero-counter vector.
    assert_eq!(
        words(&d, random::key_data(&random::split(&k, 1))),
        [0x6627e8d5, 0xe169c58d]
    );
}

#[test]
fn sample_dtypes_and_named_state() {
    let d = XlaDevice::cpu(gnx_base::cache::Cache::in_memory()).unwrap();
    let root = random::key(42, Philox4x32);
    for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64] {
        let u = random::uniform(&root, (32, 32), dtype);
        let vals = bytes(&d, gnx::lax::convert_element_type(&u, DType::F64));
        let vals: Vec<_> = vals
            .chunks_exact(8)
            .map(|b| f64::from_ne_bytes(b.try_into().unwrap()))
            .collect();
        assert!(vals.iter().all(|v| (0.0..1.0).contains(v)));
        if dtype == DType::F64 {
            assert!(vals.iter().any(|&v| f64::from(v as f32) != v));
        }
        let n = random::normal(&root, (32, 32), dtype);
        let vals = bytes(&d, gnx::lax::convert_element_type(&n, DType::F64));
        let vals: Vec<_> = vals
            .chunks_exact(8)
            .map(|b| f64::from_ne_bytes(b.try_into().unwrap()))
            .collect();
        assert!((vals.iter().sum::<f64>() / 1024.).abs() < 0.15);
        assert!((vals.iter().map(|v| v * v).sum::<f64>() / 1024. - 1.).abs() < 0.2);
    }
    let streams = Rngs::new(root.clone());
    let draw = jit(move |_: Array| streams.next("aux"));
    let a = draw.invoke((root.clone(),));
    let b = draw.invoke((root.clone(),));
    let reference = Rngs::new(root);
    let _ = reference.next("unused");
    assert_eq!(bytes(&d, a), bytes(&d, reference.next("aux")));
    assert_eq!(bytes(&d, b), bytes(&d, reference.next("aux")));
    let linear = Linear::init(4, 8, DType::F64, &reference);
    let bias = bytes(&d, linear.bias.value());
    assert!(
        bias.chunks_exact(8)
            .map(|b| f64::from_ne_bytes(b.try_into().unwrap()))
            .all(|v| v != 0. && v.abs() <= 0.5)
    );
}

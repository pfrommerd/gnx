use futures_executor::block_on;
use gnx::{
    Array, Callable, Eval, jit,
    nn::{Linear, Rngs},
    random::{self, Algorithm::*},
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
    // JAX 0.12.1, jax_threefry_partitionable=True, seed 0, key_data(split(key, 2)).
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
    // JAX 0.12.1 CPU, x64 enabled: use both seed words and nonzero counters.
    for (alg, key_words, split_words, folded, bits) in [
        (
            Philox4x32,
            [1796003549, 1087895578],
            [4141142979, 2623553941, 237775817, 1853558391],
            [3169161628, 3249634532],
            [2239733225, 3176046961],
        ),
        (
            ThreeFry2x32,
            [305419896, 2596069104],
            [3978822521, 2696639427, 2085429205, 1499321931],
            [2385792119, 1056902200],
            [1301508282, 621857182],
        ),
    ] {
        let k = random::key(0x123456789abcdef0, alg);
        assert_eq!(words(&d, random::key_data(&k)), key_words);
        assert_eq!(
            words(&d, random::key_data(&random::split(&k, 2))),
            split_words
        );
        assert_eq!(
            words(
                &d,
                random::key_data(&random::fold_in(&k, &gnx::lax::scalar(17., DType::U32)))
            ),
            folded
        );
        assert_eq!(words(&d, random::bits(&k, (2,), DType::U32)), bits);
    }
}

#[test]
fn sample_dtypes_and_named_state() {
    let d = XlaDevice::cpu(gnx_base::cache::Cache::in_memory()).unwrap();
    let root = random::key(42, Philox4x32);
    for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64] {
        let u = random::uniform(&root, (32, 32), dtype);
        let vals = bytes(&d, u.astype(DType::F64));
        let vals: Vec<_> = vals
            .chunks_exact(8)
            .map(|b| f64::from_ne_bytes(b.try_into().unwrap()))
            .collect();
        assert!(vals.iter().all(|v| (0.0..1.0).contains(v)));
        if dtype == DType::F64 {
            assert!(vals.iter().any(|&v| f64::from(v as f32) != v));
        }
        let n = random::normal(&root, (32, 32), dtype);
        let vals = bytes(&d, n.astype(DType::F64));
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
    let bias = bytes(&d, linear.bias().unwrap().value());
    assert!(
        bias.chunks_exact(8)
            .map(|b| f64::from_ne_bytes(b.try_into().unwrap()))
            .all(|v| v != 0. && v.abs() <= 0.5)
    );
}

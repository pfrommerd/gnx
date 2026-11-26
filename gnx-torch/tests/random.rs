use gnx::{
    Array, Callable, Eval, jit,
    random::{self, Algorithm::*},
};
use gnx_math::{DType, ExecOpts, TransferPolicy};
use gnx_torch::TorchDevice;

#[test]
fn torch_key_vectors_and_high_unsigned_words() {
    let device = TorchDevice::cpu(gnx_base::cache::Cache::in_memory()).unwrap();
    let read = |x: Array| {
        futures_executor::block_on(device.eval(
            &x,
            ExecOpts {
                transfers: TransferPolicy::AllowCopy,
            },
        ))
        .unwrap();
        device.download(x.data().unwrap()).unwrap().bytes().to_vec()
    };
    // JAX 0.12.1, seed 0: key_data(split(key, 1))[0], bits(key, dtype=uint32).
    for (algorithm, expected, expected_bits) in [
        (ThreeFry2x32, [1797259609, 2579123966], 4070199207),
        (Philox4x32, [3568036438, 3714396419], 2034863994),
    ] {
        let root = random::key(0, algorithm);
        let words = random::key_data(&random::split(&root, 1));
        let restored = random::wrap_key_data(&words, algorithm);
        let actual: Vec<_> = read(random::key_data(&restored))
            .chunks_exact(4)
            .map(|x| u32::from_ne_bytes(x.try_into().unwrap()))
            .collect();
        assert_eq!(actual, expected);
        let bits = read(random::bits(&root, (), DType::U32));
        assert_eq!(u32::from_ne_bytes(bits.try_into().unwrap()), expected_bits);
        let x = read(random::normal(&root, (4,), DType::F64));
        assert!(
            x.chunks_exact(8)
                .all(|v| f64::from_ne_bytes(v.try_into().unwrap()).is_finite())
        );
    }

    // JAX 0.12.1 CPU, x64 enabled: bits(fold_in(split(key, 3)[2],
    // uint32(0xffffffff)), (2, 3), dtype=uint32). Exercise both seed words,
    // unsigned tag conversion, and row-major counters on the Torch backend.
    for (alg, expected) in [
        (
            ThreeFry2x32,
            [
                1218079239, 2879198823, 502612289, 553449454, 2054347927, 956530362,
            ],
        ),
        (
            Philox4x32,
            [
                3118957149, 3097764800, 4221275272, 2274540258, 1820916393, 3618355186,
            ],
        ),
    ] {
        let children = random::split(&random::key(u64::MAX, alg), 3);
        let child = gnx::lax::slice_axis(&children, 0, 2, 3).reshape(());
        let key = random::fold_in(&child, &gnx::lax::scalar(f64::from(u32::MAX), DType::U32));
        let actual: Vec<_> = read(random::bits(&key, (2, 3), DType::U32))
            .chunks_exact(4)
            .map(|x| u32::from_ne_bytes(x.try_into().unwrap()))
            .collect();
        assert_eq!(actual, expected);
    }

    let seed = random::key(42, Philox4x32);
    let rngs = random::Rngs::new(seed.clone());
    let draw = jit(move |_: Array| random::bits(&rngs.next("params"), (), DType::U32));
    let reference = random::Rngs::new(seed.clone());
    for _ in 0..2 {
        assert_eq!(
            read(draw.invoke((seed.clone(),))),
            read(random::bits(&reference.next("params"), (), DType::U32))
        );
    }
}

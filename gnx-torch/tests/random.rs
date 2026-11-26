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
    for (algorithm, expected) in [
        (ThreeFry2x32, [1797259609, 2579123966]),
        (Philox4x32, [0x6627e8d5, 0xe169c58d]),
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
        assert_eq!(
            u32::from_ne_bytes(bits.try_into().unwrap()),
            expected[0] ^ expected[1]
        );
        let x = read(random::normal(&root, (4,), DType::F64));
        assert!(
            x.chunks_exact(8)
                .all(|v| f64::from_ne_bytes(v.try_into().unwrap()).is_finite())
        );
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

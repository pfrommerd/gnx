mod support;
use gnx_expr::{Value, VarInfo, pass::Builder};
use gnx_math::{
    ArrayInfo, DType, DataHandle, ExecOpts, HostBuffer, MemoryLocation, Shape, TransferPolicy,
    ops::{add_op, creation::iota},
};
use gnx_torch::TorchDevice;
use tch::{Device, Tensor};

#[test]
fn unavailable_cuda_or_invalid_ordinal_returns_an_error() {
    let err = TorchDevice::cuda(usize::MAX)
        .err()
        .expect("invalid ordinal");
    assert_eq!(
        err.kind(),
        if tch::Cuda::is_available() {
            std::io::ErrorKind::InvalidInput
        } else {
            std::io::ErrorKind::Unsupported
        }
    );
}
#[test]
#[ignore = "requires a visible NVIDIA GPU and CUDA-enabled LibTorch"]
fn cuda_import_compile_replay_and_readback() {
    let d = TorchDevice::cuda(0).unwrap();
    let same = TorchDevice::cuda(0).unwrap();
    assert_eq!(d.id(), same.id());
    let info = ArrayInfo::new(Shape::fixed([3]), DType::F32);
    let ty = VarInfo::new(info.clone());
    let mut b = Builder::new();
    let x = b.input(ty.clone());
    let idx = b.emit1(iota(0), &[], ty.clone());
    let result = b.emit1(add_op(), &[x, idx], ty);
    let p = d.compile(&b.finish(vec![result])).unwrap();
    let input = DataHandle::new(
        HostBuffer::copy_from_slice(
            info,
            &[1f32, 2., 3.]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap(),
    );
    assert!(d.import(&input, TransferPolicy::RequireZeroCopy).is_err());
    let owned = same.import(&input, TransferPolicy::AllowCopy).unwrap();
    let mut v = Value::new(owned);
    for expected in [[1f32, 3., 5.], [1., 4., 7.]] {
        v = p.run(&[v], ExecOpts::default()).unwrap().pop().unwrap();
        let out = v.downcast::<DataHandle>().unwrap();
        assert!(
            matches!(&out.shards()[0].location,MemoryLocation::Device{device,..} if *device==d.id())
        );
        let buf = same.download(out).unwrap();
        assert_eq!(
            buf.bytes(),
            expected
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>()
        );
    }
    let module = support::reload(p.modules().next().unwrap());
    let fresh = Tensor::from_slice(&[5f32, 6., 7.]).to_device(Device::Cuda(0));
    let out = module.forward_ts(&[fresh]).unwrap();
    assert_eq!(out.device(), Device::Cuda(0));
    assert_eq!(
        Vec::<f32>::try_from(out.to_device(Device::Cpu)).unwrap(),
        [5., 7., 9.]
    );
}

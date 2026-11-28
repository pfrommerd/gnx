//! Opt-in tests for CUDA PJRT and the GNX device on a real GPU.
use gnx_base::Device as _;
use xla::{ElementType, PjRtClient, XlaBuilder};

#[test]
#[ignore = "requires a CUDA XLA distribution, CUDA runtime libraries and an NVIDIA GPU"]
fn cuda_device_transfers_evaluates_and_reuses_buffers() -> Result<(), Box<dyn std::error::Error>> {
    use futures_executor::block_on;
    use gnx::{Array, Eval, ExecOpts};
    use gnx_math::{
        ArrayInfo, DType, DataHandle, HostBuffer, MemoryLocation, Shape, TransferPolicy,
    };
    use gnx_xla::XlaDevice;

    let gpu = XlaDevice::cuda(0, gnx_base::cache::Cache::in_memory())?;
    assert_eq!(gpu.platform(), "cuda");
    assert_eq!(gpu.ordinal(), 0);
    assert_eq!(
        gpu.id(),
        XlaDevice::cuda(0, gnx_base::cache::Cache::in_memory())?.id()
    );
    let bytes = [2i32, -3, 7].map(i32::to_le_bytes).concat();
    let host = DataHandle::new(HostBuffer::copy_from_slice(
        ArrayInfo::new(Shape::fixed([3]), DType::I32),
        &bytes,
    )?);
    assert_eq!(
        block_on(gpu.upload(&host, TransferPolicy::RequireZeroCopy))
            .unwrap_err()
            .kind(),
        std::io::ErrorKind::Unsupported
    );
    let arr = Array::from_data(host);
    let square = &arr * &arr;
    block_on(gpu.eval(
        &square,
        ExecOpts {
            transfers: TransferPolicy::AllowCopy,
        },
    ))?;
    let out = &square + &square;
    // The inputs are already on the GPU, so strict evaluation succeeds.
    block_on(gpu.eval(&out, ExecOpts::default()))?;
    let data = out.data().unwrap();
    assert!(
        matches!(&data.shards()[0].location, MemoryLocation::Device { device, .. } if device == &gpu.id())
    );
    assert_eq!(
        block_on(gpu.download(data))?.bytes(),
        [8i32, 18, 98].map(i32::to_le_bytes).concat()
    );
    Ok(())
}

#[test]
#[ignore = "requires a CUDA XLA distribution, CUDA runtime libraries and an NVIDIA GPU"]
fn matmul_on_cuda() -> xla::Result<()> {
    let client = PjRtClient::gpu(0.1, false)?;
    let devices = client.addressable_devices();
    let gpu = devices.first().expect("no addressable CUDA device");
    eprintln!("{}: {}", client.platform_name(), gpu.kind());
    const N: usize = 64;
    let dims = [N as i64; 2];
    let builder = XlaBuilder::new("cuda_matmul");
    let lhs = builder.parameter(0, ElementType::F32, &dims, "lhs")?;
    let rhs = builder.parameter(1, ElementType::F32, &dims, "rhs")?;
    let program = client.compile(&lhs.dot(&rhs)?.build()?)?;

    let a: Vec<_> = (0..N * N).map(|i| (i % 13) as f32 - 6.).collect();
    let b: Vec<_> = (0..N * N).map(|i| (i % 7) as f32 - 3.).collect();
    let device_inputs = [
        client.buffer_from_host_buffer(&a, &[N, N], Some(gpu))?,
        client.buffer_from_host_buffer(&b, &[N, N], Some(gpu))?,
    ];
    let result = program.execute_b(&device_inputs)?;
    let actual = result[0][0].to_literal_sync()?.to_vec::<f32>()?;
    assert_eq!(actual.len(), N * N);
    for (i, value) in actual.into_iter().enumerate() {
        let expected: f32 = (0..N).map(|k| a[(i / N) * N + k] * b[k * N + i % N]).sum();
        assert_eq!(value, expected, "output element {i}");
    }
    Ok(())
}

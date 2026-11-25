//! Run with a managed CUDA build on a host without libcuda.so.1. Kept separate
//! from the hardware tests so that `--test cuda -- --ignored` still targets GPUs.
#![cfg(feature = "cuda")]

#[test]
#[ignore = "requires a managed CUDA build with the NVIDIA driver absent"]
fn cpu_available_without_driver() -> std::io::Result<()> {
    use gnx_xla::XlaDevice;
    let cpu = XlaDevice::cpu(gnx_common::cache::Cache::in_memory())?;
    assert_eq!(cpu.platform(), "cpu");
    let err = XlaDevice::cuda(0, gnx_common::cache::Cache::in_memory())
        .expect_err("test requires no CUDA driver");
    assert!(
        err.to_string().contains("could not load the NVIDIA driver"),
        "{err}"
    );
    Ok(())
}

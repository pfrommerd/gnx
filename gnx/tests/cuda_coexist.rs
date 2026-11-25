//! Exercise each load order in a fresh process; both devices stay live in each child.
#![cfg(all(target_os = "linux", target_arch = "x86_64", feature = "qwen-example"))]

use gnx_expr::{Value, VarInfo, pass::Builder};
use gnx_math::{
    ArrayInfo, DType, DataHandle, ExecOpts, HostBuffer, Shape, TransferPolicy,
    ops::{
        contraction::{ConvAxes, ConvKernelAxes, ConvParameters, conv_general_dilated},
        ones_op,
    },
};
use gnx_torch::{Tensor, TorchDevice};
use gnx_xla::XlaDevice;
use std::{collections::BTreeSet, path::PathBuf, process::Command};

const X: [usize; 4] = [1, 8, 16, 16];
const W: [usize; 4] = [8, 8, 3, 3];
const Y: [usize; 4] = [1, 8, 14, 14];
fn info(shape: [usize; 4]) -> ArrayInfo {
    ArrayInfo::new(Shape::fixed(shape), DType::F32)
}
fn torch_conv(d: &TorchDevice, scale: f32) {
    let ones = |shape| {
        let mut b = Builder::new();
        let v = b.emit1(ones_op(), &[], VarInfo::new(info(shape)));
        let p = d.compile(&b.finish(vec![v]).build().unwrap()).unwrap();
        p.modules()
            .next()
            .unwrap()
            .forward_ts::<Tensor>(&[])
            .unwrap()
    };
    // GNX's generic convolution can lower to unfold + matmul. Call ATen's
    // explicit cuDNN operator so the test cannot pass by avoiding cuDNN.
    let y = (ones(X) * f64::from(scale))
        .f_cudnn_convolution(&ones(W), [0, 0], [1, 1], [1, 1], 1, false, true, false)
        .unwrap();
    assert_eq!(y.size(), Y.map(|i| i as i64));
    assert_eq!(y.min().double_value(&[]), 72. * f64::from(scale));
    assert_eq!(y.max().double_value(&[]), 72. * f64::from(scale));
}
fn xla_conv(d: &XlaDevice, scale: f32) {
    let mut b = Builder::new();
    let x = b.input(VarInfo::new(info(X)));
    let w = b.input(VarInfo::new(info(W)));
    let axes = ConvAxes {
        batch: 0,
        feature: 1,
        spatial: vec![2, 3],
    };
    let y = b.emit1(
        conv_general_dilated(ConvParameters {
            input: axes.clone(),
            output: axes,
            kernel: ConvKernelAxes {
                output_feature: 0,
                input_feature: 1,
                spatial: vec![2, 3],
            },
            strides: vec![1; 2],
            padding: vec![(0, 0); 2],
            lhs_dilation: vec![1; 2],
            rhs_dilation: vec![1; 2],
            feature_groups: 1,
            batch_groups: 1,
        }),
        &[x, w],
        VarInfo::new(info(Y)),
    );
    let arg = |shape: [usize; 4], v: f32| {
        Value::new(DataHandle::new(
            HostBuffer::copy_from_slice(
                info(shape),
                &v.to_le_bytes().repeat(shape.iter().product()),
            )
            .unwrap(),
        ))
    };
    let out = d
        .compile(&b.finish(vec![y]).build().unwrap())
        .unwrap()
        .run(
            &[arg(X, scale), arg(W, 1.)],
            ExecOpts {
                transfers: TransferPolicy::AllowCopy,
            },
        )
        .unwrap();
    let host = d
        .download(out[0].downcast::<DataHandle>().unwrap())
        .unwrap();
    assert_eq!(
        host.bytes(),
        (72. * scale).to_le_bytes().repeat(Y.iter().product())
    );
}
fn cudnn_paths() -> BTreeSet<PathBuf> {
    std::fs::read_to_string("/proc/self/maps")
        .unwrap()
        .lines()
        .filter_map(|line| {
            line.split_once('/')
                .map(|(_, p)| PathBuf::from("/").join(p))
        })
        .filter(|p| {
            p.file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .starts_with("libcudnn")
        })
        .collect()
}
fn check_cudnn() -> PathBuf {
    let paths = cudnn_paths();
    assert!(
        paths.iter().any(|p| p.ends_with("libcudnn_graph.so.9")),
        "cuDNN graph library not loaded: {paths:?}"
    );
    let dirs: BTreeSet<_> = paths
        .iter()
        .map(|p| p.parent().unwrap().to_owned())
        .collect();
    assert_eq!(dirs.len(), 1, "mixed cuDNN installations: {paths:?}");
    dirs.into_iter().next().unwrap()
}
#[test]
#[ignore = "requires the managed CUDA 13 runtimes and a visible NVIDIA GPU"]
fn same_process_convolutions() {
    const ORDER: &str = "GNX_TEST_CUDA_LOAD_ORDER";
    match std::env::var(ORDER).as_deref() {
        Ok("torch-first") => {
            let torch = TorchDevice::cuda(0).unwrap();
            torch_conv(&torch, 1.);
            let dir = check_cudnn();
            let xla = XlaDevice::cuda(0).unwrap();
            xla_conv(&xla, 2.);
            torch_conv(&torch, 3.);
            assert_eq!(dir, check_cudnn());
        }
        Ok("xla-first") => {
            let xla = XlaDevice::cuda(0).unwrap();
            xla_conv(&xla, 1.);
            let dir = check_cudnn();
            let torch = TorchDevice::cuda(0).unwrap();
            torch_conv(&torch, 2.);
            xla_conv(&xla, 3.);
            assert_eq!(dir, check_cudnn());
        }
        Err(_) => {
            for order in ["torch-first", "xla-first"] {
                let status = Command::new(std::env::current_exe().unwrap())
                    .args([
                        "--exact",
                        "same_process_convolutions",
                        "--ignored",
                        "--nocapture",
                    ])
                    .env(ORDER, order)
                    .status()
                    .unwrap();
                assert!(status.success(), "{order}: {status}");
            }
        }
        Ok(s) => panic!("unknown test order {s}"),
    }
}

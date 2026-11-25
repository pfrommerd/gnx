mod support;
use gnx_common::pass::Builder;
use gnx_expr::VarInfo;
use gnx_math::{ArrayInfo, DType, Shape, ops::linalg::hessenberg};
use gnx_torch::TorchDevice;
use tch::{CModule, Kind, Tensor};

fn compile(shape: &[usize], dtype: DType) -> Vec<CModule> {
    let ty =
        |dims: &[usize]| VarInfo::new(ArrayInfo::new(Shape::fixed(dims.iter().copied()), dtype));
    let n = shape[shape.len() - 1];
    let mut tau_shape = shape[..shape.len() - 1].to_vec();
    *tau_shape.last_mut().unwrap() = n.saturating_sub(1);
    let mut b = Builder::new();
    let a = b.input(ty(shape));
    let outputs = b.emit(hessenberg(), &[a], [ty(shape), ty(&tau_shape)], vec![]);
    futures_executor::block_on(
        TorchDevice::cpu(gnx_common::cache::Cache::in_memory())
            .unwrap()
            .compile(&b.finish(outputs).build().unwrap()),
    )
    .unwrap()
    .modules()
    .map(support::reload)
    .collect()
}

// Reconstruct using the independent LAPACK-style Householder product kernel.
// The reflector panel starts one row below the diagonal. Its diagonal is
// implicitly one, while those stored entries are the subdiagonal of H.
fn check_replay(modules: &[CModule], a: &Tensor, tol: f64) {
    let packed = modules[0].forward_ts(&[a]).unwrap();
    let tau = modules[1].forward_ts(&[a]).unwrap();
    let shape = a.size();
    let n = *shape.last().unwrap();
    let batch = &shape[..shape.len() - 2];
    let dims = |tail: &[i64]| [batch, tail].concat();
    assert_eq!(packed.size(), shape);
    assert_eq!(tau.size(), dims(&[(n - 1).max(0)]));
    assert_eq!(packed.kind(), a.kind());
    assert_eq!(tau.kind(), a.kind());
    if n <= 1 {
        assert!(packed.equal(a));
        return;
    }
    let panel = packed.narrow(-2, 1, n - 1).narrow(-1, 0, n - 1);
    let unit = panel.linalg_householder_product(&tau);
    let ident = Tensor::eye(n, (a.kind(), tch::Device::Cpu)).expand(&shape, false);
    let bottom = Tensor::cat(
        &[
            Tensor::zeros(dims(&[n - 1, 1]), (a.kind(), tch::Device::Cpu)),
            unit,
        ],
        -1,
    );
    let q = Tensor::cat(&[ident.narrow(-2, 0, 1), bottom], -2);
    assert!(q.adjoint().matmul(&q).allclose(&ident, tol, tol, false));
    let h = packed.triu(-1);
    let actual = q.adjoint().matmul(a).matmul(&q);
    assert!(
        actual.allclose(&h, tol, tol, false),
        "max similarity residual: {}",
        (&actual - &h).abs().max().double_value(&[])
    );
}
fn matrix() -> Tensor {
    Tensor::from_slice(&[
        1f64, 2., 3., 4., -2., 7., 3.5, 2., 5., 0., -3., 1., -6., 2.5, -2., 8.,
    ])
    .reshape([4, 4])
}
#[test]
fn packed_hessenberg_replays_on_fresh_real_batches() {
    let modules = compile(&[2, 4, 4], DType::F64);
    let a = matrix();
    let eye = Tensor::eye(4, (Kind::Double, tch::Device::Cpu));
    for x in [
        Tensor::stack(&[&a, &a.transpose(0, 1)], 0),
        Tensor::stack(&[&eye, &(&a.flip([0]) * 1.5 - &eye)], 0),
    ] {
        check_replay(&modules, &x, 1e-11);
    }
}
#[test]
fn packed_hessenberg_replays_complex_phases() {
    let modules = compile(&[4, 4], DType::C128);
    let a = matrix();
    let x = Tensor::complex(&a, &a.flip([0]).transpose(0, 1));
    check_replay(&modules, &x, 1e-11);
    check_replay(&modules, &x.adjoint(), 1e-11);
    // A purely imaginary last subdiagonal needs a final phase reflector.
    let y = Tensor::complex(&a.triu(0), &a.triu(-1));
    check_replay(&modules, &y, 1e-11);
    let pack = modules[0].forward_ts(&[&y]).unwrap();
    assert!(
        pack.diagonal(-1, -2, -1)
            .imag()
            .abs()
            .max()
            .double_value(&[])
            < 1e-11
    );
}
#[test]
fn packed_hessenberg_empty_and_singleton_matrices() {
    for n in [0, 1] {
        let m = compile(&[2, n, n], DType::F32);
        let x = Tensor::full([2, n as i64, n as i64], 3., (Kind::Float, tch::Device::Cpu));
        check_replay(&m, &x, 1e-6);
    }
}

mod support;
use gnx_expr::{VarInfo, pass::Builder};
use gnx_math::{ArrayInfo, DType, Shape, ops::linalg::eig};
use gnx_torch::TorchDevice;
use tch::{CModule, Kind, Tensor};

fn compiled(shape: &[usize], dtype: DType, left: bool, right: bool) -> Vec<CModule> {
    let ty =
        |dims: &[usize], k| VarInfo::new(ArrayInfo::new(Shape::fixed(dims.iter().copied()), k));
    let complex = match dtype {
        DType::F32 | DType::C64 => DType::C64,
        _ => DType::C128,
    };
    let mut b = Builder::new();
    let x = b.input(ty(shape, dtype));
    let mut out = vec![ty(&shape[..shape.len() - 1], complex)];
    for want in [left, right] {
        if want {
            out.push(ty(shape, complex));
        }
    }
    let results = b.emit(eig(left, right), &[x], out, vec![]);
    TorchDevice::cpu()
        .unwrap()
        .compile(&b.finish(results).build().unwrap())
        .unwrap()
        .modules()
        .map(support::reload)
        .collect()
}
fn run(modules: &[CModule], x: &Tensor) -> Vec<Tensor> {
    modules
        .iter()
        .map(|m| m.forward_ts(&[x]).unwrap())
        .collect()
}
fn complex(re: &[f64], im: &[f64]) -> Tensor {
    Tensor::complex(&Tensor::from_slice(re), &Tensor::from_slice(im))
}
fn check_vectors(a: &Tensor, w: &Tensor, v: &Tensor, left: bool, tol: f64) {
    // Check unit-length columns as well: a zero vector would pass the residual test.
    let norm2 = v.abs().square().sum_dim_intlist(&[-2i64][..], false, None);
    assert!(norm2.allclose(&norm2.ones_like(), tol, tol, false));
    let a = a.to_kind(w.kind());
    let (lhs, rhs) = if left {
        let u = v.adjoint();
        (u.matmul(&a), w.unsqueeze(-1) * u)
    } else {
        (a.matmul(v), v * w.unsqueeze(-2))
    };
    assert!(
        lhs.allclose(&rhs, tol, tol, false),
        "left={left}, max residual {}",
        (lhs - rhs).abs().max().double_value(&[])
    );
}
// Do not assume LAPACK eigenvalue order or a particular eigenvector phase.
fn check_spectrum(w: &Tensor, expected: &Tensor, tol: f64) {
    let d = (w.unsqueeze(-1) - expected.unsqueeze(-2)).abs();
    for axis in [-1, -2] {
        assert!(d.min_dim(axis, false).0.max().double_value(&[]) < tol);
    }
}
#[test]
fn batched_nonsymmetric_eig_replays_left_and_right_equations() {
    let modules = compiled(&[2, 3, 3], DType::F64, true, true);
    let base = Tensor::from_slice(&[2f64, -3., 4., 3., 2., 1., 0., 0., 7.]).reshape([3, 3]);
    let spectrum = complex(&[2., 2., 7.], &[3., -3., 0.]);
    for (scale, shift) in [(1., 0.), (-2., 1.5)] {
        let a = &base * scale + Tensor::eye(3, (Kind::Double, tch::Device::Cpu)) * shift;
        let batch = Tensor::stack(&[a.shallow_clone(), a.transpose(0, 1) * 0.5], 0);
        let out = run(&modules, &batch);
        let expected = &spectrum * scale + shift;
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].size(), [2, 3]);
        for v in &out {
            assert_eq!(v.kind(), Kind::ComplexDouble);
        }
        check_spectrum(
            &out[0],
            &Tensor::stack(&[expected.shallow_clone(), expected * 0.5], 0),
            1e-11,
        );
        check_vectors(&batch, &out[0], &out[1], true, 1e-11);
        check_vectors(&batch, &out[0], &out[2], false, 1e-11);
    }
}
#[test]
fn complex_eig_optional_vectors_use_conjugate_left_equation() {
    let a = complex(&[1., 3., 0., -2.], &[2., -1., 0., 4.])
        .reshape([2, 2])
        .to_kind(Kind::ComplexFloat);
    let expected = complex(&[1., -2.], &[2., 4.]).to_kind(Kind::ComplexFloat);
    for (left, right) in [(false, false), (true, false), (false, true), (true, true)] {
        let out = run(&compiled(&[2, 2], DType::C64, left, right), &a);
        assert_eq!(out.len(), 1 + usize::from(left) + usize::from(right));
        for v in &out {
            assert_eq!(v.kind(), Kind::ComplexFloat);
        }
        check_spectrum(&out[0], &expected, 1e-5);
        if left {
            check_vectors(&a, &out[0], &out[1], true, 1e-5)
        }
        if right {
            check_vectors(&a, &out[0], out.last().unwrap(), false, 1e-5)
        }
    }
}
#[test]
fn repeated_eigenvalues_do_not_duplicate_a_left_column() {
    let modules = compiled(&[3, 3], DType::F32, true, true);
    let eye = Tensor::eye(3, (Kind::Float, tch::Device::Cpu));
    let out = run(&modules, &eye);
    let left = &out[1];
    assert!(left.adjoint().matmul(left).allclose(
        &eye.to_kind(Kind::ComplexFloat),
        1e-6,
        1e-6,
        false
    ));
    let jordan = Tensor::from_slice(&[2f32, 1., 0., 0., 2., 0., 0., 0., 3.]).reshape([3, 3]);
    let out = run(&modules, &jordan);
    check_vectors(&jordan, &out[0], &out[1], true, 1e-5);
    check_vectors(&jordan, &out[0], &out[2], false, 1e-5);
}

mod support;
use gnx_expr::{VarInfo, pass::Builder};
use gnx_math::{ArrayInfo, DType, Shape, ops::linalg::tridiagonal};
use gnx_torch::TorchDevice;
use tch::{CModule, Kind, Tensor};

fn compile(dims: &[usize], dtype: DType, lower: bool) -> Vec<CModule> {
    let real = match dtype {
        DType::C64 => DType::F32,
        DType::C128 => DType::F64,
        k => k,
    };
    let vector = &dims[..dims.len() - 1];
    let mut off = vector.to_vec();
    *off.last_mut().unwrap() = dims[dims.len() - 1].saturating_sub(1);
    let ty =
        |dims: &[usize], k| VarInfo::new(ArrayInfo::new(Shape::fixed(dims.iter().copied()), k));
    let mut b = Builder::new();
    let x = b.input(ty(dims, dtype));
    let out = b.emit(
        tridiagonal(lower),
        &[x],
        [
            ty(dims, dtype),
            ty(vector, real),
            ty(&off, real),
            ty(&off, dtype),
        ],
        vec![],
    );
    TorchDevice::cpu()
        .unwrap()
        .compile(&b.finish(out))
        .unwrap()
        .modules()
        .map(support::reload)
        .collect()
}

fn check(modules: &[CModule], input: &Tensor, a: &Tensor, lower: bool, tol: f64) {
    let result: Vec<_> = modules
        .iter()
        .map(|m| m.forward_ts(&[input]).unwrap())
        .collect();
    let [packed, d, e, tau] = result.as_slice() else {
        panic!("expected four results")
    };
    let shape = a.size();
    let n = *shape.last().unwrap();
    let vector = &shape[..shape.len() - 1];
    let mut off = vector.to_vec();
    *off.last_mut().unwrap() = (n - 1).max(0);
    assert_eq!(packed.size(), shape);
    assert_eq!(d.size(), vector);
    assert_eq!(e.size(), off);
    assert_eq!(tau.size(), off);
    assert_eq!(packed.kind(), a.kind());
    assert_eq!(tau.kind(), a.kind());
    assert_eq!(d.kind(), a.real().kind());
    assert_eq!(e.kind(), d.kind());
    let side = if lower { -1 } else { 1 };
    assert!(packed.diagonal(0, -2, -1).equal(&d.to_kind(a.kind())));
    assert!(packed.diagonal(side, -2, -1).equal(&e.to_kind(a.kind())));
    if lower {
        assert!(packed.triu(1).equal(&input.triu(1)))
    } else {
        assert!(packed.tril(-1).equal(&input.tril(-1)))
    }
    if n <= 1 {
        assert!(d.equal(&a.diagonal(0, -2, -1).real()));
        return;
    }
    let identity = Tensor::eye(n, (a.kind(), tch::Device::Cpu)).expand(&shape, false);
    let mut q = identity.shallow_clone();
    // Form each full reflector independently from the stored column. Upper
    // storage represents Q=P_(n-2)...P_0 rather than P_0...P_(n-2).
    for step in 0..n - 1 {
        let i = if lower { step } else { n - 2 - step };
        let col = packed.select(-1, if lower { i } else { i + 1 });
        let unit = col.narrow(-1, 0, 1).ones_like();
        let v = if lower {
            Tensor::cat(
                &[
                    col.narrow(-1, 0, i + 1).zeros_like(),
                    unit,
                    col.narrow(-1, i + 2, n - i - 2),
                ],
                -1,
            )
        } else {
            Tensor::cat(
                &[
                    col.narrow(-1, 0, i),
                    unit,
                    col.narrow(-1, i + 1, n - i - 1).zeros_like(),
                ],
                -1,
            )
        }
        .unsqueeze(-1);
        let p = &identity - v.matmul(&v.adjoint()) * tau.narrow(-1, i, 1).unsqueeze(-1);
        q = q.matmul(&p);
    }
    let t = (d.diag_embed(0, -2, -1) + e.diag_embed(1, -2, -1) + e.diag_embed(-1, -2, -1))
        .to_kind(a.kind());
    assert!(q.adjoint().matmul(&q).allclose(&identity, tol, tol, false));
    let actual = q.adjoint().matmul(a).matmul(&q);
    assert!(
        actual.allclose(&t, tol, tol, false),
        "lower={lower}, max residual {}",
        (actual - t).abs().max().double_value(&[])
    );
}
fn base() -> Tensor {
    Tensor::from_slice(&[
        1f64, -2., 3., 1., 2., 0., 5., -4., 3., 7., 1., 9., -3., 2., 4., -2.,
    ])
    .reshape([4, 4])
}
#[test]
fn packed_tridiagonal_replays_symmetric_batches_in_both_orientations() {
    let x = base();
    let a = &x + x.transpose(-1, -2);
    let eye = Tensor::eye(4, (Kind::Double, tch::Device::Cpu));
    for lower in [true, false] {
        let m = compile(&[2, 4, 4], DType::F64, lower);
        for mat in [
            Tensor::stack(&[&a, &(&a * -2. + &eye)], 0),
            Tensor::stack(&[&eye, &a.flip([0, 1])], 0),
        ] {
            check(&m, &mat, &mat, lower, 1e-11);
        }
    }
}
#[test]
fn packed_tridiagonal_replays_hermitian_and_reads_only_selected_triangle() {
    let x = base();
    let z = Tensor::complex(&x, &x.flip([1]));
    let a = &z + z.adjoint();
    for lower in [true, false] {
        let m = compile(&[4, 4], DType::C128, lower);
        for mat in [a.shallow_clone(), a.flip([0, 1]) * -1.75] {
            check(&m, &mat, &mat, lower, 1e-11);
            let unused = if lower {
                mat.ones_like().triu(1)
            } else {
                mat.ones_like().tril(-1)
            };
            let poison = &mat
                + unused * 97.
                + Tensor::complex(
                    &x.zeros_like(),
                    &Tensor::eye(4, (Kind::Double, tch::Device::Cpu)),
                );
            check(&m, &poison, &mat, lower, 1e-11);
        }
    }
}
#[test]
fn tridiagonal_empty_and_singleton_return_empty_reflectors() {
    for n in [0, 1] {
        let m = compile(&[2, n, n], DType::C64, false);
        let a = Tensor::full(
            [2, n as i64, n as i64],
            4.,
            (Kind::ComplexFloat, tch::Device::Cpu),
        );
        check(&m, &a, &a, false, 1e-6);
    }
}

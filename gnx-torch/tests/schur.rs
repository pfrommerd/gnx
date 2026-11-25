mod support;
use gnx_expr::{VarInfo, pass::Builder};
use gnx_math::{ArrayInfo, DType, Shape, ops::linalg::schur};
use gnx_torch::TorchDevice;
use tch::{CModule, Kind, Tensor};

fn compiled(dims: &[usize], dtype: DType, vectors: bool) -> Vec<CModule> {
    let ty = VarInfo::new(ArrayInfo::new(Shape::fixed(dims.iter().copied()), dtype));
    let mut b = Builder::new();
    let x = b.input(ty.clone());
    let out = b.emit(
        schur(vectors),
        &[x],
        vec![ty; 1 + usize::from(vectors)],
        vec![],
    );
    TorchDevice::cpu()
        .unwrap()
        .compile(&b.finish(out).build().unwrap())
        .unwrap()
        .modules()
        .map(support::reload)
        .collect()
}
fn replay(m: &[CModule], x: &Tensor) -> Vec<Tensor> {
    m.iter().map(|m| m.forward_ts(&[x]).unwrap()).collect()
}
fn check(a: &Tensor, t: &Tensor, q: &Tensor, tol: f64) {
    assert_eq!(t.size(), a.size());
    assert_eq!(q.size(), a.size());
    assert_eq!(q.kind(), a.kind());
    assert_eq!(t.kind(), a.kind());
    let n = a.size()[a.dim() - 1];
    let eye = Tensor::eye(n, (a.kind(), tch::Device::Cpu)).expand(a.size(), false);
    assert!(q.adjoint().matmul(q).allclose(&eye, tol, tol, false));
    let actual = q.adjoint().matmul(a).matmul(q);
    assert!(
        actual.allclose(t, tol, tol, false),
        "max similarity residual {}",
        (actual - t).abs().max().double_value(&[])
    );
}
fn real_form(t: &Tensor, expected_pairs: usize) {
    assert!(t.tril(-2).equal(&t.zeros_like()));
    let n = t.size()[0];
    let mut count = 0;
    let mut i = 0;
    while i < n - 1 {
        let c = t.double_value(&[i + 1, i]);
        if c != 0. {
            assert_eq!(t.double_value(&[i, i]), t.double_value(&[i + 1, i + 1]));
            assert!(c * t.double_value(&[i, i + 1]) < 0.);
            if i + 2 < n {
                assert_eq!(t.double_value(&[i + 2, i + 1]), 0.)
            }
            count += 1;
            i += 2;
        } else {
            i += 1;
        }
    }
    assert_eq!(count, expected_pairs);
}
fn matrix(v: &[f64]) -> Tensor {
    Tensor::from_slice(v).reshape([4, 4])
}
#[test]
fn real_schur_replays_fresh_batches_with_different_block_structures() {
    let m = compiled(&[2, 4, 4], DType::F64, true);
    let u = matrix(&[
        1., 1., 1., 1., 1., -1., 1., -1., 1., 1., -1., -1., 1., -1., -1., 1.,
    ]) * 0.5;
    let mix = |t: Tensor| u.matmul(&t).matmul(&u.transpose(0, 1));
    let pairs = mix(matrix(&[
        2., 5., 1., 3., -3., 4., -1., 2., 0., 0., -2., 1., 0., 0., -7., -4.,
    ]));
    let singles = mix(matrix(&[
        1., 8., 2., -3., 0., 4., 6., 5., 0., 0., -3., 7., 0., 0., 0., 9.,
    ]));
    let mixed = mix(matrix(&[
        1., 0., 4., 1., 0., -1., 5., 2., 0., -4., -2., 3., 0., 0., 0., 8.,
    ]));
    for (a, counts) in [
        (Tensor::stack(&[&pairs, &singles], 0), [2, 0]),
        (Tensor::stack(&[&mixed, &pairs.transpose(0, 1)], 0), [1, 2]),
    ] {
        let out = replay(&m, &a);
        check(&a, &out[0], &out[1], 1e-10);
        for (j, n) in counts.into_iter().enumerate() {
            real_form(&out[0].get(j as i64), n);
        }
    }
}
#[test]
fn complex_schur_replays_unitary_triangular_form_and_defective_matrix() {
    let m = compiled(&[4, 4], DType::C128, true);
    let form_only = compiled(&[4, 4], DType::C128, false);
    let re = matrix(&[
        2., -4., 3., 0., 7., 1., 4., 1., 2., 1., 3., -5., 2., 0., 3., 4.,
    ]);
    let dense = Tensor::complex(&re, &re.flip([0]));
    let jordan = matrix(&[
        2., 1., 0., 0., 0., 2., 1., 0., 0., 0., 2., 1., 0., 0., 0., 2.,
    ]);
    let defective = Tensor::complex(&jordan, &jordan);
    for a in [dense, defective] {
        let out = replay(&m, &a);
        check(&a, &out[0], &out[1], 1e-10);
        assert!(out[0].tril(-1).equal(&a.zeros_like()));
        let only = replay(&form_only, &a);
        assert_eq!(only.len(), 1);
        assert!(only[0].allclose(&out[0], 1e-10, 1e-10, false));
    }
}
#[test]
fn schur_replays_empty_and_singleton_matrices() {
    for n in [0, 1] {
        let m = compiled(&[2, n, n], DType::F32, true);
        let a = Tensor::full([2, n as i64, n as i64], 7., (Kind::Float, tch::Device::Cpu));
        let out = replay(&m, &a);
        check(&a, &out[0], &out[1], 1e-6);
        assert!(out[0].equal(&a));
    }
}

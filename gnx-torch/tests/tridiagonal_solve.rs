mod support;
use gnx_expr::{ValueInfo, pass::Builder};
use gnx_math::{ArrayInfo, DType, Shape, ops::linalg::tridiagonal_solve};
use gnx_torch::TorchDevice;
use tch::{CModule, Kind, Tensor};

fn compile(dims: &[usize], cols: usize, k: DType, perturb: bool) -> CModule {
    let ty = |dims: &[usize]| ValueInfo::new(ArrayInfo::new(Shape::fixed(dims.iter().copied()), k));
    let rhs = ty(&[dims, &[cols]].concat());
    let mut b = Builder::new();
    let d = std::array::from_fn::<_, 3, _>(|_| b.input(ty(dims)));
    let r = b.input(rhs.clone());
    let x = b.emit1(tridiagonal_solve(perturb), &[d[0], d[1], d[2], r], rhs);
    let p = TorchDevice::cpu()
        .unwrap()
        .compile(&b.finish(vec![x]))
        .unwrap();
    support::reload(p.modules().next().unwrap())
}
fn dense(dl: &Tensor, d: &Tensor, du: &Tensor) -> Tensor {
    let n = *d.size().last().unwrap();
    d.diag_embed(0, -2, -1)
        + dl.narrow(-1, 1, n - 1).diag_embed(-1, -2, -1)
        + du.narrow(-1, 0, n - 1).diag_embed(1, -2, -1)
}
#[test]
fn tridiagonal_solve_replays_fresh_batches_with_multiple_rhs_and_pivoting() {
    let row = |v: &[f64]| Tensor::from_slice(v).reshape([2, 4]);
    // Padding is deliberately NaN. The first system requires a row swap;
    // the second does not, and the two systems switch places on replay.
    let dl = row(&[f64::NAN, 2., -1., 3., f64::NAN, -0.5, 2., 0.25]);
    let du = row(&[3., 1., -2., f64::NAN, 1., 2., -1., f64::NAN]);
    let d = row(&[0., 1., 6., -4., 4., 5., 6., 7.]);
    let rhs = Tensor::arange(24, (Kind::Double, tch::Device::Cpu)).reshape([2, 4, 3]) - 5.;
    for (dtype, kind) in [
        (DType::F64, Kind::Double),
        (DType::C128, Kind::ComplexDouble),
    ] {
        let m = compile(&[2, 4], 3, dtype, false);
        let val = |a: Tensor, t: f64| {
            if kind == Kind::ComplexDouble {
                Tensor::complex(&a, &(&a * t))
            } else {
                a
            }
        };
        for j in 0..2 {
            let fresh = |t: &Tensor| {
                if j == 0 {
                    t.shallow_clone()
                } else {
                    t.flip([0])
                }
            };
            let l = val(fresh(&dl) * (1. + 0.125 * j as f64), -0.25);
            let u = val(fresh(&du), 0.75);
            let diag = val(fresh(&d), 0.125);
            let b = val(&rhs * (j + 1) as f64 + 0.25 * j as f64, -0.5);
            let x = m.forward_ts(&[&l, &diag, &u, &b]).unwrap();
            assert_eq!(x.kind(), kind);
            assert_eq!(x.size(), [2, 4, 3]);
            let product = dense(&l, &diag, &u).matmul(&x);
            assert!(
                product.allclose(&b, 1e-11, 1e-11, false),
                "A X - B: {:?}",
                (product - b).abs().max()
            );
        }
    }
}
#[test]
fn tridiagonal_solve_perturbs_only_singular_batch_members_after_reload() {
    let off = Tensor::zeros([2, 2], (Kind::Double, tch::Device::Cpu));
    let rhs = Tensor::from_slice(&[1f64, 2., 3., 4., 5., 6., 7., 8.]).reshape([2, 2, 2]);
    for perturb in [false, true] {
        let m = compile(&[2, 2], 2, DType::F64, perturb);
        for singular in [0i64, 1] {
            let mut d = Tensor::from_slice(&[2f64, 0., 1., 1.]).reshape([2, 2]);
            if singular == 1 {
                d = d.flip([0])
            }
            let x = m.forward_ts(&[&off, &d, &off, &rhs]).unwrap();
            assert!(x.get(1 - singular).equal(&rhs.get(1 - singular)));
            if perturb {
                let shifted = d.get(singular) + 2. * f64::EPSILON;
                assert!((shifted.unsqueeze(-1) * x.get(singular)).allclose(
                    &rhs.get(singular),
                    1e-14,
                    1e-14,
                    false
                ));
            } else {
                assert!(bool::try_from(x.get(singular).isnan().all()).unwrap());
            }
        }
    }
}
#[test]
fn tridiagonal_solve_handles_singleton_and_empty_rhs() {
    for (n, k) in [(1usize, 2usize), (0, 3), (3, 0)] {
        let m = compile(&[2, n], k, DType::F32, false);
        let d = Tensor::full([2, n as i64], 4., (Kind::Float, tch::Device::Cpu));
        let b = Tensor::full([2, n as i64, k as i64], 8., (Kind::Float, tch::Device::Cpu));
        let x = m.forward_ts(&[&d, &d, &d, &b]).unwrap();
        assert!(x.equal(&(&b / 4.)));
    }
}

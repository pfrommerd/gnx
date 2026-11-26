mod support;
use gnx_base::pass::Builder;
use gnx_expr::VarInfo;
use gnx_math::{
    ArrayInfo, DType, Shape,
    ops::{add_op, elementwise, linalg::lu_pivots_to_permutation},
};
use gnx_torch::TorchDevice;
use gnx_transforms::cf::{cond_op, switch_op, while_op};
use tch::Tensor;

fn info(shape: &[usize], k: DType) -> VarInfo {
    VarInfo::new(ArrayInfo::new(Shape::fixed(shape.iter().copied()), k))
}
#[test]
fn reloaded_cond_executes_only_the_selected_branch() {
    let mut b = Builder::new();
    let pred = b.input(info(&[], DType::Bool));
    let pivot = info(&[1], DType::I32);
    let left = b.input(pivot.clone());
    let right = b.input(pivot);
    let row = info(&[3], DType::I32);
    let arg = b.input(row.clone());
    let mut branches = vec![];
    for (captured, op) in [(left, add_op()), (right, elementwise::sub())] {
        branches.push(
            b.region::<std::io::Error>(&[captured], std::slice::from_ref(&row), |b, cap, args| {
                let perm = b.emit1(lu_pivots_to_permutation(3), cap, row.clone());
                Ok(vec![b.emit1(op, &[args[0], perm], row.clone())])
            })
            .unwrap(),
        );
    }
    let out = b.emit(cond_op(), &[pred, arg], [row], branches);
    let p = futures_executor::block_on(
        TorchDevice::cpu(gnx_base::cache::Cache::in_memory())
            .unwrap()
            .compile(&b.finish(out).build().unwrap()),
    )
    .unwrap();
    assert_eq!(p.modules().len(), 1);
    let m = support::reload(p.modules().next().unwrap());
    let run = |flag: bool, l: i32, r: i32| {
        m.forward_ts(&[
            Tensor::from(flag),
            Tensor::from_slice(&[l]),
            Tensor::from_slice(&[r]),
            Tensor::from_slice(&[10i32, 20, 30]),
        ])
    };
    // The inactive branch would index past the end of a three-element row.
    assert_eq!(
        Vec::<i32>::try_from(run(true, 2, 9).unwrap()).unwrap(),
        [12, 21, 30]
    );
    assert_eq!(
        Vec::<i32>::try_from(run(false, 9, 1).unwrap()).unwrap(),
        [9, 20, 28]
    );
    for error in [run(false, 2, 9).unwrap_err(), run(true, 9, 1).unwrap_err()] {
        assert!(error.to_string().contains("out of bounds"), "{error}");
    }
    assert_eq!(
        Vec::<i32>::try_from(run(true, 0, 9).unwrap()).unwrap(),
        [10, 21, 32]
    );
}
#[test]
fn reloaded_switch_clamps_and_executes_only_one_branch() {
    let mut b = Builder::new();
    let selector = b.input(info(&[], DType::I32));
    let piv = info(&[1], DType::I32);
    let pivots = std::array::from_fn::<_, 3, _>(|_| b.input(piv.clone()));
    let row = info(&[3], DType::I32);
    let branches = pivots
        .iter()
        .map(|&v| {
            b.region::<std::io::Error>(&[v], &[], |b, cap, _| {
                Ok(vec![b.emit1(lu_pivots_to_permutation(3), cap, row.clone())])
            })
            .unwrap()
        })
        .collect();
    let out = b.emit(switch_op(), &[selector], [row], branches);
    let p = futures_executor::block_on(
        TorchDevice::cpu(gnx_base::cache::Cache::in_memory())
            .unwrap()
            .compile(&b.finish(out).build().unwrap()),
    )
    .unwrap();
    let m = support::reload(p.modules().next().unwrap());
    for (i, which, expected) in [
        (-4, 0, [0, 1, 2]),
        (0, 0, [0, 1, 2]),
        (1, 1, [1, 0, 2]),
        (2, 2, [2, 1, 0]),
        (75, 2, [2, 1, 0]),
    ] {
        let mut args = vec![Tensor::from(i)];
        args.extend((0..3).map(|j| Tensor::from_slice(&[if j == which { j } else { 9 }])));
        let out = m.forward_ts(&args).unwrap();
        assert_eq!(Vec::<i32>::try_from(out).unwrap(), expected)
    }
}

#[test]
fn reloaded_while_handles_zero_one_and_many_iterations_with_live_captures() {
    let mut b = Builder::new();
    let val = info(&[], DType::I32);
    let start = b.input(val.clone());
    let limit = b.input(val.clone());
    let step = b.input(val.clone());
    let types = [val.clone(), val.clone()];
    let cond = b
        .region::<std::io::Error>(&[limit], &types, |b, c, a| {
            Ok(vec![b.emit1(
                elementwise::lt(),
                &[a[0], c[0]],
                info(&[], DType::Bool),
            )])
        })
        .unwrap();
    let body = b
        .region::<std::io::Error>(&[step, start], &types, |b, c, a| {
            let idx = b.emit1(add_op(), &[a[0], c[0]], val.clone());
            let term = b.emit1(add_op(), &[a[0], c[1]], val.clone());
            let acc = b.emit1(add_op(), &[a[1], term], val.clone());
            Ok(vec![idx, acc])
        })
        .unwrap();
    // start initializes two independent carries and is a loop-invariant capture.
    let carry = b.emit(while_op(), &[start, start], types, vec![cond, body]);
    let result = b.emit1(elementwise::sub(), &[carry[1], carry[0]], val);
    let p = futures_executor::block_on(
        TorchDevice::cpu(gnx_base::cache::Cache::in_memory())
            .unwrap()
            .compile(&b.finish(vec![result]).build().unwrap()),
    )
    .unwrap();
    let m = support::reload(p.modules().next().unwrap());
    for (s, l, k) in [(4, 2, 1), (3, 5, 2), (2, 8, 2), (-3, 12, 1)] {
        let y = m
            .forward_ts(&[Tensor::from(s), Tensor::from(l), Tensor::from(k)])
            .unwrap();
        let (mut i, mut a) = (s, s);
        while i < l {
            a += i + s;
            i += k;
        }
        assert_eq!(i32::try_from(y).unwrap(), a - i);
    }
}

#[test]
fn failed_branch_lowering_preserves_error_and_restores_tracer() {
    use gnx_math::shard::{Sharding, constraint_op};
    let build = |spec| {
        let mut b = Builder::new();
        let pred = b.input(info(&[], DType::Bool));
        let ty = info(&[1], DType::I32);
        let x = b.input(ty.clone());
        let yes = b
            .region::<std::io::Error>(&[], std::slice::from_ref(&ty), |b, _, args| {
                Ok(vec![b.emit1(constraint_op(spec), args, ty.clone())])
            })
            .unwrap();
        let no = b
            .region::<std::io::Error>(&[], std::slice::from_ref(&ty), |_, _, args| {
                Ok(args.to_vec())
            })
            .unwrap();
        let out = b.emit(cond_op(), &[pred, x], [ty], vec![yes, no]);
        b.finish(out).build().unwrap()
    };
    let device = TorchDevice::cpu(gnx_base::cache::Cache::in_memory()).unwrap();
    let err = futures_executor::block_on(
        device.compile(&build(Sharding::new(vec![vec!["mesh".into()]]).unwrap())),
    )
    .err()
    .unwrap();
    assert_eq!(err.kind(), std::io::ErrorKind::Unsupported);
    assert!(err.to_string().contains("device mesh"));
    let p = futures_executor::block_on(device.compile(&build(Sharding::replicated(1)))).unwrap();
    let m = support::reload(p.modules().next().unwrap());
    let out = m
        .forward_ts(&[Tensor::from(true), Tensor::from_slice(&[8i32])])
        .unwrap();
    assert_eq!(Vec::<i32>::try_from(out).unwrap(), [8]);
}

#[test]
fn nested_cond_preserves_captures_and_forwarded_result_bindings() {
    let mut b = Builder::new();
    let flag = info(&[], DType::Bool);
    let val = info(&[], DType::I32);
    let outer = b.input(flag.clone());
    let inner = b.input(flag);
    let x = b.input(val.clone());
    let y = b.input(val.clone());
    let yes = b
        .region::<std::io::Error>(&[inner, y], std::slice::from_ref(&val), |b, c, a| {
            let same = b.region::<std::io::Error>(&[], std::slice::from_ref(&val), |_, _, a| {
                Ok(vec![a[0], a[0]])
            })?;
            let other =
                b.region::<std::io::Error>(&[c[1]], std::slice::from_ref(&val), |_, c, a| {
                    Ok(vec![c[0], a[0]])
                })?;
            Ok(b.emit(
                cond_op(),
                &[c[0], a[0]],
                [val.clone(), val.clone()],
                vec![same, other],
            ))
        })
        .unwrap();
    let no = b
        .region::<std::io::Error>(&[y], std::slice::from_ref(&val), |_, c, _| {
            Ok(vec![c[0], c[0]])
        })
        .unwrap();
    let pair = b.emit(
        cond_op(),
        &[outer, x],
        [val.clone(), val.clone()],
        vec![yes, no],
    );
    let delta = b.emit1(elementwise::sub(), &pair, val.clone());
    let out = b.emit1(add_op(), &[delta, x], val);
    let p = futures_executor::block_on(
        TorchDevice::cpu(gnx_base::cache::Cache::in_memory())
            .unwrap()
            .compile(&b.finish(vec![out]).build().unwrap()),
    )
    .unwrap();
    let m = support::reload(p.modules().next().unwrap());
    for (outer, inner, expected) in [
        (true, true, 3),
        (true, false, 7),
        (false, true, 3),
        (false, false, 3),
    ] {
        let out = m
            .forward_ts(&[
                Tensor::from(outer),
                Tensor::from(inner),
                Tensor::from(3i32),
                Tensor::from(7i32),
            ])
            .unwrap();
        assert_eq!(i32::try_from(out).unwrap(), expected);
    }
}

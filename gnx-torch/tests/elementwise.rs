use gnx_expr::{Eqn, Expr, Op, VarInfo, VarScope};
use gnx_math::{
    ArrayInfo, DType, Shape,
    ops::{elementwise as ew, select_op},
};
use gnx_torch::{TorchDevice, TorchExecutable};
use tch::{CModule, Kind, Tensor};

fn info(k: DType, shape: &[usize]) -> VarInfo {
    VarInfo::new(ArrayInfo::new(Shape::fixed(shape.iter().copied()), k))
}
fn compile(op: Op, inputs: &[VarInfo], output: VarInfo) -> std::io::Result<TorchExecutable> {
    let mut vars = VarScope::new();
    let args: Vec<_> = inputs
        .iter()
        .map(|info| (vars.create_var(), info.clone()))
        .collect();
    let out = vars.create_var();
    let eqn = Eqn::new(
        op,
        args.iter().map(|(v, _)| *v).collect(),
        vec![(out, output)],
    );
    TorchDevice::cpu()?.compile(&Expr::new(vec![], args, vec![eqn], vec![out]))
}
fn binary(op: Op, kind: DType, result: DType, n: usize) -> CModule {
    let ty = info(kind, &[n]);
    let program = compile(op, &[ty.clone(), ty], info(result, &[n])).unwrap();
    // Run through TorchScript without GNX's execution wrapper.
    reload(program.modules().next().unwrap())
}
mod support;
use support::reload;

fn bitcast(input: DType, dims: &[usize], output: DType, result_dims: &[usize]) -> CModule {
    let p = compile(
        gnx_math::ops::dtype::bitcast_convert_type(),
        &[info(input, dims)],
        info(output, result_dims),
    )
    .unwrap();
    reload(p.modules().next().unwrap())
}

#[test]
fn bitcast_equal_width_replays_exact_float_encodings() {
    let m = bitcast(DType::F32, &[2, 2], DType::I32, &[2, 2]);
    for bits in [
        [0x80000000u32, 0x7f800001, 0xffc12345, 1],
        [0x3f800000, 0xffffffff, 0x00800000, 0],
    ] {
        let input = Tensor::from_slice(&bits.map(f32::from_bits)).reshape([2, 2]);
        let out = m.forward_ts(&[input]).unwrap();
        assert_eq!(out.kind(), Kind::Int);
        assert_eq!(out.size(), [2, 2]);
        assert_eq!(
            Vec::<i32>::try_from(out.flatten(0, -1)).unwrap(),
            bits.map(|v| v as i32)
        );
    }
}

#[test]
fn bitcast_width_changes_replay_gnx_minor_axis_and_endianness() {
    let split = bitcast(DType::I32, &[2, 2], DType::U8, &[2, 2, 4]);
    let join = bitcast(DType::U8, &[2, 2, 4], DType::I32, &[2, 2]);
    // A transpose forces logical (not underlying storage) element order.
    let words = [0x78563412i32, i32::MIN, -1, 0x01020304];
    let x = Tensor::from_slice(&words).reshape([2, 2]).transpose(0, 1);
    let out = split.forward_ts(&[x]).unwrap();
    assert_eq!(out.kind(), Kind::Uint8);
    assert_eq!(out.size(), [2, 2, 4]);
    let expected = [words[0], words[2], words[1], words[3]];
    let bytes = expected
        .into_iter()
        .flat_map(i32::to_le_bytes)
        .collect::<Vec<_>>();
    assert_eq!(Vec::<u8>::try_from(out.flatten(0, -1)).unwrap(), bytes);
    // A contiguous slice can still start one byte off a 32-bit boundary.
    let backing = Tensor::from_slice(&[&[0u8][..], &bytes].concat());
    let shifted = backing.narrow(0, 1, 16).reshape([2, 2, 4]);
    let out = join.forward_ts(&[shifted]).unwrap();
    assert_eq!(out.kind(), Kind::Int);
    assert_eq!(out.size(), [2, 2]);
    assert_eq!(Vec::<i32>::try_from(out.flatten(0, -1)).unwrap(), expected);
}

#[test]
fn bitcast_scalar_and_empty_shapes_after_reload() {
    let split = bitcast(DType::I32, &[], DType::I16, &[2]);
    let join = bitcast(DType::I16, &[2], DType::I32, &[]);
    let halves = split.forward_ts(&[Tensor::from(0x1234abcdi32)]).unwrap();
    assert_eq!(halves.size(), [2]);
    assert_eq!(
        Vec::<i16>::try_from(&halves).unwrap(),
        [0xabcdu16 as i16, 0x1234]
    );
    let word = join.forward_ts(&[halves]).unwrap();
    assert_eq!(word.dim(), 0);
    assert_eq!(i32::try_from(word).unwrap(), 0x1234abcd);
    let join = bitcast(DType::U8, &[0, 4], DType::F32, &[0]);
    let out = join
        .forward_ts(&[Tensor::zeros([0, 4], (Kind::Uint8, tch::Device::Cpu))])
        .unwrap();
    assert_eq!(out.kind(), Kind::Float);
    assert_eq!(out.size(), [0]);
}

#[test]
fn bitcast_rejects_reshape_with_same_byte_count_but_wrong_minor_axis() {
    for (input, result) in [
        (info(DType::U8, &[2, 8]), info(DType::I32, &[2, 2])),
        (info(DType::I32, &[2]), info(DType::U8, &[8])),
    ] {
        let e = compile(
            gnx_math::ops::dtype::bitcast_convert_type(),
            &[input],
            result,
        )
        .err()
        .unwrap();
        assert_eq!(e.kind(), std::io::ErrorKind::InvalidInput);
    }
}

#[test]
fn total_order_replays_ieee_encodings_after_reload() {
    // Explicit ascending IEEE binary32 order, including signaling and quiet NaNs.
    let order = [
        0xffc00002u32,
        0xffc00001,
        0xff800002,
        0xff800001,
        0xff800000,
        0xbf800000,
        0x80000000,
        0,
        0x3f800000,
        0x7f800000,
        0x7f800001,
        0x7f800002,
        0x7fc00001,
        0x7fc00002,
    ];
    let n = order.len();
    let lhs: Vec<_> = order
        .iter()
        .flat_map(|&x| std::iter::repeat_n(f32::from_bits(x), n))
        .collect();
    let rhs: Vec<_> = order
        .iter()
        .cycle()
        .take(n * n)
        .map(|&x| f32::from_bits(x))
        .collect();
    let a = Tensor::from_slice(&lhs);
    let b = Tensor::from_slice(&rhs);
    for (i, op) in [ew::eq_to(), ew::lt_to(), ew::le_to()]
        .into_iter()
        .enumerate()
    {
        let m = binary(op, DType::F32, DType::Bool, n * n);
        for (x, y, swap) in [(&a, &b, false), (&b, &a, true)] {
            let v = m.forward_ts(&[x, y]).unwrap();
            assert_eq!(v.kind(), Kind::Bool);
            let got = Vec::<bool>::try_from(v).unwrap();
            for (k, val) in got.into_iter().enumerate() {
                let (l, r) = if swap { (k % n, k / n) } else { (k / n, k % n) };
                assert_eq!(
                    val,
                    match i {
                        0 => l == r,
                        1 => l < r,
                        _ => l <= r,
                    },
                    "op {i}: {:08x} vs {:08x}",
                    order[l],
                    order[r]
                );
            }
        }
    }
}

#[test]
fn total_order_preserves_payload_bits_at_each_supported_float_width() {
    // Each pair differs only in the least significant fraction bit. Casting
    // f64 to f32 or canonicalizing a low-precision NaN would lose the order.
    for (dtype, kind, base, sign, word) in [
        (DType::F16, Kind::Half, 0x7e00i64, 0x8000i64, Kind::Int16),
        (DType::BF16, Kind::BFloat16, 0x7fc0, 0x8000, Kind::Int16),
        (
            DType::F64,
            Kind::Double,
            0x7ff8000000000000,
            i64::MIN,
            Kind::Int64,
        ),
    ] {
        let bits = |values: &[i64]| {
            Tensor::from_slice(values)
                .to_kind(word)
                .f_view_copy_dtype(kind)
                .unwrap()
        };
        let a = bits(&[base + 1, sign | (base + 2), sign, base + 1]);
        let b = bits(&[base + 2, sign | (base + 1), 0, base + 1]);
        for (op, want) in [
            (ew::eq_to(), [false, false, false, true]),
            (ew::lt_to(), [true, true, true, false]),
            (ew::le_to(), [true; 4]),
        ] {
            let y = binary(op, dtype, DType::Bool, 4)
                .forward_ts(&[&a, &b])
                .unwrap();
            assert_eq!(Vec::<bool>::try_from(y).unwrap(), want);
        }
    }
}

#[test]
fn polygamma_replays_distinct_order_tensors_against_reference_values() {
    // Closed forms at 1, 1/2, and their shifts (Abramowitz & Stegun 6.4).
    let gamma = 0.5772156649015329_f64;
    let zeta3 = 1.2020569031595943_f64;
    let pi = std::f64::consts::PI;
    let m = binary(
        gnx_math::ops::special::polygamma(),
        DType::F64,
        DType::F64,
        6,
    );
    let x = Tensor::from_slice(&[1f64, 0.5, 2., -0.5, 1., 1.]);
    for (orders, expected) in [
        (
            [0., 1., 2., 1., 3., 4.],
            [
                -gamma,
                pi * pi / 2.,
                2. - 2. * zeta3,
                pi * pi / 2. + 4.,
                pi.powi(4) / 15.,
                -24. * 1.03692775514337,
            ],
        ),
        (
            [2., 0., 1., 2., 0., 5.],
            [
                -2. * zeta3,
                -gamma - 2. * 2f64.ln(),
                pi * pi / 6. - 1.,
                16. - 14. * zeta3,
                -gamma,
                120. * pi.powi(6) / 945.,
            ],
        ),
    ] {
        let y = m.forward_ts(&[&Tensor::from_slice(&orders), &x]).unwrap();
        assert_eq!(y.kind(), Kind::Double);
        for (got, want) in Vec::<f64>::try_from(y).unwrap().into_iter().zip(expected) {
            assert!(
                (got - want).abs() < 1e-12 * (1. + want.abs()),
                "{got} vs {want}"
            );
        }
    }
}

#[test]
fn polygamma_replays_invalid_orders_and_poles_per_lane() {
    let m = binary(
        gnx_math::ops::special::polygamma(),
        DType::F32,
        DType::F32,
        6,
    );
    let n = Tensor::from_slice(&[-1f32, 1.5, f32::NAN, 1., 2., 0.]);
    let x = Tensor::from_slice(&[2f32, 2., 2., -2., -2., 1.]);
    let y = m.forward_ts(&[n, x]).unwrap();
    assert_eq!(y.kind(), Kind::Float);
    let vals = Vec::<f32>::try_from(y).unwrap();
    assert!(vals[..3].iter().all(|v| v.is_nan()));
    assert_eq!(vals[3], f32::INFINITY);
    assert!(vals[4].is_nan());
    assert!((vals[5] + 0.57721566).abs() < 1e-6);
}

#[test]
fn igamma_shape_gradient_replays_against_high_precision_derivatives() {
    // mpmath 1.3.0, mp.dps=90; mp.diff(lambda s:
    // mp.gammainc(s, 0, x, regularized=True), a). Neither tensor is constant.
    let m = binary(
        gnx_math::ops::special::igamma_grad_a(),
        DType::F64,
        DType::F64,
        6,
    );
    for (a, x, expected) in [
        (
            [0.125f64, 0.5, 2.75, 9.5, 125., 12.],
            [0.001f64, 2.25, 1.125, 11.75, 127., 80.],
            [
                -2.919245319386072816,
                -0.104126005929797818,
                -0.162637522951417370,
                -0.093049196895470387,
                -0.034867256730562466,
                -8.77768614984147011e-22,
            ],
        ),
        (
            [0.000001, 1.25, 4.5, 20., 1000., 32.],
            [0.125, 0.75, 10., 20., 999., 1.],
            [
                -1.62342441852979582,
                -0.421569887234410577,
                -0.018587380363503416,
                -0.089579205539122536,
                -0.012616712415226564,
                -5.02044244564369938e-36,
            ],
        ),
        (
            [0.000001, 16., 2., 8192., 0.03125, 2.5],
            [1.000001, 16., 64., 8300., 1e-13, 23.],
            [
                -0.219384015465948418,
                -0.100257271302648113,
                -3.91108221516862667e-26,
                -0.00216213497082615826,
                -11.7404778878893233,
                -2.24929810615781875e-8,
            ],
        ),
    ] {
        let y = m
            .forward_ts(&[Tensor::from_slice(&a), Tensor::from_slice(&x)])
            .unwrap();
        assert_eq!(y.kind(), Kind::Double);
        for (i, (got, want)) in Vec::<f64>::try_from(y)
            .unwrap()
            .into_iter()
            .zip(expected)
            .enumerate()
        {
            assert!(
                (got / want - 1.).abs() < 1e-11,
                "a={}, x={}: {got} vs {want}",
                a[i],
                x[i]
            );
        }
    }
}

#[test]
fn igamma_shape_gradient_handles_boundaries_per_lane_after_reload() {
    let m = binary(
        gnx_math::ops::special::igamma_grad_a(),
        DType::F32,
        DType::F32,
        7,
    );
    let a = Tensor::from_slice(&[1f32, 2., 0., -1., 1., f32::NAN, 1.]);
    let x = Tensor::from_slice(&[0f32, f32::INFINITY, 0., 1., -1., 1., f32::NAN]);
    let y = m.forward_ts(&[a, x]).unwrap();
    assert_eq!(y.kind(), Kind::Float);
    let v = Vec::<f32>::try_from(y).unwrap();
    assert_eq!(v[..2], [0., 0.]);
    assert!(v[2..].iter().all(|x| x.is_nan()));
}

fn beta_module(dtype: DType, n: usize) -> CModule {
    let t = info(dtype, &[n]);
    let p = compile(
        gnx_math::ops::special::regularized_incomplete_beta(),
        &[t.clone(), t.clone(), t.clone()],
        t,
    )
    .unwrap();
    reload(p.modules().next().unwrap())
}
#[test]
fn incomplete_beta_replays_fresh_parameters_and_both_tails() {
    // mpmath 1.3.0 with mp.dps=100: betainc(a,b,0,x,regularized=True).
    let m = beta_module(DType::F64, 7);
    for (a, b, x, reference) in [
        (
            [1.25f64, 0.25, 32., 3.5, 200., 6., 0.5],
            [2.5f64, 0.75, 4.5, 8.25, 300., 3., 4.],
            [0.125f64, 0.0625, 0.03125, 0.875, 0.4, 0., 1.],
            [
                0.196029963448887297_f64,
                0.451596461534647821,
                1.24019082938314721e-44,
                0.999997503327850060,
                0.502428616319931996,
                0.,
                1.,
            ],
        ),
        (
            [2.5, 0.75, 18.25, 4.5, 125., 0.03125, 4.],
            [1.25, 0.25, 41.5, 32., 60., 0.125, 7.],
            [0.75, 0.5, 0.15625, 0.625, 0.6875, 2f64.powi(-30), 0.984375],
            [
                0.584183140656579937,
                0.219450073830409936,
                0.00229360095397234652,
                0.999999999902812985,
                0.628130733629732204,
                0.420128941626639941,
                0.999999999973818793,
            ],
        ),
        // Strongly asymmetric shapes, tiny probabilities, and the exact
        // symmetry identity I_(1/2)(a,a)=1/2 at a sharp central peak.
        (
            [10., 0.1, 1e12, 100., 2., 7e-4, 0.01],
            [1e-16, 1e20, 1e12, 1e-8, 1., 4e-3, 1e12],
            [0.99, 1e-20, 0.5, 0.999, 2f64.powi(-500), 0.0625, 4e-12],
            [
                1.86442961952588776e-16,
                0.975872656273672219,
                0.5,
                1.82700360792184214e-8,
                9.33263618503218879e-302,
                0.849455767328477612,
                0.999961391309647079,
            ],
        ),
    ] {
        let y = m
            .forward_ts(&[
                Tensor::from_slice(&a),
                Tensor::from_slice(&b),
                Tensor::from_slice(&x),
            ])
            .unwrap();
        assert_eq!(y.kind(), Kind::Double);
        for (i, (v, r)) in Vec::<f64>::try_from(y)
            .unwrap()
            .into_iter()
            .zip(reference)
            .enumerate()
        {
            if x[i] == 0. || x[i] == 1. {
                assert_eq!(v, x[i]);
            } else {
                let tail = r.min(1. - r);
                assert!(
                    (v - r).abs() <= 2e-12 * tail + f64::EPSILON * r,
                    "I({},{}; {}) = {v} vs {r}",
                    a[i],
                    b[i],
                    x[i]
                );
            }
        }
    }
}

#[test]
fn incomplete_beta_replays_invalid_lanes_and_f32_endpoints() {
    let m = beta_module(DType::F32, 8);
    let args = [
        [0f32, 1., 1., 1., 1., 1., 2., 1.],
        [1., -1., 1., 1., 1., 3., 2., 4.],
        [0., 1., -0.1, 1.1, f32::NAN, 0., 0.5, 1.],
    ]
    .map(|v| Tensor::from_slice(&v));
    let y = m.forward_ts(&args).unwrap();
    assert_eq!(y.kind(), Kind::Float);
    let vals = Vec::<f32>::try_from(y).unwrap();
    assert!(vals[..5].iter().all(|v| v.is_nan()));
    assert_eq!(vals[5..], [0., 0.5, 1.]);
}

fn precision_trace(dtype: DType, length: usize, e: u8, m: u8) -> CModule {
    let t = info(dtype, &[length]);
    let p = compile(
        gnx_math::ops::dtype::reduce_precision(e, m),
        std::slice::from_ref(&t),
        t.clone(),
    )
    .unwrap();
    reload(p.modules().next().unwrap())
}

#[test]
fn reduce_precision_rounds_then_clips_after_reload() {
    // E3M2: smallest normal 0.25, largest finite 14. The values just below
    // 0.25 and 16 test mantissa rounding before exponent clipping.
    let module = precision_trace(DType::F32, 12, 3, 2);
    for sign in [1f32, -1.] {
        let inputs = [
            1.125,
            1.375,
            1.625,
            1.875,
            0.125,
            0.234375,
            14.,
            15.,
            16.,
            0.,
            f32::INFINITY,
            f32::NAN,
        ]
        .map(|x| x * sign);
        let expected = [
            1.,
            1.5,
            1.5,
            2.,
            0.,
            0.25,
            14.,
            f32::INFINITY,
            f32::INFINITY,
            0.,
            f32::INFINITY,
            f32::NAN,
        ]
        .map(|x| x * sign);
        let y = module.forward_ts(&[Tensor::from_slice(&inputs)]).unwrap();
        assert_eq!(y.kind(), Kind::Float);
        for (a, b) in Vec::<f32>::try_from(y).unwrap().into_iter().zip(expected) {
            if b.is_nan() {
                assert!(a.is_nan())
            } else {
                assert_eq!(a.to_bits(), b.to_bits())
            }
        }
    }
}

#[test]
fn reduce_precision_zero_mantissa_and_native_exponent_after_reload() {
    let p = precision_trace(DType::F64, 5, 11, 0);
    let y = p
        .forward_ts(&[Tensor::from_slice(&[1.5f64, 3., -3., -f64::NAN, f64::MAX])])
        .unwrap();
    assert_eq!(
        Vec::<f64>::try_from(y).unwrap(),
        [2., 2., -2., f64::INFINITY, f64::INFINITY]
    );
    let p = precision_trace(DType::F64, 4, 11, 51);
    // With all exponent bits retained, subnormal fractions are rounded rather
    // than flushed. The low bit of each of these source words is discarded.
    let raw = [1, 3, (1u64 << 63) | 1, f64::MIN_POSITIVE.to_bits() - 1];
    let y = p
        .forward_ts(&[Tensor::from_slice(&raw.map(f64::from_bits))])
        .unwrap();
    assert_eq!(
        Vec::<f64>::try_from(y)
            .unwrap()
            .into_iter()
            .map(f64::to_bits)
            .collect::<Vec<_>>(),
        [0, 4, 1u64 << 63, f64::MIN_POSITIVE.to_bits()]
    );
}

#[test]
fn signed_divrem_uses_truncation_and_handles_exceptional_lanes_after_reload() {
    let x = Tensor::from_slice(&[-7i64, 7, -7, 13, i64::MIN, i64::MIN]);
    let y = Tensor::from_slice(&[3i64, -3, -3, 0, -1, 1]);
    for (op, expected) in [
        (ew::div(), [-2, -2, 2, -1, i64::MIN, i64::MIN]),
        (ew::rem(), [-1, 1, -1, 13, 0, 0]),
    ] {
        let result = binary(op, DType::I64, DType::I64, 6)
            .forward_ts(&[&x, &y])
            .unwrap();
        assert_eq!(result.kind(), Kind::Int64);
        assert_eq!(Vec::<i64>::try_from(result).unwrap(), expected);
    }
    let x = Tensor::from_slice(&[255u8, 42]);
    let y = Tensor::from_slice(&[2u8, 0]);
    for (op, expected) in [(ew::div(), [127, 255]), (ew::rem(), [1, 42])] {
        let result = binary(op, DType::U8, DType::U8, 2)
            .forward_ts(&[&x, &y])
            .unwrap();
        assert_eq!(result.kind(), Kind::Uint8);
        assert_eq!(Vec::<u8>::try_from(result).unwrap(), expected);
    }
}

#[test]
fn floating_arithmetic_and_extrema_preserve_special_values() {
    let a = Tensor::from_slice(&[-7.5f32, 0., -0., f32::NAN]);
    let b = Tensor::from_slice(&[2f32, -0., 0., 1.]);
    for (op, expected) in [
        (ew::min(), [-7.5, -0., -0., f32::NAN]),
        (ew::max(), [2., 0., 0., f32::NAN]),
        (ew::sub(), [-9.5, 0., -0., f32::NAN]),
        (ew::rem(), [-1.5, f32::NAN, f32::NAN, f32::NAN]),
        (ew::div(), [-3.75, f32::NAN, f32::NAN, f32::NAN]),
    ] {
        let out = binary(op, DType::F32, DType::F32, 4)
            .forward_ts(&[&a, &b])
            .unwrap();
        assert_eq!(out.kind(), Kind::Float);
        for (got, want) in Vec::<f32>::try_from(out).unwrap().into_iter().zip(expected) {
            if want.is_nan() {
                assert!(got.is_nan())
            } else {
                assert_eq!(got.to_bits(), want.to_bits())
            }
        }
    }
    let ty = info(DType::F32, &[4]);
    for (op, expected) in [(ew::abs(), [7.5f32, 0., 0.]), (ew::neg(), [7.5, -0., 0.])] {
        let p = compile(op, std::slice::from_ref(&ty), ty.clone()).unwrap();
        let out = p.modules().next().unwrap().forward_ts(&[&a]).unwrap();
        let vals = Vec::<f32>::try_from(out).unwrap();
        assert!(vals[3].is_nan());
        assert_eq!(
            vals[..3].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected.map(f32::to_bits)
        );
    }
}

#[test]
fn comparisons_return_predicates_and_select_replays_both_branches() {
    let a = Tensor::from_slice(&[1i32, 2, 4]);
    let b = Tensor::from_slice(&[3i32, 2, 1]);
    let values = info(DType::I32, &[3]);
    let pred = info(DType::Bool, &[3]);
    let p = compile(select_op(), &[pred, values.clone(), values.clone()], values).unwrap();
    let select = reload(p.modules().next().unwrap());
    for (op, expected) in [
        (ew::eq(), [false, true, false]),
        (ew::ne(), [true, false, true]),
        (ew::lt(), [true, false, false]),
        (ew::le(), [true, true, false]),
        (ew::gt(), [false, false, true]),
        (ew::ge(), [false, true, true]),
    ] {
        let mask = binary(op, DType::I32, DType::Bool, 3)
            .forward_ts(&[&a, &b])
            .unwrap();
        assert_eq!(mask.kind(), Kind::Bool);
        assert_eq!(Vec::<bool>::try_from(&mask).unwrap(), expected);
        let out = select.forward_ts(&[&mask, &a, &b]).unwrap();
        assert_eq!(out.kind(), Kind::Int);
        assert_eq!(
            Vec::<i32>::try_from(out).unwrap(),
            [
                if expected[0] { 1 } else { 3 },
                2,
                if expected[2] { 4 } else { 1 }
            ]
        );
    }
}

#[test]
fn invalid_or_unsupported_signatures_fail_at_compile_time() {
    let i = info(DType::I32, &[2]);
    let f = info(DType::F32, &[2]);
    let pred = info(DType::Bool, &[2]);
    let complex = info(DType::C64, &[2]);
    for (op, inputs, result, kind) in [
        (
            ew::sub(),
            vec![i.clone(), f.clone()],
            f.clone(),
            std::io::ErrorKind::InvalidInput,
        ),
        (
            ew::eq(),
            vec![i.clone(), info(DType::I32, &[1])],
            pred.clone(),
            std::io::ErrorKind::InvalidInput,
        ),
        (
            ew::lt(),
            vec![complex.clone(), complex],
            pred.clone(),
            std::io::ErrorKind::Unsupported,
        ),
        (
            ew::div(),
            vec![pred.clone(), pred.clone()],
            pred,
            std::io::ErrorKind::Unsupported,
        ),
        (
            select_op(),
            vec![i, f.clone(), f.clone()],
            f,
            std::io::ErrorKind::InvalidInput,
        ),
    ] {
        assert_eq!(compile(op, &inputs, result).err().unwrap().kind(), kind);
    }
}

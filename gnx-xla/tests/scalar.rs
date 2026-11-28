use futures_executor::block_on;
use gnx::{
    Array, Callable, Eval, IntoScalar,
    array::{ArrayRef, DType},
    grad,
    half::{bf16, f16},
    lax, map, np,
};
use gnx_math::{ExecOpts, TransferPolicy};
use gnx_xla::XlaDevice;
fn device() -> XlaDevice {
    XlaDevice::cpu(gnx_base::cache::Cache::in_memory()).unwrap()
}
fn read(d: &XlaDevice, x: Array) -> Vec<u8> {
    block_on(d.eval(
        &x,
        ExecOpts {
            transfers: TransferPolicy::AllowCopy,
        },
    ))
    .unwrap();
    d.download(x.data().unwrap()).unwrap().bytes().to_vec()
}
#[test]
fn scalar_types_and_bits() {
    let d = device();
    let check = |x: Array, t: DType, expected: &[u8]| {
        assert!(x.shape().dims().is_empty());
        assert_eq!(x.dtype(), t);
        assert_eq!(read(&d, x), expected);
    };
    macro_rules! check {
        ($v:expr,$t:ident) => {{
            let v = $v;
            check(Array::scalar(v), DType::$t, &v.to_ne_bytes());
        }};
    }
    check(Array::scalar(true), DType::Bool, &[1]);
    check!(i8::MIN, I8);
    check!(i16::MIN, I16);
    check!(i32::MAX, I32);
    check!(i64::MIN, I64);
    check!(i64::MAX, I64);
    check!((1_i64 << 53) + 1, I64);
    check!(u8::MAX, U8);
    check!(u16::MAX, U16);
    check!(u32::MAX, U32);
    check!(u64::MAX, U64);
    check!(0.5_f32, F32);
    check(Array::scalar(0.5), DType::F64, &0.5_f64.to_ne_bytes());
    for bits in [1, 0x8000, 0x7bff, 0x7e13] {
        check(
            Array::scalar(f16::from_bits(bits)),
            DType::F16,
            &bits.to_ne_bytes(),
        );
    }
    for bits in [1, 0x8000, 0x7f7f, 0x7fc1] {
        check(
            Array::scalar(bf16::from_bits(bits)),
            DType::BF16,
            &bits.to_ne_bytes(),
        );
    }
    check(
        lax::scalar(u64::MAX, DType::U64),
        DType::U64,
        &u64::MAX.to_ne_bytes(),
    );
}
#[test]
fn checked_conversions() {
    use DType::*;
    assert!(lax::try_scalar(300, U8).is_err());
    assert!(lax::try_scalar(-1, U64).is_err());
    assert!(lax::try_scalar(0.5, I32).is_err());
    assert!(lax::try_scalar(u64::MAX, F64).is_err());
    assert!(lax::try_scalar(2_f64.powi(64), U64).is_err());
    assert!(lax::try_scalar(0.1, F32).is_err());
    assert!(lax::try_scalar(1, KeyPhilox4x32).is_err());
    assert!(Array::scalar(1).try_astype(KeyThreeFry2x32).is_err());
    assert!(
        std::panic::catch_unwind(|| {
            let _ = Array::scalar(3_i64) * 0.5;
        })
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(|| {
            let _ = 300 + Array::scalar(3_u8);
        })
        .is_err()
    );
}
#[test]
fn weak_literals_and_casts_under_transforms() {
    let d = device();
    let x = Array::scalar(4_f32);
    let val = |x: Array| f32::from_ne_bytes(read(&d, x).try_into().unwrap());
    let assert_val = |x: Array, v| {
        assert_eq!(x.dtype(), DType::F32);
        assert_eq!(val(x), v);
    };
    assert_val(&x + 0.5, 4.5);
    assert_val(1 + &x, 5.);
    assert_val(x.clone() - 0.5, 3.5);
    assert_val(0.5 - &x, -3.5);
    assert_val(x.clone() * 2, 8.);
    assert_val(0.5 * x.clone(), 2.);
    assert_val(&x / 2, 2.);
    assert_val(2 / &x, 0.5);
    fn as_array<T: IntoScalar>(x: T) -> Array {
        Array::from(x)
    }
    let b = as_array(bf16::from_f32(2.));
    assert_eq!((&b + 3).dtype(), DType::BF16);
    assert_val((f16::from_f32(3.) * b).astype(DType::F32), 6.);
    let cell = ArrayRef::new(x.clone());
    let cast = cell.astype(DType::F64);
    cell.set(Array::scalar(9_f32));
    assert_val(cast.astype(DType::F32), 4.);
    let a = lax::iota([3], 0, DType::F32);
    let y = map(|x: Array| x.astype(DType::F64)).invoke((a.clone(),));
    assert_eq!(read(&d, y), [0_f64, 1., 2.].map(f64::to_ne_bytes).concat());
    let (dx,) = grad(|x: Array| np::sum(&(x.astype(DType::F64) * 3.0), np::ALL))
        .invoke((a,))
        .graph;
    assert_eq!(dx.dtype(), DType::F32);
    assert_eq!(read(&d, dx), [3_f32; 3].map(f32::to_ne_bytes).concat());
}

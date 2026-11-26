//! Pure, counter-based random draws. A key is a scalar logical dtype, not a
//! mutable generator. Reusing a key repeats a draw. Use `split` or `fold_in`
//! to derive a new key.
use crate::{Array, ArrayInfo, DType, IntoShape, Shape, lax, ops};
use gnx_base::{fixed_dims, invoke_array};
use gnx_expr::{LoweringError, Op};

/// The algorithm is encoded in the key dtype, so it is static under `jit`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Algorithm {
    ThreeFry2x32,
    Philox4x32,
}
impl Algorithm {
    fn dtype(self) -> DType {
        match self {
            Self::ThreeFry2x32 => DType::KeyThreeFry2x32,
            Self::Philox4x32 => DType::KeyPhilox4x32,
        }
    }
}
// Store the two logical words in a single 64-bit element so reshaping,
// indexing and mapping keys never expose a physical word axis. Arithmetic
// uses i64 lanes with explicit masks/shifts on both backends.
fn scalar(n: u32) -> Array {
    lax::scalar(n, DType::I64)
}
fn op(op: Op, x: &Array, rest: &[&Array], dtype: DType) -> Array {
    let mut inputs = vec![x];
    inputs.extend_from_slice(rest);
    invoke_array(op, &inputs, &ArrayInfo::new(x.shape(), dtype))
}
fn binary(opcode: Op, x: &Array, y: &Array) -> Array {
    op(opcode, x, &[y], x.dtype())
}
fn splat(x: &Array, shape: Shape) -> Array {
    lax::broadcast_in_dim(x, shape, &[])
}
fn imm(opcode: Op, x: &Array, n: u32) -> Array {
    binary(opcode, x, &splat(&scalar(n), x.shape()))
}
fn word(x: &Array) -> Array {
    imm(ops::bitwise::and(), x, u32::MAX)
}
fn right(x: &Array, n: u32) -> Array {
    imm(ops::bitwise::shift_right_logical(), x, n)
}
fn left(x: &Array, n: u32) -> Array {
    imm(ops::bitwise::shift_left(), x, n)
}
fn xor(x: &Array, y: &Array) -> Array {
    binary(ops::bitwise::xor(), x, y)
}
fn repr(x: &Array, dtype: DType) -> Array {
    op(ops::dtype::bitcast_convert_type(), x, &[], dtype)
}
fn pack(hi: &Array, lo: &Array, dtype: DType) -> Array {
    repr(&(left(hi, 32) + lo), dtype)
}
fn unpack(k: &Array) -> [Array; 2] {
    let p = repr(k, DType::I64);
    [right(&p, 32), word(&p)]
}
fn check_key(k: &Array) {
    assert!(
        k.dtype().is_key() && k.shape().dims().is_empty(),
        "random draw requires one scalar key; map over batched keys"
    );
}

/// Construct a scalar key from the high and low 32-bit halves of `seed`.
pub fn key(seed: u64, algorithm: Algorithm) -> Array {
    let hi = scalar((seed >> 32) as u32);
    let lo = scalar(seed as u32);
    let [a, b] = match algorithm {
        Algorithm::ThreeFry2x32 => [hi, lo],
        Algorithm::Philox4x32 => {
            let [a, b, _, _] = philox([scalar(0), scalar(0)], [hi, lo, scalar(0), scalar(0)]);
            [a, b]
        }
    };
    pack(&a, &b, algorithm.dtype())
}
/// Export two u32 words per key (most significant word first).
pub fn key_data(keys: &Array) -> Array {
    assert!(keys.dtype().is_key());
    let mut dims = keys.fixed_dims().expect("fixed key shape");
    let axis = dims.len();
    dims.push(1);
    let [hi, lo] = unpack(keys).map(|x| x.astype(DType::U32).reshape(&dims));
    lax::concatenate(&[&hi, &lo], axis)
}
/// Import `U32[..., 2]` as a key array of logical shape `...`.
pub fn wrap_key_data(data: &Array, algorithm: Algorithm) -> Array {
    assert_eq!(data.dtype(), DType::U32);
    let mut dims = data.fixed_dims().expect("fixed key data shape");
    assert_eq!(
        dims.pop(),
        Some(2),
        "key data must have a trailing axis of length two"
    );
    let lane = |i| {
        lax::slice_axis(data, dims.len(), i, i + 1)
            .reshape(&dims)
            .astype(DType::I64)
    };
    pack(&lane(0), &lane(1), algorithm.dtype())
}
// Threefry2x32-20, using JAX's partitionable row-major 64-bit counter.
fn threefry([k0, k1]: [Array; 2], mut a: Array, mut b: Array) -> [Array; 2] {
    let k2 = imm(ops::bitwise::xor(), &xor(&k0, &k1), 0x1bd11bda);
    let ks = [k0, k1, k2];
    a = word(&(a + &ks[0]));
    b = word(&(b + &ks[1]));
    for (i, r) in [13, 15, 26, 6, 17, 29, 16, 24]
        .into_iter()
        .cycle()
        .take(20)
        .enumerate()
    {
        a = word(&(a + &b));
        b = xor(&a, &(word(&left(&b, r)) + right(&b, 32 - r)));
        if i % 4 == 3 {
            let s = (i + 1) / 4;
            a = word(&(a + &ks[s % 3]));
            b = word(&(b + &ks[(s + 1) % 3] + scalar(s as u32)));
        }
    }
    [a, b]
}
// Random123 Philox4x32-10. JAX assigns the counter lanes at the call site.
fn philox([mut k0, mut k1]: [Array; 2], [mut a, mut b, mut c, mut d]: [Array; 4]) -> [Array; 4] {
    for i in 0..10 {
        let p = a * scalar(0xd2511f53);
        let q = c * scalar(0xcd9e8d57);
        a = xor(&xor(&right(&q, 32), &b), &k0);
        c = xor(&xor(&right(&p, 32), &d), &k1);
        b = word(&q);
        d = word(&p);
        if i < 9 {
            k0 = word(&(k0 + scalar(0x9e3779b9)));
            k1 = word(&(k1 + scalar(0xbb67ae85)));
        }
    }
    [a, b, c, d]
}
#[derive(Clone, Copy)]
enum Purpose {
    DeriveKey,
    RandomBits,
}
fn hash(k: &Array, hi: Array, lo: Array, purpose: Purpose) -> [Array; 2] {
    check_key(k);
    let words = unpack(k).map(|x| splat(&x, hi.shape()));
    match k.dtype() {
        DType::KeyThreeFry2x32 => threefry(words, hi, lo),
        DType::KeyPhilox4x32 => {
            let z = splat(&scalar(0), hi.shape());
            match purpose {
                Purpose::DeriveKey => {
                    let [a, b, _, _] = philox(words, [z.clone(), z, hi, lo]);
                    [a, b]
                }
                Purpose::RandomBits => {
                    let [a, b, c, d] = philox(words, [hi, lo, z.clone(), z]);
                    [xor(&a, &b), xor(&c, &d)]
                }
            }
        }
        _ => unreachable!(),
    }
}
fn block(k: &Array, shape: &Shape, purpose: Purpose) -> Result<[Array; 2], LoweringError> {
    let dims = fixed_dims(&ArrayInfo::new(shape, DType::I64))?;
    let n = dims
        .into_iter()
        .try_fold(1usize, usize::checked_mul)
        .filter(|&n| i64::try_from(n).is_ok())
        .ok_or_else(|| gnx_base::ops::signature("random shape overflow"))?;
    let idx = lax::iota([n], 0, DType::I64).reshape(shape);
    Ok(hash(k, right(&idx, 32), word(&idx), purpose))
}
/// Derive `n` keys, with logical shape `[n]` and the parent's dtype.
pub fn split(k: &Array, n: usize) -> Array {
    let [a, b] = block(k, &[n].into_shape(), Purpose::DeriveKey).expect("random::split");
    pack(&a, &b, k.dtype())
}
/// Return the two scalar elements of `split(k, 2)`.
pub fn split2(k: &Array) -> (Array, Array) {
    let keys = split(k, 2);
    (keys.at(0), keys.at(1))
}
fn fold(k: &Array, tag: Array) -> Array {
    let [a, b] = hash(k, scalar(0), tag, Purpose::DeriveKey);
    pack(&a, &b, k.dtype())
}
/// Deterministically derive a scalar key from a scalar U32 tag (which may be traced).
pub fn fold_in(k: &Array, tag: &Array) -> Array {
    assert!(
        tag.dtype() == DType::U32 && tag.shape().dims().is_empty(),
        "fold_in requires a scalar U32 tag"
    );
    fold(k, tag.astype(DType::I64))
}
pub fn bits(k: &Array, shape: impl IntoShape, dtype: DType) -> Array {
    try_bits(k, shape, dtype).expect("random::bits")
}
/// Checked raw bit generation; the result dtype is currently restricted to U32.
pub fn try_bits(k: &Array, shape: impl IntoShape, dtype: DType) -> Result<Array, LoweringError> {
    validate(k)?;
    if dtype != DType::U32 {
        return Err(gnx_base::ops::signature("bits requires U32"));
    }
    let [a, b] = block(k, &shape.into_shape(), Purpose::RandomBits)?;
    Ok(xor(&a, &b).astype(dtype))
}
fn precision(dtype: DType) -> Result<(u32, DType), LoweringError> {
    Ok(match dtype {
        DType::BF16 => (8, DType::F32),
        DType::F16 => (11, DType::F32),
        DType::F32 => (24, DType::F32),
        DType::F64 => (53, DType::F64),
        _ => {
            return Err(gnx_base::ops::signature(
                "random samples require BF16, F16, F32, or F64",
            ));
        }
    })
}
fn fraction(k: &Array, shape: &Shape, p: u32, work: DType) -> Result<Array, LoweringError> {
    let [a, b] = block(k, shape, Purpose::RandomBits)?;
    let integer = if p > 32 {
        left(&a, p - 32) + right(&b, 64 - p)
    } else {
        right(&xor(&a, &b), 32 - p)
    };
    Ok(integer.astype(work) * 2f64.powi(-(p as i32)))
}
fn validate(k: &Array) -> Result<(), LoweringError> {
    if !k.dtype().is_key() || !k.shape().dims().is_empty() {
        return Err(gnx_base::ops::signature("expected a scalar RNG key"));
    }
    Ok(())
}
pub fn try_uniform(k: &Array, shape: impl IntoShape, dtype: DType) -> Result<Array, LoweringError> {
    let shape = shape.into_shape();
    validate(k)?;
    let (p, work) = precision(dtype)?;
    Ok(fraction(k, &shape, p, work)?.astype(dtype))
}
/// Uniform samples in [0, 1), including after conversion to the output dtype.
pub fn uniform(k: &Array, shape: impl IntoShape, dtype: DType) -> Array {
    try_uniform(k, shape, dtype).expect("random::uniform")
}
pub fn try_normal(k: &Array, shape: impl IntoShape, dtype: DType) -> Result<Array, LoweringError> {
    let shape = shape.into_shape();
    validate(k)?;
    let (_, work) = precision(dtype)?;
    let (p, _) = precision(work)?;
    let (rad, angle) = split2(k);
    let u = fraction(&rad, &shape, p, work)?;
    let v = fraction(&angle, &shape, p, work)?;
    let r2 = lax::log1p(&lax::neg(&u)) * -2.0;
    let r = op(ops::elementwise::sqrt(), &r2, &[], work);
    Ok((r * lax::cos(&(v * std::f64::consts::TAU))).astype(dtype))
}
/// Standard normal draws using Box–Muller, computed in F64 for F64 output.
pub fn normal(k: &Array, shape: impl IntoShape, dtype: DType) -> Array {
    try_normal(k, shape, dtype).expect("random::normal")
}

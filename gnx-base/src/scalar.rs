//! Rust scalar types and exact, archivable constant payloads.
use crate::{
    Array, ArrayInfo, ArrayRef, DType, invoke_array,
    ops::{self, signature},
};
use gnx_expr::LoweringError;
use half::{bf16, f16};
type Result<T> = std::result::Result<T, LoweringError>;

mod sealed {
    pub trait Sealed {}
}
/// The GNX element type corresponding to a Rust scalar type.
pub trait AsDType: sealed::Sealed {
    const DTYPE: DType;
}
/// Rust values that can be embedded in a traced scalar constant without loss.
pub trait IntoScalar: AsDType + Copy {
    fn into_scalar(self) -> Scalar;
}

/// Half values retain their bit patterns. Integers never pass through a float when stored.
#[derive(Clone, Copy, Debug, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub enum Scalar {
    Bool(bool),
    Int(i64),
    UInt(u64),
    F16(u16),
    BF16(u16),
    F32(f32),
    F64(f64),
}
macro_rules! scalar_types {
    ($($ty:ty => $dtype:ident: $variant:ident($convert:expr)),+ $(,)?) => {$(
        impl sealed::Sealed for $ty {}
        impl AsDType for $ty { const DTYPE: DType = DType::$dtype; }
        impl IntoScalar for $ty {
            fn into_scalar(self) -> Scalar { Scalar::$variant(($convert)(self)) }
        }
    )+};
}
scalar_types! {
    bool => Bool: Bool(std::convert::identity),
    i8 => I8: Int(i64::from),
    i16 => I16: Int(i64::from),
    i32 => I32: Int(i64::from),
    i64 => I64: Int(i64::from),
    u8 => U8: UInt(u64::from),
    u16 => U16: UInt(u64::from),
    u32 => U32: UInt(u64::from),
    u64 => U64: UInt(u64::from),
    f16 => F16: F16(f16::to_bits),
    bf16 => BF16: BF16(bf16::to_bits),
    f32 => F32: F32(f32::from),
    f64 => F64: F64(f64::from),
}

#[cfg(target_pointer_width = "64")]
scalar_types! {isize=>I64:Int(|v:isize|v as i64), usize=>U64:UInt(|v:usize|v as u64)}
#[cfg(target_pointer_width = "32")]
scalar_types! {isize=>I32:Int(|v:isize|v as i64), usize=>U32:UInt(|v:usize|v as u64)}
#[cfg(target_pointer_width = "16")]
scalar_types! {isize=>I16:Int(|v:isize|v as i64), usize=>U16:UInt(|v:usize|v as u64)}

impl Scalar {
    fn integer(self) -> Option<i128> {
        match self {
            Self::Int(v) => Some(v.into()),
            Self::UInt(v) => Some(v.into()),
            _ => None,
        }
    }
    fn real(self) -> Option<f64> {
        match self {
            Self::F16(b) => Some(f16::from_bits(b).to_f64()),
            Self::BF16(b) => Some(bf16::from_bits(b).to_f64()),
            Self::F32(v) => Some(v.into()),
            Self::F64(v) => Some(v),
            _ => None,
        }
    }
    /// Convert to the requested result dtype without changing the numerical value.
    pub fn exact(self, dtype: DType) -> Result<Self> {
        self.convert(dtype, true)
    }
    /// Floating-point arithmetic literals round to the array precision, but must not overflow.
    pub(crate) fn weak(self, dtype: DType) -> Result<Self> {
        if self.real().is_some()
            && matches!(
                dtype,
                DType::Bool
                    | DType::I8
                    | DType::I16
                    | DType::I32
                    | DType::I64
                    | DType::U8
                    | DType::U16
                    | DType::U32
                    | DType::U64
            )
        {
            return Err(signature("floating literal requires floating array"));
        }
        self.convert(dtype, false)
    }
    fn convert(self, dtype: DType, exact: bool) -> Result<Self> {
        use DType::*;
        let invalid = || signature("scalar is not representable in the requested dtype");
        if matches!(
            (self, dtype),
            (Self::F16(_), F16)
                | (Self::BF16(_), BF16)
                | (Self::F32(_), F32 | C64)
                | (Self::F64(_), F64 | C128)
                | (Self::Bool(_), Bool)
        ) {
            return Ok(self);
        }
        if matches!(dtype, I8 | I16 | I32 | I64 | U8 | U16 | U32 | U64 | Bool) {
            let n = if let Some(v) = self.integer() {
                v
            } else {
                let f = self.real().ok_or_else(invalid)?;
                if !f.is_finite() || f.fract() != 0. {
                    return Err(invalid());
                }
                f as i128
            };
            macro_rules! narrow {
                ($t:ty) => {
                    <$t>::try_from(n)
                        .map(IntoScalar::into_scalar)
                        .map_err(|_| invalid())
                };
            }
            return match dtype {
                I8 => narrow!(i8),
                I16 => narrow!(i16),
                I32 => narrow!(i32),
                I64 => narrow!(i64),
                U8 => narrow!(u8),
                U16 => narrow!(u16),
                U32 => narrow!(u32),
                U64 => narrow!(u64),
                Bool if exact && (0..=1).contains(&n) => Ok(Self::Bool(n != 0)),
                _ => Err(invalid()),
            };
        }
        let integer = self.integer();
        let v = match integer {
            Some(n) => n as f64,
            None => self.real().ok_or_else(invalid)?,
        };
        let result = match dtype {
            F16 => f16::from_f64(v).into_scalar(),
            BF16 => bf16::from_f64(v).into_scalar(),
            F32 | C64 => (v as f32).into_scalar(),
            F64 | C128 => Self::F64(v),
            F8E4M3FN | F8E5M2 | F8E8M0FNU => {
                if !fits_fp8(v, dtype, exact || integer.is_some()) {
                    return Err(invalid());
                }
                Self::F64(v)
            }
            _ => return Err(invalid()),
        };
        let rounded = result.real().unwrap();
        if (v.is_finite() && !rounded.is_finite())
            || (exact && !v.is_nan() && rounded != v)
            || integer.is_some_and(|n| rounded as i128 != n)
        {
            return Err(invalid());
        }
        Ok(result)
    }
}
// FP8 values use XLA's native conversion; check the range and quantization grid first.
fn fits_fp8(v: f64, t: DType, exact: bool) -> bool {
    if v.is_nan() {
        return true;
    }
    let x = v.abs();
    if t == DType::F8E8M0FNU {
        return v > 0.
            && v.is_finite()
            && v >= 2f64.powi(-127)
            && v <= 2f64.powi(127)
            && (!exact || v.to_bits() & ((1u64 << 52) - 1) == 0);
    }
    let (max, emin, m) = if t == DType::F8E4M3FN {
        (448., -6, 3)
    } else {
        (57344., -14, 2)
    };
    if x.is_infinite() {
        return t == DType::F8E5M2;
    }
    if x > max {
        return false;
    }
    if !exact || x == 0. {
        return true;
    }
    let exp = ((x.to_bits() >> 52) as i32 - 1023).max(emin);
    (x / 2f64.powi(exp - m)).fract() == 0.
}

impl Array {
    /// A rank-zero constant with its Rust type's dtype (`0.5` is F64, `0.5_f32` is F32).
    pub fn scalar<T: IntoScalar>(value: T) -> Self {
        Self::from_scalar(value.into_scalar(), T::DTYPE)
    }
    pub(crate) fn from_scalar(value: Scalar, dtype: DType) -> Self {
        invoke_array(
            ops::creation::scalar_op(value),
            &[],
            &ArrayInfo::new((), dtype),
        )
    }
    /// Create an exactly representable constant with an explicit result dtype.
    pub fn try_scalar_as<T: IntoScalar>(value: T, dtype: DType) -> Result<Self> {
        Ok(Self::from_scalar(value.into_scalar().exact(dtype)?, dtype))
    }
    pub fn try_astype(&self, dtype: DType) -> Result<Self> {
        if dtype == self.dtype() {
            return Ok(self.clone());
        }
        if dtype.is_key() || self.dtype().is_key() {
            return Err(signature("use random::key_data or wrap_key_data for keys"));
        }
        Ok(invoke_array(
            ops::convert_element_type_op(),
            &[self],
            &ArrayInfo::new(self.shape(), dtype),
        ))
    }
    #[track_caller]
    pub fn astype(&self, dtype: DType) -> Self {
        self.try_astype(dtype).expect("Array::astype")
    }
}
impl<T: IntoScalar> From<T> for Array {
    fn from(v: T) -> Self {
        Self::scalar(v)
    }
}
impl ArrayRef {
    pub fn try_astype(&self, dtype: DType) -> Result<Array> {
        self.get().try_astype(dtype)
    }
    #[track_caller]
    pub fn astype(&self, dtype: DType) -> Array {
        self.get().astype(dtype)
    }
}

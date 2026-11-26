use super::*;
use DType::*;
use ops::elementwise as e;

pub(super) fn integer(t: DType) -> bool {
    matches!(t, I8 | I16 | I32 | I64 | U8 | U16 | U32 | U64)
}
fn complex(t: DType) -> bool {
    matches!(t, C64 | C128)
}
fn inexact(t: DType) -> bool {
    !integer(t) && t != Bool
}

macro_rules! unary {
    ($name:ident, $checked:ident, $op:ident, $valid:expr) => {
        primitive! {
            #[doc = concat!("Elementwise `", stringify!($name), "`, preserving shape and dtype.")]
            $name, $checked(x: &Array) -> Array {
                if !($valid)(x.dtype()) { return Err(signature(concat!("unsupported dtype for ", stringify!($name)))); }
                Ok(invoke_array(e::$op(), &[x], &ArrayInfo::new(x.shape(), x.dtype())))
            }
        }
    };
}
unary!(exp, try_exp, exp, inexact);
unary!(log1p, try_log1p, log1p, inexact);
unary!(cos, try_cos, cos, inexact);
unary!(sin, try_sin, sin, inexact);
unary!(rsqrt, try_rsqrt, rsqrt, inexact);
unary!(logistic, try_logistic, logistic, inexact);
unary!(neg, try_neg, neg, |t| t != Bool);
primitive! {
    /// Absolute value; complex inputs produce real magnitudes.
    abs, try_abs(x: &Array) -> Array {
        let dtype = match x.dtype() { Bool => return Err(signature("abs requires numeric input")), C64 => F32, C128 => F64, t => t };
        Ok(invoke_array(e::abs(), &[x], &ArrayInfo::new(x.shape(), dtype)))
    }
}

macro_rules! binary {
    ($name:ident, $checked:ident, $op:expr, $valid:expr, $dtype:expr) => {
        primitive! {
            #[doc = concat!("Elementwise `", stringify!($name), "` on operands of matching shape and dtype.")]
            $name, $checked(x: &Array, y: &Array) -> Array {
                let t = x.dtype();
                if t != y.dtype() || x.shape() != y.shape() { return Err(signature(concat!(stringify!($name), " requires matching shapes and dtypes"))); }
                if !($valid)(t) { return Err(signature(concat!("unsupported dtype for ", stringify!($name)))); }
                Ok(invoke_array($op, &[x,y], &ArrayInfo::new(x.shape(), ($dtype)(t))))
            }
        }
    };
}
binary!(ne, try_ne, e::ne(), |_| true, |_| Bool);
binary!(lt, try_lt, e::lt(), |t| !complex(t), |_| Bool);
binary!(max, try_max, e::max(), |t| !complex(t), |t| t);
binary!(
    bitwise_and,
    try_bitwise_and,
    ops::bitwise::and(),
    |t| integer(t) || t == Bool,
    |t| t
);

/// Convert element type without changing shape.
pub fn convert_element_type(x: &Array, dtype: DType) -> Array {
    if dtype == x.dtype() {
        return x.clone();
    }
    invoke_array(
        ops::dtype::convert_element_type(),
        &[x],
        &ArrayInfo::new(x.shape(), dtype),
    )
}

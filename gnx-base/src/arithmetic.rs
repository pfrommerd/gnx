//! Lazy elementwise arithmetic on owned or borrowed arrays.
use crate::{Array, ArrayInfo, DType, Dim, Shape, ops};
use gnx_expr::{Invocation, LoweringError, Op, VarInfo};
use std::ops::{Add, Div, Mul, Rem, Sub};

pub fn apply(op: Op, inputs: &[&Array], info: &ArrayInfo) -> Array {
    Invocation::invoke(
        op,
        vec![],
        inputs.iter().map(|x| x.tracer().clone()).collect(),
        vec![VarInfo::new(info.clone())],
    )
    .remove(0)
    .try_cast::<Array>()
    .expect("array operation produces an array")
    .into()
}

fn binary(lhs: &Array, rhs: &Array, op: Op) -> Result<Array, LoweringError> {
    let dtype = lhs.dtype();
    if dtype.is_key() {
        return Err(ops::signature("keys do not support arithmetic"));
    }
    if dtype != rhs.dtype() {
        return Err(ops::signature("array arithmetic requires matching dtypes"));
    }
    match dtype {
        DType::Bool if !(op == ops::add_op() || op == ops::mul_op()) => {
            return Err(ops::signature(
                "boolean arrays only support addition and multiplication",
            ));
        }
        DType::C64 | DType::C128 if op == ops::rem_op() => {
            return Err(ops::signature("complex arrays do not support remainder"));
        }
        _ => (),
    }
    if lhs.shape() == rhs.shape() {
        return Ok(apply(
            op,
            &[lhs, rhs],
            &ArrayInfo::new(lhs.shape().clone(), dtype),
        ));
    }
    let (lhs_shape, rhs_shape) = (lhs.shape(), rhs.shape());
    let (a, b) = (lhs_shape.dims(), rhs_shape.dims());
    let rank = a.len().max(b.len());
    let mut dims = vec![1; rank];
    for operand in [a, b] {
        for (out, axis) in dims.iter_mut().rev().zip(operand.iter().rev()) {
            let Dim::Fixed(n) = *axis else {
                return Err(ops::signature(
                    "broadcasting different shapes requires fixed dimensions",
                ));
            };
            if *out == 1 {
                *out = n;
            } else if n != 1 && n != *out {
                return Err(ops::signature(
                    "incompatible shapes for elementwise broadcasting",
                ));
            }
        }
    }
    let info = ArrayInfo::new(Shape::fixed(dims), dtype);
    let expand = |x: &Array| {
        if &x.shape() == info.shape() {
            return x.clone();
        }
        let axes = (rank - x.shape().dims().len()..rank).collect::<Vec<_>>();
        apply(ops::broadcast_in_dim_op(&axes), &[x], &info)
    };
    Ok(apply(op, &[&expand(lhs), &expand(rhs)], &info))
}

macro_rules! arithmetic {
    ($trait:ident, $method:ident, $try_method:ident, $op:ident) => {
        impl Array {
            #[doc = concat!("Build a lazy `",stringify!($method),"` expression. Operands must have matching dtypes; fixed shapes broadcast from the right.")]
            pub fn $try_method(&self, rhs: &Array) -> Result<Array, LoweringError> {
                binary(self, rhs, ops::$op())
            }
        }
        impl $trait<&Array> for &Array {
            type Output = Array;
            #[track_caller]
            fn $method(self, rhs: &Array) -> Array {
                self.$try_method(rhs).unwrap_or_else(|err| panic!("{}({}, {}): {err}", stringify!($method), self.tracer().info(), rhs.tracer().info()))
            }
        }
        impl $trait<Array> for &Array {
            type Output = Array;
            #[track_caller]
            fn $method(self, rhs: Array) -> Array { self.$method(&rhs) }
        }
        impl $trait<&Array> for Array {
            type Output = Array;
            #[track_caller]
            fn $method(self, rhs: &Array) -> Array { (&self).$method(rhs) }
        }
        impl $trait<Array> for Array {
            type Output = Array;
            #[track_caller]
            fn $method(self, rhs: Array) -> Array { (&self).$method(&rhs) }
        }
    }
}
arithmetic!(Add, add, try_add, add_op);
arithmetic!(Sub, sub, try_sub, sub_op);
arithmetic!(Mul, mul, try_mul, mul_op);
arithmetic!(Div, div, try_div, div_op);
arithmetic!(Rem, rem, try_rem, rem_op);

#[track_caller]
fn literal<T: crate::IntoScalar>(x: &Array, v: T) -> Array {
    let dtype = x.dtype();
    let value = v
        .into_scalar()
        .weak(dtype)
        .expect("invalid scalar arithmetic operand");
    // The three FP8 formats have no native Rust scalar type. Let the conversion
    // primitive perform their final rounding instead of storing a non-exact literal.
    if matches!(dtype, DType::F8E4M3FN | DType::F8E5M2 | DType::F8E8M0FNU) {
        Array::from_scalar(value, DType::F64).astype(dtype)
    } else {
        Array::from_scalar(value, dtype)
    }
}
macro_rules! scalar_arithmetic {
    ($trait:ident,$method:ident; $($ty:ty),+) => {
        impl<T:crate::IntoScalar> $trait<T> for &Array {
            type Output=Array;
            #[track_caller] fn $method(self,v:T)->Array {self.$method(&literal(self,v))}
        }
        impl<T:crate::IntoScalar> $trait<T> for Array {
            type Output=Array;
            #[track_caller] fn $method(self,v:T)->Array {(&self).$method(v)}
        }
        $(impl $trait<&Array> for $ty {
            type Output=Array;
            #[track_caller] fn $method(self,x:&Array)->Array {literal(x,self).$method(x)}
        }
        impl $trait<Array> for $ty {
            type Output=Array;
            #[track_caller] fn $method(self,x:Array)->Array {self.$method(&x)}
        })+
    };
}
macro_rules! scalar_operator {
    ($trait:ident, $method:ident) => {
        scalar_arithmetic!($trait, $method; bool, i8, i16, i32, i64, isize,
            u8, u16, u32, u64, usize, half::f16, half::bf16, f32, f64);
    };
}
scalar_operator!(Add, add);
scalar_operator!(Sub, sub);
scalar_operator!(Mul, mul);
scalar_operator!(Div, div);

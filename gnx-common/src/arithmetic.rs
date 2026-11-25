//! Lazy elementwise arithmetic on owned or borrowed arrays.
use crate::{Array, ArrayInfo, DType, Dim, Shape, ops};
use gnx_expr::{Invocation, LoweringError, Op, VarInfo};
use std::ops::{Add, Div, Mul, Rem, Sub};

fn apply(op: Op, inputs: &[&Array], info: &ArrayInfo) -> Array {
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

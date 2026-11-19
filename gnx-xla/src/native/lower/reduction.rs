use std::io;

use gnx_expr::{Eqn, Expr};
use gnx_math::ops::reduction::{ReduceOperation, ReduceWindowOperation};
use gnx_math_base::ops::reduction::ReduceAddOperation;
use xla::{ElementType, XlaBuilder, XlaComputation, XlaOp};

use super::{
    LoweringContext, XlaLowering, indices, invalid, is_pred, items, lower_body, metadata, native,
    shape, unary,
};

fn computation(reducer: &Expr) -> io::Result<XlaComputation> {
    let builder = XlaBuilder::new("reducer");
    let inputs = reducer
        .inputs()
        .iter()
        .enumerate()
        .map(|(i, (_, info))| {
            native(builder.parameter_s(i as i64, &shape(metadata(info)?)?, "scalar"))
        })
        .collect::<io::Result<Vec<_>>>()?;
    let results = lower_body(&builder, reducer, &[], &inputs)?;
    match results.as_slice() {
        [result] => native(result.build()),
        _ => native(native(builder.tuple(&results))?.build()),
    }
}

fn reduce(
    eqn: &Eqn,
    inputs: &[XlaOp],
    build: impl FnOnce(&[XlaOp], &[XlaOp], &XlaComputation) -> io::Result<XlaOp>,
) -> io::Result<Vec<XlaOp>> {
    let count = eqn.outputs().len();
    let [reducer] = eqn.subexprs() else {
        return Err(invalid("reduction needs one reducer region"));
    };
    if count == 0 || inputs.len() != count * 2 {
        return Err(invalid("invalid reduction arity"));
    }
    let (operands, initial) = inputs.split_at(count);
    let first_shape = native(operands[0].array_shape())?;
    for (operand, init) in operands.iter().zip(initial) {
        let operand = native(operand.array_shape())?;
        let init = native(init.array_shape())?;
        if operand.dims() != first_shape.dims()
            || !init.dims().is_empty()
            || init.ty() != operand.ty()
        {
            return Err(invalid("reduction operand or initial value metadata"));
        }
    }
    let result = build(operands, initial, &computation(reducer.expr())?)?;
    if count == 1 {
        Ok(vec![result])
    } else {
        items(&result, count)
    }
}

impl XlaLowering for ReduceOperation {
    fn lower_xla(
        &self,
        _: &LoweringContext<'_>,
        eqn: &Eqn,
        inputs: &[XlaOp],
    ) -> io::Result<Vec<XlaOp>> {
        reduce(eqn, inputs, |operands, initial, reducer| {
            native(XlaOp::reduce_variadic(
                operands,
                initial,
                reducer,
                &indices(self.axes())?,
            ))
        })
    }
}

impl XlaLowering for ReduceAddOperation {
    fn lower_xla(
        &self,
        ctx: &LoweringContext<'_>,
        _: &Eqn,
        inputs: &[XlaOp],
    ) -> io::Result<Vec<XlaOp>> {
        let input = unary("reduce_add", inputs)?;
        let axes = indices(self.axes())?;
        if !is_pred(input)? {
            return Ok(vec![native(input.reduce_sum(&axes, false))?]);
        }
        let reducer = XlaBuilder::new("reduce_or");
        let lhs = native(reducer.parameter(0, ElementType::Pred, &[], "lhs"))?;
        let rhs = native(reducer.parameter(1, ElementType::Pred, &[], "rhs"))?;
        let combine = native(native(lhs.or(&rhs))?.build())?;
        let zero = native(ctx.builder().zero(ElementType::Pred))?;
        Ok(vec![native(input.reduce(zero, combine, &axes, false))?])
    }
}

impl XlaLowering for ReduceWindowOperation {
    fn lower_xla(
        &self,
        _: &LoweringContext<'_>,
        eqn: &Eqn,
        inputs: &[XlaOp],
    ) -> io::Result<Vec<XlaOp>> {
        let window = self.window();
        reduce(eqn, inputs, |operands, initial, reducer| {
            native(XlaOp::reduce_window(
                operands,
                initial,
                reducer,
                &indices(&window.dimensions)?,
                &indices(&window.strides)?,
                &indices(&window.base_dilation)?,
                &indices(&window.window_dilation)?,
                &window.padding,
            ))
        })
    }
}

crate::register_xla_lowering!(ReduceOperation, ReduceAddOperation, ReduceWindowOperation);

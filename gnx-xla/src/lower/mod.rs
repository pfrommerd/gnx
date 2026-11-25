mod contraction;
mod elementwise;
mod indexing;
mod linalg;
mod ordering;
mod reduction;
mod structural;

use std::collections::HashMap;
use std::io;

use gnx_expr::{Eqn, Expr, Var, VarInfoRef};
use gnx_math::ops::{
    AddOperation, BroadcastInDimOperation, EyeOperation, MulOperation, OnesOperation,
    ReshapeOperation, RevOperation, SelectOperation, TransposeOperation, ZerosOperation,
};
use gnx_math::{ArrayInfo, ArrayInfoData, DType, Dim};
use gnx_transforms::ad::StopGradientOperation;
use gnx_transforms::cf::{
    CallOperation, CondOperation, ScanOperation, SwitchOperation, WhileOperation,
};
use xla::{ElementType, Shape, XlaBuilder, XlaComputation, XlaOp};

use super::{invalid, native, unsupported};

/// XLA lowering for an operation: an op interface, registered with
/// [`register_xla_lowering!`](crate::register_xla_lowering).
pub trait XlaLowering {
    fn lower_xla(
        &self,
        ctx: &LoweringContext<'_>,
        eqn: Eqn<'_>,
        operands: &[XlaOp],
    ) -> io::Result<Vec<XlaOp>>;
}

/// Register `XlaLowering` implementations for operation types.
#[macro_export]
macro_rules! register_xla_lowering {
    ($($op:ty),+ $(,)?) => {
        $crate::__gnx_expr::register_op_interface!(dyn $crate::XlaLowering: $($op),+);
    };
}

struct Child<'a> {
    expr: Expr<'a>,
    captures: Vec<XlaOp>,
}

pub struct LoweringContext<'a> {
    builder: &'a XlaBuilder,
    children: Vec<Child<'a>>,
}

impl LoweringContext<'_> {
    pub fn builder(&self) -> &XlaBuilder {
        self.builder
    }

    /// Inline a child region with its existing parent captures.
    pub fn lower_region(&self, index: usize, operands: &[XlaOp]) -> io::Result<Vec<XlaOp>> {
        let child = self
            .children
            .get(index)
            .ok_or_else(|| invalid("missing XLA region"))?;
        lower_body(self.builder, child.expr, &child.captures, operands)
    }
}

pub(super) fn metadata(info: VarInfoRef<'_>) -> io::Result<ArrayInfo> {
    info.downcast::<ArrayInfoData>()
        .map(ArrayInfoData::to_native)
        .ok_or_else(|| unsupported("XLA only supports array values"))
}

pub(super) fn dimensions(info: &ArrayInfo) -> io::Result<Vec<i64>> {
    info.shape()
        .dims()
        .iter()
        .map(|dim| match dim {
            Dim::Fixed(n) => i64::try_from(*n).map_err(|_| invalid("XLA dimension overflow")),
            _ => Err(unsupported("XLA requires fixed dimensions")),
        })
        .collect()
}

pub(super) fn element(dtype: DType) -> ElementType {
    match dtype {
        DType::Bool => ElementType::Pred,
        DType::U8 => ElementType::U8,
        DType::U16 => ElementType::U16,
        DType::U32 => ElementType::U32,
        DType::U64 => ElementType::U64,
        DType::I8 => ElementType::S8,
        DType::I16 => ElementType::S16,
        DType::I32 => ElementType::S32,
        DType::I64 => ElementType::S64,
        DType::BF16 => ElementType::Bf16,
        DType::F8E4M3FN => ElementType::F8E4M3FN,
        DType::F8E5M2 => ElementType::F8E5M2,
        DType::F8E8M0FNU => ElementType::F8E8M0FNU,
        DType::F16 => ElementType::F16,
        DType::F32 => ElementType::F32,
        DType::F64 => ElementType::F64,
        DType::C64 => ElementType::C64,
        DType::C128 => ElementType::C128,
    }
}

pub(super) fn shape(info: &ArrayInfo) -> io::Result<Shape> {
    Ok(Shape::array_with_type(
        element(info.dtype()),
        dimensions(&info)?,
    ))
}

pub(super) fn lower(expr: Expr<'_>) -> io::Result<XlaComputation> {
    let builder = XlaBuilder::new("gnx");
    let parameters = expr
        .closure_inputs()
        .chain(expr.inputs())
        .enumerate()
        .map(|(i, param)| {
            let info = metadata(param.info_ref())?;
            native(builder.parameter_s(i as i64, &shape(&info)?, &format!("arg{i}")))
        })
        .collect::<io::Result<Vec<_>>>()?;
    let (captures, inputs) = parameters.split_at(expr.closure_inputs().len());
    let results = lower_body(&builder, expr, captures, inputs)?;
    native(native(builder.tuple(&results))?.build())
}

fn lower_body(
    builder: &XlaBuilder,
    expr: Expr<'_>,
    captures: &[XlaOp],
    inputs: &[XlaOp],
) -> io::Result<Vec<XlaOp>> {
    if captures.len() != expr.closure_inputs().len() || inputs.len() != expr.inputs().len() {
        return Err(invalid("XLA region argument arity mismatch"));
    }
    let mut values: HashMap<Var, XlaOp> = HashMap::new();
    for (param, value) in expr
        .closure_inputs()
        .chain(expr.inputs())
        .zip(captures.iter().chain(inputs))
    {
        check_shape(value, &metadata(param.info_ref())?)?;
        values.insert(param.var(), value.clone());
    }
    for (index, eqn) in expr.eqns().enumerate() {
        let op = eqn.op();
        let results = (|| {
            let rule = op
                .interface::<dyn XlaLowering>()
                .map_err(|error| unsupported(error.to_string()))?;
            let operands = eqn
                .inputs()
                .iter()
                .map(|v| values[v].clone())
                .collect::<Vec<_>>();
            let children = eqn
                .subexprs()
                .map(|sub| Child {
                    expr: sub.expr(),
                    captures: (sub.provided_vars().iter())
                        .map(|v| values[v].clone())
                        .collect(),
                })
                .collect();
            let results = rule.lower_xla(&LoweringContext { builder, children }, eqn, &operands)?;
            if results.len() != eqn.outputs().len() {
                return Err(invalid("XLA lowering result arity mismatch"));
            }
            for (param, value) in eqn.outputs().zip(&results) {
                check_shape(value, &metadata(param.info_ref())?)?;
            }
            Ok(results)
        })()
        .map_err(|err| {
            io::Error::new(
                err.kind(),
                format!("equation {index} ({}.{}) : {err}", op.dialect(), op.name()),
            )
        })?;
        for (param, value) in eqn.outputs().zip(results) {
            values.insert(param.var(), value);
        }
    }
    Ok(expr
        .outputs()
        .iter()
        .map(|var| values[var].clone())
        .collect())
}

fn check_shape(value: &XlaOp, info: &ArrayInfo) -> io::Result<()> {
    if native(value.shape())? != shape(info)? {
        return Err(invalid("XLA lowering changed array shape or dtype"));
    }
    Ok(())
}

fn is_pred(op: &XlaOp) -> io::Result<bool> {
    Ok(native(op.ty())? == xla::PrimitiveType::Pred)
}

fn result_info(eqn: Eqn<'_>) -> io::Result<ArrayInfo> {
    let mut outputs = eqn.outputs();
    match (outputs.next(), outputs.next()) {
        (Some(output), None) => metadata(output.info_ref()),
        _ => Err(invalid("array operations have one result")),
    }
}

fn indices(values: &[usize]) -> io::Result<Vec<i64>> {
    values
        .iter()
        .copied()
        .map(|index| i64::try_from(index).map_err(|_| invalid("index overflow")))
        .collect()
}

fn unary<'a>(name: &str, inputs: &'a [XlaOp]) -> io::Result<&'a XlaOp> {
    match inputs {
        [input] => Ok(input),
        _ => Err(invalid(format!("array.{name} takes one operand"))),
    }
}

impl XlaLowering for StopGradientOperation {
    fn lower_xla(
        &self,
        _: &LoweringContext<'_>,
        _: Eqn<'_>,
        inputs: &[XlaOp],
    ) -> io::Result<Vec<XlaOp>> {
        Ok(vec![unary("stop_gradient", inputs)?.clone()])
    }
}

crate::register_xla_lowering!(StopGradientOperation);

impl XlaLowering for AddOperation {
    fn lower_xla(
        &self,
        _: &LoweringContext<'_>,
        _: Eqn<'_>,
        inputs: &[XlaOp],
    ) -> io::Result<Vec<XlaOp>> {
        let [a, b] = inputs else {
            return Err(invalid("array.add requires two inputs"));
        };
        Ok(vec![if is_pred(a)? {
            native(a.or(b))?
        } else {
            native(a.add_(b))?
        }])
    }
}

impl XlaLowering for MulOperation {
    fn lower_xla(
        &self,
        _: &LoweringContext<'_>,
        _: Eqn<'_>,
        inputs: &[XlaOp],
    ) -> io::Result<Vec<XlaOp>> {
        let [a, b] = inputs else {
            return Err(invalid("array.mul requires two inputs"));
        };
        Ok(vec![if is_pred(a)? {
            native(a.and(b))?
        } else {
            native(a.mul_(b))?
        }])
    }
}

impl XlaLowering for SelectOperation {
    fn lower_xla(
        &self,
        _: &LoweringContext<'_>,
        _: Eqn<'_>,
        inputs: &[XlaOp],
    ) -> io::Result<Vec<XlaOp>> {
        let [predicate, on_true, on_false] = inputs else {
            return Err(invalid("array.select requires three inputs"));
        };
        Ok(vec![native(predicate.select(on_true, on_false))?])
    }
}

impl XlaLowering for ZerosOperation {
    fn lower_xla(
        &self,
        ctx: &LoweringContext<'_>,
        eqn: Eqn<'_>,
        _: &[XlaOp],
    ) -> io::Result<Vec<XlaOp>> {
        let info = result_info(eqn)?;
        let zero = native(ctx.builder.zero(element(info.dtype())))?;
        Ok(vec![native(zero.broadcast(&dimensions(&info)?))?])
    }
}

impl XlaLowering for OnesOperation {
    fn lower_xla(
        &self,
        ctx: &LoweringContext<'_>,
        eqn: Eqn<'_>,
        _: &[XlaOp],
    ) -> io::Result<Vec<XlaOp>> {
        let info = result_info(eqn)?;
        let one = native(ctx.builder.one(element(info.dtype())))?;
        Ok(vec![native(one.broadcast(&dimensions(&info)?))?])
    }
}

impl XlaLowering for EyeOperation {
    fn lower_xla(
        &self,
        ctx: &LoweringContext<'_>,
        eqn: Eqn<'_>,
        _: &[XlaOp],
    ) -> io::Result<Vec<XlaOp>> {
        let info = result_info(eqn)?;
        let dims = dimensions(&info)?;
        let ty = element(info.dtype());
        let rows = native(ctx.builder.iota(ElementType::S64, &dims, 0))?;
        let columns = native(ctx.builder.iota(ElementType::S64, &dims, 1))?;
        let one = native(native(ctx.builder.one(ty))?.broadcast(&dims))?;
        let zero = native(native(ctx.builder.zero(ty))?.broadcast(&dims))?;
        Ok(vec![native(
            native(rows.eq(&columns))?.select(&one, &zero),
        )?])
    }
}

impl XlaLowering for ReshapeOperation {
    fn lower_xla(
        &self,
        _: &LoweringContext<'_>,
        eqn: Eqn<'_>,
        inputs: &[XlaOp],
    ) -> io::Result<Vec<XlaOp>> {
        let input = unary("reshape", inputs)?;
        Ok(vec![native(
            input.reshape(&dimensions(&result_info(eqn)?)?),
        )?])
    }
}

impl XlaLowering for BroadcastInDimOperation {
    fn lower_xla(
        &self,
        _: &LoweringContext<'_>,
        eqn: Eqn<'_>,
        inputs: &[XlaOp],
    ) -> io::Result<Vec<XlaOp>> {
        let input = unary("broadcast_in_dim", inputs)?;
        let shape = dimensions(&result_info(eqn)?)?;
        Ok(vec![native(
            input.broadcast_in_dim(&shape, &indices(&self.dims())?),
        )?])
    }
}

impl XlaLowering for TransposeOperation {
    fn lower_xla(
        &self,
        _: &LoweringContext<'_>,
        _: Eqn<'_>,
        inputs: &[XlaOp],
    ) -> io::Result<Vec<XlaOp>> {
        let input = unary("transpose", inputs)?;
        Ok(vec![native(input.transpose(&indices(&self.perm())?))?])
    }
}

impl XlaLowering for RevOperation {
    fn lower_xla(
        &self,
        _: &LoweringContext<'_>,
        _: Eqn<'_>,
        inputs: &[XlaOp],
    ) -> io::Result<Vec<XlaOp>> {
        let input = unary("rev", inputs)?;
        Ok(vec![native(input.rev(&indices(&self.axes())?))?])
    }
}

crate::register_xla_lowering!(
    AddOperation,
    MulOperation,
    SelectOperation,
    ZerosOperation,
    OnesOperation,
    EyeOperation,
    ReshapeOperation,
    BroadcastInDimOperation,
    TransposeOperation,
    RevOperation,
);

fn local_sharding(eqn: Eqn<'_>, inputs: &[XlaOp]) -> io::Result<Vec<XlaOp>> {
    let mut outputs = eqn.outputs();
    let (Some(output), None) = (outputs.next(), outputs.next()) else {
        return Err(invalid("sharding annotation result arity"));
    };
    let rank = metadata(output.info_ref())?.shape().dims().len();
    let spec = gnx_math::shard::Sharding::from_op_ref(eqn.op(), rank).map_err(io::Error::other)?;
    if !spec.is_replicated() {
        return Err(unsupported(
            "named sharding axes are unavailable on the local XLA device",
        ));
    }
    Ok(inputs.to_vec())
}

impl XlaLowering for gnx_math::shard::Constraint {
    fn lower_xla(
        &self,
        _: &LoweringContext<'_>,
        eqn: Eqn<'_>,
        inputs: &[XlaOp],
    ) -> io::Result<Vec<XlaOp>> {
        local_sharding(eqn, inputs)
    }
}
impl XlaLowering for gnx_math::shard::Replicate {
    fn lower_xla(
        &self,
        _: &LoweringContext<'_>,
        eqn: Eqn<'_>,
        inputs: &[XlaOp],
    ) -> io::Result<Vec<XlaOp>> {
        local_sharding(eqn, inputs)
    }
}
crate::register_xla_lowering!(gnx_math::shard::Constraint);
crate::register_xla_lowering!(gnx_math::shard::Replicate);

impl XlaLowering for CallOperation {
    fn lower_xla(
        &self,
        ctx: &LoweringContext<'_>,
        _eqn: Eqn<'_>,
        inputs: &[XlaOp],
    ) -> io::Result<Vec<XlaOp>> {
        ctx.lower_region(0, inputs)
    }
}
crate::register_xla_lowering!(CallOperation);

fn items(tuple: &XlaOp, count: usize) -> io::Result<Vec<XlaOp>> {
    (0..count)
        .map(|i| native(tuple.get_tuple_element(i as i64)))
        .collect()
}

fn tuple_shape(values: &[XlaOp]) -> io::Result<Shape> {
    Ok(Shape::tuple(
        values
            .iter()
            .map(|op| native(op.shape()))
            .collect::<io::Result<_>>()?,
    ))
}

fn branch(
    child: &Child<'_>,
    operands: &[XlaOp],
    parent: &XlaBuilder,
) -> io::Result<(XlaOp, XlaComputation)> {
    let builder = XlaBuilder::new("branch");
    let args = child
        .captures
        .iter()
        .chain(operands)
        .cloned()
        .collect::<Vec<_>>();
    let p = native(builder.parameter_s(0, &tuple_shape(&args)?, "region"))?;
    let values = items(&p, args.len())?;
    let (captures, operands) = values.split_at(child.captures.len());
    let outputs = lower_body(&builder, child.expr, captures, operands)?;
    Ok((
        native(parent.tuple(&args))?,
        native(native(builder.tuple(&outputs))?.build())?,
    ))
}

impl XlaLowering for CondOperation {
    fn lower_xla(
        &self,
        ctx: &LoweringContext<'_>,
        eqn: Eqn<'_>,
        inputs: &[XlaOp],
    ) -> io::Result<Vec<XlaOp>> {
        let ([pred, operands @ ..], [when_true, when_false]) = (inputs, ctx.children.as_slice())
        else {
            return Err(invalid("invalid cond signature"));
        };
        let (t, t_body) = branch(when_true, operands, ctx.builder)?;
        let (f, f_body) = branch(when_false, operands, ctx.builder)?;
        items(
            &native(pred.conditional(t, t_body, f, f_body))?,
            eqn.outputs().len(),
        )
    }
}
crate::register_xla_lowering!(CondOperation);

impl XlaLowering for SwitchOperation {
    fn lower_xla(
        &self,
        ctx: &LoweringContext<'_>,
        eqn: Eqn<'_>,
        inputs: &[XlaOp],
    ) -> io::Result<Vec<XlaOp>> {
        fn lower_switch(
            builder: &XlaBuilder,
            selector: &XlaOp,
            operands: &[XlaOp],
            children: &[Child<'_>],
            index: usize,
        ) -> io::Result<XlaOp> {
            let (child, rest) = children
                .split_first()
                .ok_or_else(|| invalid("missing switch branches"))?;
            if rest.is_empty() {
                return native(builder.tuple(&lower_body(
                    builder,
                    child.expr,
                    &child.captures,
                    operands,
                )?));
            }
            let (yes, yes_body) = branch(child, operands, builder)?;
            let args = std::iter::once(selector)
                .chain(operands)
                .chain(rest.iter().flat_map(|child| &child.captures))
                .cloned()
                .collect::<Vec<_>>();
            let next = XlaBuilder::new("switch_else");
            let values = items(
                &native(next.parameter_s(0, &tuple_shape(&args)?, "switch"))?,
                args.len(),
            )?;
            let mut remaining = &values[1 + operands.len()..];
            let rest: Vec<_> = rest
                .iter()
                .map(|child| {
                    let (captures, tail) = remaining.split_at(child.captures.len());
                    remaining = tail;
                    Child {
                        expr: child.expr,
                        captures: captures.to_vec(),
                    }
                })
                .collect();
            let no_body = native(
                lower_switch(
                    &next,
                    &values[0],
                    &values[1..1 + operands.len()],
                    &rest,
                    index + 1,
                )?
                .build(),
            )?;
            let cutoff =
                native(builder.constant_r0(
                    i32::try_from(index).map_err(|_| invalid("switch index overflow"))?,
                ))?;
            // JAX clamps out-of-range selectors to the first or last branch.
            let test = native(selector.le(&cutoff))?;
            native(test.conditional(yes, yes_body, native(builder.tuple(&args))?, no_body))
        }
        let [selector, operands @ ..] = inputs else {
            return Err(invalid("missing switch selector"));
        };
        items(
            &lower_switch(ctx.builder, selector, operands, &ctx.children, 0)?,
            eqn.outputs().len(),
        )
    }
}
crate::register_xla_lowering!(SwitchOperation);

impl XlaLowering for WhileOperation {
    fn lower_xla(
        &self,
        ctx: &LoweringContext<'_>,
        _eqn: Eqn<'_>,
        carry: &[XlaOp],
    ) -> io::Result<Vec<XlaOp>> {
        let [condition, body] = ctx.children.as_slice() else {
            return Err(invalid("missing while regions"));
        };
        let state = carry
            .iter()
            .chain(&condition.captures)
            .chain(&body.captures)
            .cloned()
            .collect::<Vec<_>>();
        let shape = tuple_shape(&state)?;
        let cond_builder = XlaBuilder::new("while_predicate");
        let args = items(
            &native(cond_builder.parameter_s(0, &shape, "state"))?,
            state.len(),
        )?;
        let midpoint = carry.len() + condition.captures.len();
        let predicate = lower_body(
            &cond_builder,
            condition.expr,
            &args[carry.len()..midpoint],
            &args[..carry.len()],
        )?;
        let [predicate] = predicate.as_slice() else {
            return Err(invalid("while predicate arity"));
        };
        // Emit the predicate last even when the region forwards an input.
        let inverted = native(predicate.not())?;
        let condition = native(native(inverted.not())?.build())?;
        let body_builder = XlaBuilder::new("while_body");
        let args = items(
            &native(body_builder.parameter_s(0, &shape, "state"))?,
            state.len(),
        )?;
        let mut outputs = lower_body(
            &body_builder,
            body.expr,
            &args[midpoint..],
            &args[..carry.len()],
        )?;
        outputs.extend_from_slice(&args[carry.len()..]);
        let body = native(native(body_builder.tuple(&outputs))?.build())?;
        items(
            &native(XlaOp::while_(
                condition,
                body,
                native(ctx.builder.tuple(&state))?,
            ))?,
            carry.len(),
        )
    }
}
crate::register_xla_lowering!(WhileOperation);

impl XlaLowering for ScanOperation {
    fn lower_xla(
        &self,
        ctx: &LoweringContext<'_>,
        eqn: Eqn<'_>,
        inputs: &[XlaOp],
    ) -> io::Result<Vec<XlaOp>> {
        let configuration = self.parameters();
        let carry = configuration
            .physical_carry_len()
            .map_err(io::Error::other)?;
        let [child] = ctx.children.as_slice() else {
            return Err(invalid("missing scan body"));
        };
        if eqn.outputs().len() < carry {
            return Err(invalid("invalid scan carry"));
        }
        let ys = eqn
            .outputs()
            .skip(carry)
            .map(|output| {
                let info = metadata(output.info_ref())?;
                native(
                    native(ctx.builder.zero(element(info.dtype())))?.broadcast(&dimensions(&info)?),
                )
            })
            .collect::<io::Result<Vec<_>>>()?;
        if configuration.length == 0 {
            return Ok(inputs[..carry].iter().cloned().chain(ys).collect());
        }
        let length =
            i64::try_from(configuration.length).map_err(|_| invalid("scan length overflow"))?;
        // Loop state: (index, carry..., stacked inputs..., stacked outputs..., captures...).
        let start = native(ctx.builder.constant_r0(0_i64))?;
        let state = std::iter::once(start)
            .chain(inputs.iter().cloned())
            .chain(ys)
            .chain(child.captures.iter().cloned())
            .collect::<Vec<_>>();
        let shape = tuple_shape(&state)?;
        let cond = XlaBuilder::new("scan_predicate");
        let p = native(cond.parameter_s(0, &shape, "state"))?;
        let predicate =
            native(native(p.get_tuple_element(0))?.lt(&native(cond.constant_r0(length))?))?;
        let cond = native(predicate.build())?;
        let builder = XlaBuilder::new("scan_body");
        let p = native(builder.parameter_s(0, &shape, "state"))?;
        let args = items(&p, state.len())?;
        let index = &args[0];
        let zero = native(builder.constant_r0(0_i64))?;
        let mut step_inputs = args[1..1 + carry].to_vec();
        for input in &args[1 + carry..1 + inputs.len()] {
            let shape = native(input.array_shape())?;
            let dims = shape.dims();
            let mut slice = dims.to_vec();
            slice[0] = 1;
            let offsets: Vec<_> = std::iter::once(index)
                .chain(std::iter::repeat_n(&zero, dims.len() - 1))
                .collect();
            step_inputs.push(native(
                native(input.dynamic_slice(&offsets, &slice))?.reshape(&dims[1..]),
            )?);
        }
        let start_ys = 1 + inputs.len();
        let end_ys = start_ys + eqn.outputs().len() - carry;
        let step_outputs = lower_body(&builder, child.expr, &args[end_ys..], &step_inputs)?;
        let mut next = vec![native(index + &native(builder.constant_r0(1_i64))?)?];
        next.extend_from_slice(&step_outputs[..carry]);
        next.extend_from_slice(&args[1 + carry..start_ys]);
        for (y, elem) in args[start_ys..end_ys].iter().zip(&step_outputs[carry..]) {
            let shape = native(y.array_shape())?;
            let mut slice = shape.dims().to_vec();
            slice[0] = 1;
            let offsets: Vec<_> = std::iter::once(index)
                .chain(std::iter::repeat_n(&zero, slice.len() - 1))
                .collect();
            next.push(native(
                y.dynamic_update_slice(&native(elem.reshape(&slice))?, &offsets),
            )?);
        }
        next.extend_from_slice(&args[end_ys..]);
        let body = native(native(builder.tuple(&next))?.build())?;
        let result = native(XlaOp::while_(
            cond,
            body,
            native(ctx.builder.tuple(&state))?,
        ))?;
        (1..1 + carry)
            .chain(start_ys..end_ys)
            .map(|i| native(result.get_tuple_element(i as i64)))
            .collect()
    }
}
crate::register_xla_lowering!(ScanOperation);

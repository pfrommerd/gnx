use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::io;
use std::sync::LazyLock;

use gnx_expr::{Eqn, Expr, Operation, ValueInfo, Var};
use gnx_math::{ArrayInfo, DType, Dim, ops::AddOperation};
use gnx_transforms::control_flow::{
    CallOperation, CondOperation, ScanOperation, ScanParameters, SwitchOperation, WhileOperation,
};
use xla::{ElementType, Shape, XlaBuilder, XlaComputation, XlaOp};

use super::{invalid, native, unsupported};

pub trait XlaLowering {
    fn lower_xla(
        &self,
        ctx: &LoweringContext<'_>,
        eqn: &Eqn,
        operands: &[XlaOp],
    ) -> io::Result<Vec<XlaOp>>;
}

type Rule = fn(&dyn Operation, &LoweringContext<'_>, &Eqn, &[XlaOp]) -> io::Result<Vec<XlaOp>>;

#[doc(hidden)]
pub struct Registration {
    type_id: fn() -> TypeId,
    rule: Rule,
}

impl Registration {
    pub const fn of<T: Operation + XlaLowering>() -> Self {
        Self {
            type_id: TypeId::of::<T>,
            rule: |op, ctx, eqn, inputs| {
                (op as &dyn Any)
                    .downcast_ref::<T>()
                    .expect("registered operation type")
                    .lower_xla(ctx, eqn, inputs)
            },
        }
    }
}

inventory::collect!(Registration);

#[macro_export]
macro_rules! register_xla_lowering {
    ($op:ty) => {
        $crate::__inventory::submit! { $crate::Registration::of::<$op>() }
    };
}

static RULES: LazyLock<HashMap<TypeId, Option<Rule>>> = LazyLock::new(|| {
    let mut rules = HashMap::new();
    for registration in inventory::iter::<Registration> {
        rules
            .entry((registration.type_id)())
            .and_modify(|rule| *rule = None)
            .or_insert(Some(registration.rule));
    }
    rules
});

struct Child<'a> {
    expr: &'a Expr,
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

pub(super) fn metadata(info: &ValueInfo) -> io::Result<&ArrayInfo> {
    info.downcast()
        .map_err(|_| unsupported("XLA only supports array values"))
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
        dimensions(info)?,
    ))
}

pub(super) fn lower(expr: &Expr) -> io::Result<XlaComputation> {
    let builder = XlaBuilder::new("gnx");
    let parameters = expr
        .closure()
        .iter()
        .chain(expr.inputs())
        .enumerate()
        .map(|(i, (_, info))| {
            native(builder.parameter_s(i as i64, &shape(metadata(info)?)?, &format!("arg{i}")))
        })
        .collect::<io::Result<Vec<_>>>()?;
    let (captures, inputs) = parameters.split_at(expr.closure().len());
    let results = lower_body(&builder, expr, captures, inputs)?;
    native(native(builder.tuple(&results))?.build())
}

fn lower_body(
    builder: &XlaBuilder,
    expr: &Expr,
    captures: &[XlaOp],
    inputs: &[XlaOp],
) -> io::Result<Vec<XlaOp>> {
    if captures.len() != expr.closure().len() || inputs.len() != expr.inputs().len() {
        return Err(invalid("XLA region argument arity mismatch"));
    }
    let mut values: HashMap<Var, XlaOp> = HashMap::new();
    for ((var, info), value) in expr
        .closure()
        .iter()
        .chain(expr.inputs())
        .zip(captures.iter().chain(inputs))
    {
        check_shape(value, metadata(info)?)?;
        values.insert(*var, value.clone());
    }
    for (index, eqn) in expr.eqns().iter().enumerate() {
        let op = eqn.op().operation();
        let results = (|| {
            let rule = match RULES.get(&(op as &dyn Any).type_id()) {
                Some(Some(rule)) => rule,
                Some(None) => return Err(invalid("multiple registered XLA lowerings")),
                None => return Err(unsupported("no registered XLA lowering")),
            };
            let operands = eqn
                .inputs()
                .iter()
                .map(|v| values[v].clone())
                .collect::<Vec<_>>();
            let children = eqn
                .subexprs()
                .iter()
                .map(|sub| Child {
                    expr: sub.expr(),
                    captures: sub.closure().iter().map(|v| values[v].clone()).collect(),
                })
                .collect();
            let results = rule(op, &LoweringContext { builder, children }, eqn, &operands)?;
            if results.len() != eqn.outputs().len() {
                return Err(invalid("XLA lowering result arity mismatch"));
            }
            for ((_, info), value) in eqn.outputs().iter().zip(&results) {
                check_shape(value, metadata(info)?)?;
            }
            Ok(results)
        })()
        .map_err(|err| {
            io::Error::new(
                err.kind(),
                format!(
                    "equation {index} ({}.{}) : {err}",
                    op.dialect().name(),
                    op.name()
                ),
            )
        })?;
        for ((var, _), value) in eqn.outputs().iter().zip(results) {
            values.insert(*var, value);
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

impl XlaLowering for AddOperation {
    fn lower_xla(
        &self,
        _ctx: &LoweringContext<'_>,
        eqn: &Eqn,
        inputs: &[XlaOp],
    ) -> io::Result<Vec<XlaOp>> {
        let [a, b] = inputs else {
            return Err(invalid("array.add requires two inputs"));
        };
        if !eqn.op().attrs().is_empty() || !eqn.subexprs().is_empty() {
            return Err(invalid(
                "array.add does not take attributes or subexpressions",
            ));
        }
        let result = if native(a.ty())? == xla::PrimitiveType::Pred {
            native(a.or(b))?
        } else {
            native(a + b)?
        };
        Ok(vec![result])
    }
}
crate::register_xla_lowering!(AddOperation);

fn local_sharding(eqn: &Eqn, inputs: &[XlaOp]) -> io::Result<Vec<XlaOp>> {
    let [(_, info)] = eqn.outputs() else {
        return Err(invalid("sharding annotation result arity"));
    };
    let spec = gnx_math::shard::Sharding::from_op(eqn.op(), metadata(info)?.shape().dims().len())
        .map_err(io::Error::other)?;
    if !spec.is_replicated() {
        return Err(unsupported("named sharding axes require a Mesh device"));
    }
    Ok(inputs.to_vec())
}

impl XlaLowering for gnx_math::shard::Constraint {
    fn lower_xla(
        &self,
        _: &LoweringContext<'_>,
        eqn: &Eqn,
        inputs: &[XlaOp],
    ) -> io::Result<Vec<XlaOp>> {
        local_sharding(eqn, inputs)
    }
}
impl XlaLowering for gnx_math::shard::Replicate {
    fn lower_xla(
        &self,
        _: &LoweringContext<'_>,
        eqn: &Eqn,
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
        _eqn: &Eqn,
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
        eqn: &Eqn,
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
        eqn: &Eqn,
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
        _eqn: &Eqn,
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
        let condition = native(predicate.build())?;
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
        eqn: &Eqn,
        inputs: &[XlaOp],
    ) -> io::Result<Vec<XlaOp>> {
        let configuration = ScanParameters::from_op(eqn.op()).map_err(io::Error::other)?;
        let carry = configuration
            .physical_carry_len()
            .map_err(io::Error::other)?;
        let [child] = ctx.children.as_slice() else {
            return Err(invalid("missing scan body"));
        };
        let ys = eqn
            .outputs()
            .get(carry..)
            .ok_or_else(|| invalid("invalid scan carry"))?
            .iter()
            .map(|(_, info)| {
                let info = metadata(info)?;
                native(
                    native(ctx.builder.zero(element(info.dtype())))?.broadcast(&dimensions(info)?),
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

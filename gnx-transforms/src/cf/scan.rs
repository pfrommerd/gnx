//! `cf.scan`: fold a captured body over axis zero of its inputs.

use gnx_expr::lower::{LoweringContext, RegionArgument};
use gnx_expr::{Capture, Dialect, Eqn, HEqn, Kind, ObjectTracer, Op, Operation, Trace, ValueInfo};
use gnx_math::{ArrayInfo, Dim};

use super::{
    BoundRegion, CONTROL_FLOW, RegionResult, StateLayout, check_child, count_attr, defer_regions,
    expect_inputs, expr_output_infos, parameters, signature, state_len, usize_attr, value_infos,
    value_results, with_state,
};

pub struct ScanOperation;

/// Construct a scan operation. The counts must fit in signed 64-bit attributes.
pub fn scan_op(carry_len: usize, length: usize) -> RegionResult<Op> {
    Ok(Op::new_with_attrs(
        &ScanOperation,
        [
            ("carry".into(), count_attr(carry_len)?),
            ("length".into(), count_attr(length)?),
        ]
        .into(),
    ))
}

#[derive(Clone, Copy)]
pub struct ScanParameters {
    /// Number of public carry values; private cell state follows these slots.
    pub carry_len: usize,
    pub state_len: usize,
    pub length: usize,
}

impl ScanParameters {
    pub fn from_op(op: &Op) -> RegionResult<Self> {
        fn get(op: &Op, field: &str) -> RegionResult<usize> {
            usize_attr(
                op.attrs()
                    .get(field)
                    .ok_or_else(|| signature("missing scan attribute"))?,
            )
        }
        if op
            .attrs()
            .keys()
            .any(|key| !matches!(key.as_str(), "carry" | "length" | "state"))
        {
            return Err(signature("unexpected scan attribute"));
        }
        let parameters = Self {
            carry_len: get(op, "carry")?,
            state_len: state_len(op)?,
            length: get(op, "length")?,
        };
        parameters.physical_carry_len()?;
        Ok(parameters)
    }

    pub fn physical_carry_len(self) -> RegionResult<usize> {
        self.carry_len
            .checked_add(self.state_len)
            .ok_or_else(|| signature("scan carry count overflow"))
    }

    /// `(physical carry length, length)` of a scan equation.
    pub(crate) fn layout(eqn: &Eqn) -> RegionResult<(usize, usize)> {
        let parameters = Self::from_op(eqn.op())?;
        Ok((parameters.physical_carry_len()?, parameters.length))
    }
}

fn slice(info: &ValueInfo) -> RegionResult<(usize, ValueInfo)> {
    let array = info
        .downcast::<ArrayInfo>()
        .map_err(|_| signature("scan inputs and stacked outputs must be arrays"))?;
    let [Dim::Fixed(length), tail @ ..] = array.shape().dims() else {
        return Err(signature("scan needs a fixed leading axis"));
    };
    Ok((
        *length,
        ValueInfo::new(ArrayInfo::new(
            tail.iter().copied().collect(),
            array.dtype(),
        )),
    ))
}

fn stack(info: &ValueInfo, length: usize) -> RegionResult<ValueInfo> {
    let array = info
        .downcast::<ArrayInfo>()
        .map_err(|_| signature("scan step outputs must be arrays"))?;
    let shape = std::iter::once(Dim::Fixed(length))
        .chain(array.shape().dims().iter().copied())
        .collect();
    Ok(ValueInfo::new(ArrayInfo::new(shape, array.dtype())))
}

/// Scan over axis zero of each input array. The body receives `carry` followed
/// by one slice from each input and returns the new carry followed by per-step
/// array results. Returns `(final_carry, stacked_results)`. If `length` is not
/// supplied it is inferred from the inputs; with no inputs it is required.
pub fn scan(
    body: &Capture,
    carry: &[Trace],
    inputs: &[Trace],
    length: Option<usize>,
) -> RegionResult<(Vec<Trace>, Vec<Trace>)> {
    let carry_infos: Vec<_> = carry.iter().map(|trace| trace.info().clone()).collect();
    let mut length = length;
    let mut argument_infos = carry_infos.clone();
    for input in inputs {
        let (size, item) = slice(input.info())?;
        if *length.get_or_insert(size) != size {
            return Err(signature("scan input lengths must match"));
        }
        argument_infos.push(item);
    }
    let length = length.ok_or_else(|| signature("scan without inputs requires a length"))?;
    expect_inputs(body.expr(), argument_infos.into_iter().map(Kind::Value))?;
    let outputs = value_results(body.expr())?;
    let Some((body_carry, ys)) = outputs.split_at_checked(carry.len()) else {
        return Err(signature("scan carry result arity"));
    };
    if body_carry != carry_infos {
        return Err(signature("scan carry types must be invariant"));
    }
    let mut infos = carry_infos;
    infos.extend(
        ys.iter()
            .map(|info| stack(info, length))
            .collect::<RegionResult<Vec<_>>>()?,
    );
    let mut result = defer_regions(
        scan_op(carry.len(), length)?,
        vec![BoundRegion {
            capture: body,
            inputs: &[],
            read_only: false,
        }],
        carry
            .iter()
            .chain(inputs)
            .cloned()
            .map(ObjectTracer::from)
            .collect(),
        infos,
    )?;
    let ys = result.split_off(carry.len());
    Ok((result, ys))
}

impl Operation for ScanOperation {
    fn name(&self) -> &'static str {
        "scan"
    }
    fn dialect(&self) -> &'static dyn Dialect {
        &CONTROL_FLOW
    }

    fn lower(&self, eqn: &HEqn, ctx: &mut LoweringContext<'_>) -> RegionResult<()> {
        let [body] = eqn.subexprs() else {
            return Err(signature("scan needs one body"));
        };
        let configuration = ScanParameters::from_op(eqn.op())?;
        let length = configuration.length;
        let carry_len = configuration.physical_carry_len()?;
        let (values, mut infos): (Vec<_>, Vec<_>) = eqn
            .inputs()
            .iter()
            .map(|&var| ctx.value_operand(var))
            .collect::<RegionResult<Vec<_>>>()?
            .into_iter()
            .unzip();
        let outputs = value_infos(eqn.outputs())?;
        let step_outputs = scan_step_infos(carry_len, length, outputs)?;
        if infos.get(..carry_len) != Some(&step_outputs[..carry_len]) {
            return Err(signature("scan carry types must be invariant"));
        }
        for info in &mut infos[carry_len..] {
            let (size, item) = slice(info)?;
            if size != length {
                return Err(signature("scan input length does not match"));
            }
            *info = item;
        }
        let args = vec![RegionArgument::Value; infos.len()];
        let regions = [ctx.lower_structured_region(body, &args, &infos)?];
        let state = StateLayout::new(ctx, &regions)?;
        let fixed =
            [regions[0].fix(ctx, &state.slots, carry_len, Some(carry_len), &step_outputs)?];
        let op = with_state(eqn.op(), state.cells.len())?;
        state.emit(ctx, eqn, op, values, (carry_len, carry_len), &fixed)
    }

    fn check_expr(&self, eqn: &Eqn) -> RegionResult<()> {
        let [body] = eqn.subexprs() else {
            return Err(signature("scan needs one body"));
        };
        let configuration = ScanParameters::from_op(eqn.op())?;
        let length = configuration.length;
        let carry_len = configuration.physical_carry_len()?;
        let step_outputs = scan_step_infos(carry_len, length, expr_output_infos(eqn))?;
        let inputs = body.expr().inputs();
        if eqn.inputs().len() != inputs.len()
            || inputs.len() < carry_len
            || !parameters(body.expr())
                .take(carry_len)
                .eq(step_outputs[..carry_len].iter())
        {
            return Err(signature("scan carry or operand signature"));
        }
        check_child(body.expr(), parameters(body.expr()).cloned(), &step_outputs)
    }
}

fn scan_step_infos(
    carry_len: usize,
    length: usize,
    mut outputs: Vec<ValueInfo>,
) -> RegionResult<Vec<ValueInfo>> {
    let Some(ys) = outputs.get_mut(carry_len..) else {
        return Err(signature("scan carry result arity"));
    };
    for info in ys {
        let (size, item) = slice(info)?;
        if size != length {
            return Err(signature("scan output length does not match"));
        }
        *info = item;
    }
    Ok(outputs)
}

//! `cf.scan`: fold a captured body over axis zero of its inputs.

use std::fmt;

use gnx_common::{ArrayInfo, Dim};
use gnx_expr::lower::{LoweringContext, RegionArgument};
use gnx_expr::{
    Capture, Eqn, HEqn, Invocation, Kind, ObjectTracer, Op, OpRef, Operation, Trace, VarInfo,
};
use rkyv::{Archive, Serialize};

use super::{
    RegionResult, StateLayout, add_state, check_child, count, expect_inputs, expr_output_infos,
    parameters, signature, stored, value_infos, value_results,
};

/// Builds a `cf::scan`, stored as a [`ScanOperation`].
#[derive(Archive, Serialize)]
#[rkyv(archived = ScanOperation)]
pub struct Scan {
    carry_len: u64,
    state_len: u64,
    length: u64,
}

impl ScanOperation {
    pub fn parameters(&self) -> ScanParameters {
        ScanParameters {
            carry_len: count(self.carry_len),
            state_len: count(self.state_len),
            length: count(self.length),
        }
    }
}

/// Construct a scan over `length` elements with `carry_len` carry values.
pub fn scan_op(carry_len: usize, length: usize) -> Op {
    ScanParameters {
        carry_len,
        state_len: 0,
        length,
    }
    .to_op()
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ScanParameters {
    /// Number of public carry values; private cell state follows these slots.
    pub carry_len: usize,
    pub state_len: usize,
    pub length: usize,
}

impl ScanParameters {
    fn to_op(self) -> Op {
        Op::new(Scan {
            carry_len: stored(self.carry_len),
            state_len: stored(self.state_len),
            length: stored(self.length),
        })
    }

    pub fn from_op(op: &Op) -> RegionResult<Self> {
        Self::from_operation(op.get())
    }

    /// Like [`ScanParameters::from_op`], for an operation read from an expression.
    pub fn from_op_ref(op: OpRef<'_>) -> RegionResult<Self> {
        Self::from_operation(op.get())
    }

    fn from_operation(op: &dyn Operation) -> RegionResult<Self> {
        let parameters = op
            .downcast_ref::<ScanOperation>()
            .ok_or_else(|| signature("expected a scan operation"))?
            .parameters();
        parameters.physical_carry_len()?;
        Ok(parameters)
    }

    pub fn physical_carry_len(self) -> RegionResult<usize> {
        self.carry_len
            .checked_add(self.state_len)
            .ok_or_else(|| signature("scan carry count overflow"))
    }

    /// `(physical carry length, length)` of a scan equation.
    pub(crate) fn layout(eqn: Eqn<'_>) -> RegionResult<(usize, usize)> {
        let parameters = Self::from_op_ref(eqn.op())?;
        Ok((parameters.physical_carry_len()?, parameters.length))
    }
}

pub(super) fn slice(info: &VarInfo) -> RegionResult<(usize, VarInfo)> {
    let array = ArrayInfo::from_info(info)
        .ok_or_else(|| signature("scan inputs and stacked outputs must be arrays"))?;
    let [Dim::Fixed(length), tail @ ..] = array.shape().dims() else {
        return Err(signature("scan needs a fixed leading axis"));
    };
    Ok((
        *length,
        VarInfo::new(ArrayInfo::new(
            tail.iter().copied().collect::<gnx_common::Shape>(),
            array.dtype(),
        )),
    ))
}

fn stack(info: &VarInfo, length: usize) -> RegionResult<VarInfo> {
    let array =
        ArrayInfo::from_info(info).ok_or_else(|| signature("scan step outputs must be arrays"))?;
    let shape: gnx_common::Shape = std::iter::once(Dim::Fixed(length))
        .chain(array.shape().dims().iter().copied())
        .collect();
    Ok(VarInfo::new(ArrayInfo::new(shape, array.dtype())))
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
    expect_inputs(
        body.expr(),
        argument_infos.into_iter().map(|info| (Kind::Value, info)),
    )?;
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
    let mut result = Invocation::apply(
        scan_op(carry.len(), length),
        vec![body.into()],
        carry
            .iter()
            .chain(inputs)
            .cloned()
            .map(ObjectTracer::from)
            .collect(),
        infos,
    );
    let ys = result.split_off(carry.len());
    Ok((result, ys))
}

impl Operation for ScanOperation {
    fn display(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ScanParameters {
            carry_len,
            length,
            state_len,
        } = self.parameters();
        write!(f, "cf::scan<carry: {carry_len}, length: {length}")?;
        if state_len != 0 {
            write!(f, ", state: {state_len}")?;
        }
        write!(f, ">")
    }

    fn lower(&self, eqn: HEqn<'_>, ctx: &mut LoweringContext<'_>) -> RegionResult<()> {
        let mut subexprs = eqn.subexprs();
        let (Some(body), None) = (subexprs.next(), subexprs.next()) else {
            return Err(signature("scan needs one body"));
        };
        let configuration = self.parameters();
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
        let parameters = ScanParameters {
            state_len: add_state(configuration.state_len, state.cells.len())?,
            ..configuration
        };
        parameters.physical_carry_len()?;
        let op = parameters.to_op();
        state.emit(ctx, eqn, op, values, (carry_len, carry_len), &fixed)
    }

    fn check_expr(&self, eqn: Eqn<'_>) -> RegionResult<()> {
        let mut subexprs = eqn.subexprs();
        let (Some(body), None) = (subexprs.next(), subexprs.next()) else {
            return Err(signature("scan needs one body"));
        };
        let configuration = self.parameters();
        let length = configuration.length;
        let carry_len = configuration.physical_carry_len()?;
        let step_outputs = scan_step_infos(carry_len, length, expr_output_infos(eqn))?;
        let inputs = body.expr().inputs();
        if eqn.inputs().len() != inputs.len()
            || inputs.len() < carry_len
            || !parameters(body.expr())
                .take(carry_len)
                .eq(step_outputs[..carry_len].iter().cloned())
        {
            return Err(signature("scan carry or operand signature"));
        }
        check_child(body.expr(), parameters(body.expr()), &step_outputs)
    }
}

fn scan_step_infos(
    carry_len: usize,
    length: usize,
    mut outputs: Vec<VarInfo>,
) -> RegionResult<Vec<VarInfo>> {
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

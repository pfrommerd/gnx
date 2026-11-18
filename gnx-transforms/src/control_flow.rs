//! Trace and lower the `cf` dialect's control-flow operations.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use gnx_expr::lower::{FixedRegion, LoweringContext, RegionArgument, RegionCell, StructuredRegion};
use gnx_expr::{
    Attr, CallError, Capture, CellAccess, CellKey, CellTrace, DeferredApplyPlan, DeferredLocal,
    DeferredProjection, DeferredRegion, Dialect, Eqn, Expr, HEqn, HExpr, Invocation, Kind,
    LoweredExpr, LoweringError, ObjectTracer, Op, Operation, Trace, TraceInput, ValueInfo, Var,
};
use gnx_math::{ArrayInfo, DType, Dim, Shape};

pub struct ControlFlowDialect;
pub struct CallOperation;

static CONTROL_FLOW: ControlFlowDialect = ControlFlowDialect;
static CALL: CallOperation = CallOperation;
gnx_expr::register_dialect!(&CONTROL_FLOW);

impl Dialect for ControlFlowDialect {
    fn name(&self) -> &'static str {
        "cf"
    }

    fn create(&self, name: &'static str) -> Option<&'static dyn Operation> {
        match name {
            "call" => Some(&CALL),
            "cond" => Some(&CondOperation),
            "switch" => Some(&SwitchOperation),
            "scan" => Some(&ScanOperation),
            "while_loop" => Some(&WhileOperation),
            _ => None,
        }
    }
}

impl Operation for CallOperation {
    fn name(&self) -> &'static str {
        "call"
    }

    fn dialect(&self) -> &'static dyn Dialect {
        &CONTROL_FLOW
    }

    fn region_operand(&self, eqn: &HEqn, region: usize, param: usize) -> Option<Var> {
        (region == 0)
            .then(|| eqn.inputs().get(param).copied())
            .flatten()
    }

    fn lower(&self, eqn: &HEqn, ctx: &mut LoweringContext<'_>) -> Result<(), LoweringError> {
        let [sub] = eqn.subexprs() else {
            return Err(LoweringError::CallArity);
        };
        let body = sub.expr();
        if sub.closure().len() != body.closure().len()
            || eqn.inputs().len() != body.inputs().len()
            || eqn.outputs().len() != body.outputs().len()
        {
            return Err(LoweringError::CallArity);
        }
        let region = ctx.lower_region(sub, eqn.inputs())?;
        let physical = ctx.allocate_results(&region.program().output_infos);
        ctx.bind_region_results(eqn.outputs(), &region, &physical)?;
        ctx.apply_region_updates(&region, &physical)?;
        if !region.program().expr.eqns().is_empty() || !physical.is_empty() {
            ctx.emit(
                eqn.op().clone(),
                region.inputs(),
                physical,
                vec![region.subexpr()],
            );
        }
        Ok(())
    }

    fn check_expr(&self, eqn: &Eqn) -> Result<(), LoweringError> {
        let [sub] = eqn.subexprs() else {
            return Err(LoweringError::CallArity);
        };
        if eqn.inputs().len() != sub.expr().inputs().len()
            || eqn.outputs().len() != sub.expr().outputs().len()
        {
            return Err(LoweringError::CallArity);
        }
        Ok(())
    }
}

pub fn call_op() -> Op {
    Op::new(&CALL)
}

/// Bind a captured function without specializing potentially aliased cells.
pub fn call(capture: &Capture, inputs: &[ObjectTracer]) -> Vec<ObjectTracer> {
    defer_call(Arc::new(capture.expr().clone()), capture.closure(), inputs)
}

/// Record a call with a value-only boundary in the trace.
pub fn call_immutable<C, I>(
    body: Arc<HExpr>,
    captures: C,
    inputs: I,
    outputs: Vec<ValueInfo>,
) -> Vec<Trace>
where
    C: IntoIterator,
    C::Item: Into<TraceInput>,
    I: IntoIterator,
    I::Item: Into<TraceInput>,
{
    assert!(
        body.closure()
            .iter()
            .chain(body.inputs())
            .all(|(_, kind)| matches!(kind, Kind::Value(_)))
    );
    assert!(
        body.outputs()
            .iter()
            .all(|var| matches!(body.kind(var), Some(Kind::Value(_))))
    );
    let captures: Vec<_> = captures.into_iter().map(Into::into).collect();
    let inputs: Vec<_> = inputs.into_iter().map(Into::into).collect();
    assert_eq!(
        inputs.len(),
        body.inputs().len(),
        "call input arity mismatch"
    );
    assert_eq!(
        outputs.len(),
        body.outputs().len(),
        "call output arity mismatch"
    );
    Invocation::invoke(call_op(), vec![(body, captures)], inputs, outputs)
        .into_iter()
        .map(|tracer| tracer.into_trace_ref())
        .collect()
}

/// Replay a specialization with arguments having the same alias pattern.
pub fn call_lowered(
    body: &LoweredExpr,
    captures: &[ObjectTracer],
    inputs: &[ObjectTracer],
) -> Result<Vec<ObjectTracer>, CallError> {
    body.replay(captures, inputs, call_immutable)
}

/// Bind a high-level body and explicit captures without lowering the call.
/// The caller's alias specialization is applied later, during lowering.
pub fn defer_call(
    body: Arc<HExpr>,
    captures: &[ObjectTracer],
    inputs: &[ObjectTracer],
) -> Vec<ObjectTracer> {
    assert_eq!(captures.len(), body.closure().len());
    assert_eq!(inputs.len(), body.inputs().len());
    let bindings: BTreeMap<_, _> = body
        .closure()
        .iter()
        .chain(body.inputs())
        .map(|(var, _)| *var)
        .zip(captures.iter().chain(inputs))
        .collect();
    let mut locals = BTreeMap::<CellKey, DeferredLocal>::new();
    let mut external_access = CellAccess::Unused;
    for (var, object) in &bindings {
        if let ObjectTracer::Cell(cell) = object {
            let access = body.cell_access(var).unwrap_or(CellAccess::Unused);
            if cell.is_maybe_aliased() {
                external_access = external_access.combine(access);
            } else {
                let local = locals
                    .entry(CellKey::from(cell))
                    .or_insert_with(|| DeferredLocal {
                        cell: cell.clone(),
                        written: false,
                    });
                local.written |= matches!(access, CellAccess::Write | CellAccess::ReadWrite);
            }
        }
    }

    enum ResultSlot<'a> {
        Passthrough(&'a ObjectTracer),
        Value(usize),
        FreshCell(usize),
    }
    let outputs: Vec<_> = body
        .outputs()
        .iter()
        .map(|var| {
            body.kind(var)
                .expect("undefined deferred call output")
                .clone()
        })
        .collect();
    let mut projections = Vec::new();
    let mut fresh = BTreeMap::new();
    let slots: Vec<_> = body
        .outputs()
        .iter()
        .zip(&outputs)
        .enumerate()
        .map(|(index, (var, kind))| {
            if let Some(&binding) = bindings.get(var) {
                return ResultSlot::Passthrough(binding);
            }
            match kind {
                Kind::Value(_) => {
                    let slot = projections.len();
                    projections.push(DeferredProjection::Value(index));
                    ResultSlot::Value(slot)
                }
                Kind::Cell(_) => ResultSlot::FreshCell(*fresh.entry(*var).or_insert_with(|| {
                    let slot = projections.len();
                    projections.push(DeferredProjection::CellContents(index));
                    slot
                })),
            }
        })
        .collect();
    let results = Invocation::defer_apply(
        call_op(),
        DeferredApplyPlan {
            regions: vec![DeferredRegion {
                body,
                captures: captures.to_vec(),
            }],
            inputs: inputs.to_vec(),
            outputs,
            projections,
            locals: locals.into_values().collect(),
            external_access,
        },
    );
    let mut fresh = BTreeMap::new();
    slots
        .into_iter()
        .map(|slot| match slot {
            ResultSlot::Passthrough(value) => value.clone(),
            ResultSlot::Value(index) => results[index].clone().into(),
            ResultSlot::FreshCell(index) => fresh
                .entry(index)
                .or_insert_with(|| CellTrace::new(results[index].clone()))
                .clone()
                .into(),
        })
        .collect()
}

pub struct CondOperation;
pub struct SwitchOperation;
pub struct WhileOperation;
pub struct ScanOperation;

type RegionResult<T> = std::result::Result<T, LoweringError>;

fn signature(message: &'static str) -> LoweringError {
    LoweringError::RegionSignature(message)
}

pub fn cond_op() -> Op {
    Op::new(&CondOperation)
}
pub fn switch_op() -> Op {
    Op::new(&SwitchOperation)
}
pub fn while_op() -> Op {
    Op::new(&WhileOperation)
}

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

fn count_attr(value: usize) -> RegionResult<Attr> {
    i64::try_from(value)
        .map(Attr::I64)
        .map_err(|_| signature("control-flow count exceeds i64"))
}

fn usize_attr(value: &Attr) -> RegionResult<usize> {
    match value {
        Attr::I64(value) => usize::try_from(*value)
            .map_err(|_| signature("control-flow count is negative or exceeds usize")),
        _ => Err(signature("expected an i64 attribute")),
    }
}

fn state_len(op: &Op) -> RegionResult<usize> {
    op.attrs()
        .get("state")
        .map(usize_attr)
        .transpose()
        .map(|size| size.unwrap_or(0))
}

fn with_state(op: &Op, new_slots: usize) -> RegionResult<Op> {
    if new_slots == 0 {
        return Ok(op.clone());
    }
    let mut attrs = op.attrs().clone();
    let size = state_len(op)?
        .checked_add(new_slots)
        .ok_or_else(|| signature("state count overflow"))?;
    attrs.insert("state".into(), count_attr(size)?);
    Ok(Op::new_with_attrs(op.operation(), attrs))
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
}

fn scalar(dtype: DType) -> ValueInfo {
    ValueInfo::new(ArrayInfo::new(Shape::fixed([]), dtype))
}

fn value_results(body: &HExpr) -> RegionResult<Vec<ValueInfo>> {
    body.outputs()
        .iter()
        .map(|var| match body.kind(var) {
            Some(Kind::Value(info)) => Ok(info.clone()),
            _ => Err(signature(
                "control flow must return values, not cell handles",
            )),
        })
        .collect()
}

fn value_infos(outputs: &[(Var, Kind)]) -> RegionResult<Vec<ValueInfo>> {
    outputs
        .iter()
        .map(|(var, kind)| match kind {
            Kind::Value(info) => Ok(info.clone()),
            Kind::Cell(_) => Err(LoweringError::ExpectedValue(*var)),
        })
        .collect()
}

fn expect_inputs(body: &HExpr, expected: impl IntoIterator<Item = Kind>) -> RegionResult<()> {
    if body
        .inputs()
        .iter()
        .map(|(_, kind)| kind.clone())
        .eq(expected)
    {
        Ok(())
    } else {
        Err(signature("region input signature"))
    }
}

fn object_kind(object: &ObjectTracer) -> Kind {
    match object {
        ObjectTracer::Ref(trace) => Kind::Value(trace.info().clone()),
        ObjectTracer::Cell(cell) => Kind::Cell(cell.type_info().clone()),
    }
}

struct BoundRegion<'a> {
    capture: &'a Capture,
    inputs: &'a [ObjectTracer],
    read_only: bool,
}

fn defer_regions(
    op: Op,
    regions: Vec<BoundRegion<'_>>,
    inputs: Vec<ObjectTracer>,
    infos: Vec<ValueInfo>,
) -> RegionResult<Vec<Trace>> {
    let mut locals = BTreeMap::<CellKey, DeferredLocal>::new();
    let mut external_access = CellAccess::Unused;
    for bound in &regions {
        let capture = bound.capture;
        let body = capture.expr();
        for ((var, _), object) in body
            .closure()
            .iter()
            .zip(capture.closure())
            .chain(body.inputs().iter().zip(bound.inputs))
        {
            if let ObjectTracer::Cell(cell) = object {
                let access = body.cell_access(var).unwrap_or(CellAccess::Unused);
                let writes = matches!(access, CellAccess::Write | CellAccess::ReadWrite);
                if bound.read_only && writes {
                    return Err(signature("while predicate cannot write a captured cell"));
                }
                if cell.is_maybe_aliased() {
                    external_access = external_access.combine(access);
                } else {
                    locals
                        .entry(CellKey::from(cell))
                        .or_insert_with(|| DeferredLocal {
                            cell: cell.clone(),
                            written: false,
                        })
                        .written |= writes;
                }
            }
        }
    }
    Ok(Invocation::defer_apply(
        op,
        DeferredApplyPlan {
            regions: regions
                .into_iter()
                .map(|region| DeferredRegion {
                    body: Arc::new(region.capture.expr().clone()),
                    captures: region.capture.closure().to_vec(),
                })
                .collect(),
            inputs,
            projections: (0..infos.len()).map(DeferredProjection::Value).collect(),
            outputs: infos.into_iter().map(Kind::Value).collect(),
            locals: locals.into_values().collect(),
            external_access,
        },
    ))
}

/// Select one of two captured branches using a scalar boolean array. Both
/// branches take the same arguments and must return matching immutable values.
/// They may read or write cells in their arguments or captures.
pub fn cond(
    predicate: Trace,
    on_true: &Capture,
    on_false: &Capture,
    inputs: &[ObjectTracer],
) -> RegionResult<Vec<Trace>> {
    record_branch(
        cond_op(),
        DType::Bool,
        predicate,
        &[on_true, on_false],
        inputs,
    )
}

/// Select a captured branch using a scalar i32 array. Indices are clamped to
/// the valid range. All branches take the same arguments and return the same
/// value signature.
pub fn switch(
    index: Trace,
    branches: &[&Capture],
    inputs: &[ObjectTracer],
) -> RegionResult<Vec<Trace>> {
    record_branch(switch_op(), DType::I32, index, branches, inputs)
}

fn record_branch(
    op: Op,
    dtype: DType,
    selector: Trace,
    branches: &[&Capture],
    inputs: &[ObjectTracer],
) -> RegionResult<Vec<Trace>> {
    let Some(first) = branches.first() else {
        return Err(signature("switch needs at least one branch"));
    };
    if selector.info() != &scalar(dtype) {
        return Err(signature("branch selector type"));
    }
    let infos = value_results(first.expr())?;
    for branch in branches {
        expect_inputs(branch.expr(), inputs.iter().map(object_kind))?;
        if value_results(branch.expr())? != infos {
            return Err(signature("branch result signatures must match"));
        }
    }
    defer_regions(
        op,
        branches
            .iter()
            .map(|branch| BoundRegion {
                capture: branch,
                inputs,
                read_only: false,
            })
            .collect(),
        std::iter::once(selector.into())
            .chain(inputs.iter().cloned())
            .collect(),
        infos,
    )
}

/// Loop over immutable carry values. `predicate` returns a single bool scalar;
/// it may read captured cells but cannot write them. `body` returns the original
/// carry signature and may read and write captured cells.
pub fn while_loop(
    predicate: &Capture,
    body: &Capture,
    carry: &[Trace],
) -> RegionResult<Vec<Trace>> {
    let infos: Vec<_> = carry.iter().map(|trace| trace.info().clone()).collect();
    for region in [predicate, body] {
        expect_inputs(region.expr(), infos.iter().cloned().map(Kind::Value))?;
    }
    if value_results(predicate.expr())? != [scalar(DType::Bool)]
        || value_results(body.expr())? != infos
    {
        return Err(signature("while predicate or carry result signature"));
    }
    defer_regions(
        while_op(),
        vec![
            BoundRegion {
                capture: predicate,
                inputs: &[],
                read_only: true,
            },
            BoundRegion {
                capture: body,
                inputs: &[],
                read_only: false,
            },
        ],
        carry.iter().cloned().map(ObjectTracer::from).collect(),
        infos,
    )
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

struct StateLayout {
    cells: Vec<RegionCell>,
    slots: Vec<(RegionCell, ValueInfo)>,
    initial: Vec<Var>,
}

impl StateLayout {
    fn new(ctx: &LoweringContext<'_>, regions: &[StructuredRegion]) -> RegionResult<Self> {
        let cells: Vec<_> = regions
            .iter()
            .flat_map(StructuredRegion::writes)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let mut slots = Vec::new();
        let mut initial = Vec::new();
        for &cell in &cells {
            let (value, info) = ctx.cell_state(cell)?;
            initial.push(value);
            slots.push((cell, info));
        }
        Ok(Self {
            cells,
            slots,
            initial,
        })
    }

    fn emit(
        self,
        ctx: &mut LoweringContext<'_>,
        eqn: &HEqn,
        op: Op,
        mut inputs: Vec<Var>,
        positions: (usize, usize),
        regions: &[FixedRegion],
    ) -> RegionResult<()> {
        let (input_position, output_position) = positions;
        inputs.splice(input_position..input_position, self.initial);
        let mut infos = value_infos(eqn.outputs())?;
        infos.splice(
            output_position..output_position,
            self.slots.into_iter().map(|(_, info)| info),
        );
        let output = ctx.allocate_results(&infos);
        ctx.bind_structured_results(eqn.outputs(), &self.cells, output_position, &output)?;
        ctx.emit(
            op,
            inputs,
            output,
            regions.iter().map(FixedRegion::subexpr).collect(),
        );
        Ok(())
    }
}

fn state_only_attrs(op: &Op, inputs: usize, outputs: usize) -> RegionResult<()> {
    if op.attrs().keys().any(|key| key != "state") || state_len(op)? > inputs.min(outputs) {
        return Err(signature("invalid control-flow state layout"));
    }
    Ok(())
}

fn lower_branch(eqn: &HEqn, ctx: &mut LoweringContext<'_>, dtype: DType) -> RegionResult<()> {
    let [selector, operands @ ..] = eqn.inputs() else {
        return Err(signature("missing branch selector"));
    };
    let (selector, info) = ctx.value_operand(*selector)?;
    if info != scalar(dtype) {
        return Err(signature("branch selector type"));
    }
    state_only_attrs(eqn.op(), operands.len(), eqn.outputs().len())?;
    let first = eqn
        .subexprs()
        .first()
        .ok_or_else(|| signature("missing branch"))?;
    if first.expr().inputs().len() != operands.len() {
        return Err(signature("branch operand arity"));
    }
    let mut values = vec![selector];
    let mut infos = Vec::new();
    let mut args = Vec::new();
    for (&operand, (_, kind)) in operands.iter().zip(first.expr().inputs()) {
        match kind {
            Kind::Value(_) => {
                let (value, info) = ctx.value_operand(operand)?;
                values.push(value);
                infos.push(info);
                args.push(RegionArgument::Value);
            }
            Kind::Cell(_) => args.push(RegionArgument::Cell(operand)),
        }
    }
    let regions = eqn
        .subexprs()
        .iter()
        .map(|sub| ctx.lower_structured_region(sub, &args, &infos))
        .collect::<RegionResult<Vec<_>>>()?;
    let state = StateLayout::new(ctx, &regions)?;
    let outputs = value_infos(eqn.outputs())?;
    let fixed = regions
        .iter()
        .map(|region| {
            region.fix(
                ctx,
                &state.slots,
                infos.len(),
                Some(outputs.len()),
                &outputs,
            )
        })
        .collect::<RegionResult<Vec<_>>>()?;
    let input_len = values.len();
    let op = with_state(eqn.op(), state.cells.len())?;
    state.emit(ctx, eqn, op, values, (input_len, outputs.len()), &fixed)
}

fn parameters(expr: &Expr) -> impl Iterator<Item = &ValueInfo> {
    expr.inputs().iter().map(|(_, info)| info)
}

fn check_child(
    expr: &Expr,
    inputs: impl IntoIterator<Item = ValueInfo>,
    outputs: &[ValueInfo],
) -> RegionResult<()> {
    if !parameters(expr).cloned().eq(inputs)
        || !expr
            .outputs()
            .iter()
            .map(|var| expr.value_info(var))
            .eq(outputs.iter().map(Some))
    {
        return Err(signature("fixed region signature"));
    }
    Ok(())
}

fn expr_output_infos(eqn: &Eqn) -> Vec<ValueInfo> {
    eqn.outputs().iter().map(|(_, info)| info.clone()).collect()
}

fn check_branches(eqn: &Eqn) -> RegionResult<()> {
    let first = eqn
        .subexprs()
        .first()
        .ok_or_else(|| signature("missing branch"))?
        .expr();
    let outputs = expr_output_infos(eqn);
    if eqn.inputs().len().checked_sub(1) != Some(first.inputs().len()) {
        return Err(signature("branch operand signature"));
    }
    state_only_attrs(eqn.op(), first.inputs().len(), outputs.len())?;
    for sub in eqn.subexprs() {
        check_child(sub.expr(), parameters(first).cloned(), &outputs)?;
    }
    Ok(())
}

macro_rules! branch_op {
    ($ty:ty, $name:literal, $dtype:expr, $arity:expr) => {
        impl Operation for $ty {
            fn name(&self) -> &'static str {
                $name
            }
            fn dialect(&self) -> &'static dyn Dialect {
                &CONTROL_FLOW
            }
            fn region_operand(&self, eqn: &HEqn, _region: usize, param: usize) -> Option<Var> {
                eqn.inputs().get(1 + param).copied()
            }
            fn lower(&self, eqn: &HEqn, ctx: &mut LoweringContext<'_>) -> RegionResult<()> {
                if !($arity)(eqn.subexprs().len()) {
                    return Err(signature("branch count"));
                }
                lower_branch(eqn, ctx, $dtype)
            }
            fn check_expr(&self, eqn: &Eqn) -> RegionResult<()> {
                if !($arity)(eqn.subexprs().len()) {
                    return Err(signature("branch count"));
                }
                check_branches(eqn)
            }
        }
    };
}
branch_op!(CondOperation, "cond", DType::Bool, |n| n == 2);
branch_op!(SwitchOperation, "switch", DType::I32, |n| n > 0);

impl Operation for WhileOperation {
    fn name(&self) -> &'static str {
        "while_loop"
    }
    fn dialect(&self) -> &'static dyn Dialect {
        &CONTROL_FLOW
    }

    fn lower(&self, eqn: &HEqn, ctx: &mut LoweringContext<'_>) -> RegionResult<()> {
        let [predicate, body] = eqn.subexprs() else {
            return Err(signature("while needs a predicate and body"));
        };
        let (values, infos): (Vec<_>, Vec<_>) = eqn
            .inputs()
            .iter()
            .map(|&var| ctx.value_operand(var))
            .collect::<RegionResult<Vec<_>>>()?
            .into_iter()
            .unzip();
        if value_infos(eqn.outputs())? != infos {
            return Err(signature("while carry result signature"));
        }
        state_only_attrs(eqn.op(), infos.len(), infos.len())?;
        let args = vec![RegionArgument::Value; infos.len()];
        let regions = [
            ctx.lower_structured_region(predicate, &args, &infos)?,
            ctx.lower_structured_region(body, &args, &infos)?,
        ];
        if regions[0].writes().next().is_some() {
            return Err(signature("while predicate cannot write a captured cell"));
        }
        let state = StateLayout::new(ctx, &regions)?;
        let size = infos.len();
        let fixed = [
            regions[0].fix(ctx, &state.slots, size, None, &[scalar(DType::Bool)])?,
            regions[1].fix(ctx, &state.slots, size, Some(size), &infos)?,
        ];
        let op = with_state(eqn.op(), state.cells.len())?;
        state.emit(ctx, eqn, op, values, (size, size), &fixed)
    }

    fn check_expr(&self, eqn: &Eqn) -> RegionResult<()> {
        let [predicate, body] = eqn.subexprs() else {
            return Err(signature("while needs a predicate and body"));
        };
        let carry = expr_output_infos(eqn);
        if carry.len() != eqn.inputs().len() {
            return Err(signature("while carry count"));
        }
        state_only_attrs(eqn.op(), carry.len(), carry.len())?;
        check_child(
            predicate.expr(),
            carry.iter().cloned(),
            &[scalar(DType::Bool)],
        )?;
        check_child(body.expr(), carry.iter().cloned(), &carry)
    }
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

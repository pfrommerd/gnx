//! Signature-wide witnesses for uniform, bounded axes. A native pointwise
//! kernel may compute padding lanes because no logical result observes them.
use crate::{ArrayInfo, ArrayInfoData, Dim, Shape};
use gnx_expr::{Expr, Symbol};
use std::{collections::HashMap, io};

fn invalid(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, msg.into())
}

#[derive(Default)]
pub struct BoundSizes(HashMap<Symbol, usize>);
impl BoundSizes {
    pub(super) fn bind(&mut self, formal: &ArrayInfo, actual: &ArrayInfo) -> io::Result<()> {
        let mismatch = || invalid(format!("expected {formal}, got {actual}"));
        if formal.dtype() != actual.dtype()
            || formal.shape().dims().len() != actual.shape().dims().len()
        {
            return Err(mismatch());
        }
        for (&axis, &size) in formal.shape().dims().iter().zip(actual.shape().dims()) {
            match (axis, size) {
                (Dim::Bounded { symbol, max }, Dim::Fixed(n)) if n <= max as usize => {
                    if self
                        .0
                        .insert(symbol, n)
                        .is_some_and(|previous| previous != n)
                    {
                        return Err(invalid(format!("conflicting lengths for {symbol}")));
                    }
                }
                (Dim::Fixed(a), Dim::Fixed(b)) if a == b => (),
                _ => return Err(mismatch()),
            }
        }
        Ok(())
    }
    pub fn resolve(&self, formal: &ArrayInfo) -> io::Result<ArrayInfo> {
        if !formal.has_bounded_axes() {
            return Ok(formal.clone());
        }
        let shape = formal
            .shape()
            .dims()
            .iter()
            .map(|&d| match d {
                Dim::Bounded { symbol, max } => self
                    .0
                    .get(&symbol)
                    .copied()
                    .filter(|n| *n <= max as usize)
                    .map(Dim::Fixed)
                    .ok_or_else(|| invalid(format!("unbound or out-of-range {symbol}"))),
                other => Ok(other),
            })
            .collect::<io::Result<Shape>>()?;
        Ok(ArrayInfo::new(shape, formal.dtype()))
    }
}

/// A logical prefix of an already allocated capacity buffer. Padding can
/// contain arbitrary values; pad_to_bound explicitly masks it. No device copy.
#[derive(Clone, Debug)]
pub struct BoundedBuffer {
    storage: crate::DataHandle,
    logical: ArrayInfo,
}
impl BoundedBuffer {
    pub fn new(storage: crate::DataHandle, logical: Shape) -> io::Result<Self> {
        let capacity = storage.array_info().shape().dims();
        if logical.dims().len() != capacity.len()
            || logical
                .dims()
                .iter()
                .zip(capacity)
                .any(|(n, c)| !matches!((n,c),(Dim::Fixed(n),Dim::Fixed(c)) if n<=c))
        {
            return Err(invalid("logical shape exceeds storage"));
        }
        Ok(Self {
            logical: ArrayInfo::new(logical, storage.array_info().dtype()),
            storage,
        })
    }
    pub fn storage(&self) -> &crate::DataHandle {
        &self.storage
    }
}
impl std::fmt::Display for BoundedBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "BoundedBuffer({})", self.logical)
    }
}
impl gnx_expr::ConcreteValue for BoundedBuffer {
    fn to_info(&self) -> gnx_expr::VarInfo {
        gnx_expr::VarInfo::new(self.logical.clone())
    }
}

impl BoundSizes {
    pub(super) fn scalar(&self, s: Symbol) -> io::Result<crate::DataHandle> {
        let n = i32::try_from(*self.0.get(&s).ok_or_else(|| invalid("unbound length"))?)
            .map_err(|_| invalid("extent exceeds i32"))?;
        Ok(crate::DataHandle::new(crate::HostBuffer::copy_from_slice(
            ArrayInfo::new(Shape::fixed([]), crate::DType::I32),
            &n.to_le_bytes(),
        )?))
    }
}

/// Replace logical length queries with native scalar inputs. Materializing a
/// prefix inserts masks, including when the caller supplied existing KV storage.
pub(super) fn lower(
    root: std::sync::Arc<gnx_expr::RootExpr>,
) -> io::Result<(std::sync::Arc<gnx_expr::RootExpr>, Vec<Symbol>)> {
    use crate::{DType, ops};
    use gnx_expr::{EqnSource, ExprBuilder, VarInfo};
    let expr = root.view();
    if !expr.eqns().any(|e| {
        e.op()
            .downcast_ref::<ops::bounded::AxisLengthOperation>()
            .is_some()
            || e.op()
                .downcast_ref::<ops::bounded::PadToBoundOperation>()
                .is_some()
    }) {
        return Ok((root, vec![]));
    }
    let mut builder = expr.to_builder();
    let mut eqns = Vec::new();
    let mut witnesses = Vec::new();
    let mut vars = HashMap::new();
    let scalar = || VarInfo::new(ArrayInfo::new(Shape::fixed([]), DType::I32));
    for eq in expr.eqns() {
        let length = eq
            .op()
            .downcast_ref::<ops::bounded::AxisLengthOperation>()
            .map(|op| op.axis());
        if length.is_none()
            && eq
                .op()
                .downcast_ref::<ops::bounded::PadToBoundOperation>()
                .is_none()
        {
            eqns.push(eq.to_source());
            continue;
        }
        let x = eq.inputs()[0];
        let info = expr
            .value_info(x)
            .unwrap()
            .downcast::<ArrayInfoData>()
            .ok_or_else(|| invalid("expected bounded array"))?
            .to_native();
        let cap = info.capacity();
        let out = eq.outputs().next().unwrap();
        let mut v = x;
        for (axis, &d) in info.shape().dims().iter().enumerate() {
            let Dim::Bounded { symbol, max } = d else {
                continue;
            };
            if length.is_some_and(|i| i != axis) {
                continue;
            }
            i32::try_from(max).map_err(|_| invalid("extent exceeds i32"))?;
            let size = *vars.entry(symbol).or_insert_with(|| {
                witnesses.push(symbol);
                builder.input(scalar())
            });
            if length.is_some() {
                v = size;
                break;
            }
            let mut emit = |op, inputs: Vec<_>, dtype| {
                let var = builder.fresh_var();
                eqns.push(EqnSource::new(
                    op,
                    vec![],
                    inputs,
                    vec![(
                        var,
                        VarInfo::new(ArrayInfo::new(cap.shape().clone(), dtype)),
                    )],
                ));
                var
            };
            let times = emit(ops::creation::iota(axis), vec![], DType::I32);
            let limit = emit(ops::broadcast_in_dim_op(&[]), vec![size], DType::I32);
            let mask = emit(ops::elementwise::lt(), vec![times, limit], DType::Bool);
            let zero = emit(ops::zeros_op(), vec![], cap.dtype());
            v = emit(ops::select_op(), vec![mask, v, zero], cap.dtype());
        }
        if length.is_some() && v == x {
            return Err(invalid("axis_length requires a bounded axis"));
        }
        eqns.push(EqnSource::new(
            ops::reshape_op(),
            vec![],
            vec![v],
            vec![(out.var(), out.info_ref().to_owned())],
        ));
    }
    let mut lowered = ExprBuilder::from_parts(
        builder.closure_symbols().to_vec(),
        builder.closure_inputs().to_vec(),
        builder.inputs().to_vec(),
        eqns,
        expr.outputs().to_vec(),
    );
    if let Some(n) = expr.name_hint() {
        lowered = lowered.with_name_hint(n);
    }
    Ok((
        lowered.build().map_err(|e| invalid(e.to_string()))?,
        witnesses,
    ))
}

/// Eliminate calls crossing a bounded signature before threading runtime
/// length operands. `inline` substitutes the callee's scoped symbol IDs.
pub(super) fn inline_calls(
    expr: Expr<'_>,
) -> io::Result<Option<std::sync::Arc<gnx_expr::RootExpr>>> {
    fn has_bounded(e: Expr<'_>) -> bool {
        e.closure_inputs().chain(e.inputs()).any(|p| {
            p.info_ref()
                .downcast::<ArrayInfoData>()
                .is_some_and(|i| !gnx_expr::Info::symbols(i).is_empty())
        })
    }
    fn needs(e: Expr<'_>) -> bool {
        e.eqns().any(|eq| {
            eq.subexprs()
                .any(|r| has_bounded(r.expr()) || needs(r.expr()))
        })
    }
    if !needs(expr) {
        return Ok(None);
    }
    let body = gnx_base::pass::rewrite(expr, &mut |b, eq| {
        for r in eq.subexprs() {
            let body = gnx_transforms::util::region_body(r).map_err(io::Error::other)?;
            if has_bounded(body.view()) {
                if eq
                    .op()
                    .downcast_ref::<gnx_transforms::cf::CallOperation>()
                    .is_none()
                {
                    return Err(io::Error::new(
                        io::ErrorKind::Unsupported,
                        "bounded axes in control-flow regions",
                    ));
                }
                return b
                    .inline(
                        body.view(),
                        r.provided_symbols(),
                        r.provided_vars(),
                        eq.inputs(),
                    )
                    .map(Some)
                    .map_err(io::Error::other);
            }
        }
        Ok(None)
    })?;
    Ok(Some(body.build().map_err(io::Error::other)?))
}

/// Prove that evaluating to the bound cannot move padding into valid lanes.
/// Extend this with op-specific rules before admitting shape-changing ops.
pub(super) fn check(expr: Expr<'_>) -> io::Result<()> {
    let mut bounds = HashMap::new();
    let mut symbolic = HashMap::new();
    for p in expr
        .closure_inputs()
        .chain(expr.inputs())
        .chain(expr.eqns().flat_map(|e| e.outputs()))
    {
        if p.info().symbols().is_empty() {
            continue;
        }
        if let Some(info) = p.info_ref().downcast::<ArrayInfoData>() {
            symbolic.insert(p.var(), info);
            for &d in info.shape().dims() {
                if let Dim::Bounded { symbol, max } = d
                    && bounds
                        .insert(symbol, max)
                        .is_some_and(|previous| previous != max)
                {
                    return Err(invalid(format!("conflicting bounds for {symbol}")));
                }
            }
        }
    }
    for eq in expr.eqns() {
        for child in eq.subexprs() {
            check(child.expr())?;
        }
        if bounds.is_empty() {
            continue;
        }
        let vars = eq
            .inputs()
            .iter()
            .copied()
            .chain(eq.outputs().map(|p| p.var()));
        let Some(formal) = vars.clone().find_map(|v| symbolic.get(&v)) else {
            continue;
        };
        let shape = formal.shape();
        let op = eq.op();
        if op
            .downcast_ref::<crate::ops::bounded::PadToBoundOperation>()
            .is_some()
            && eq
                .outputs()
                .next()
                .unwrap()
                .info_ref()
                .downcast::<ArrayInfoData>()
                .is_some_and(|i| i.to_native() == formal.to_native().capacity())
        {
            continue;
        }
        if let Some(op) = op.downcast_ref::<crate::ops::bounded::AxisLengthOperation>()
            && matches!(formal.shape().dims().get(op.axis()),Some(Dim::Bounded{max,..}) if i32::try_from(*max).is_ok())
            && eq
                .outputs()
                .next()
                .unwrap()
                .info_ref()
                .downcast::<ArrayInfoData>()
                .is_some_and(|i| {
                    i.to_native() == ArrayInfo::new(Shape::fixed([]), crate::DType::I32)
                })
        {
            continue;
        }
        if op.dialect() != "array"
            || !matches!(
                op.name(),
                "add"
                    | "sub"
                    | "mul"
                    | "neg"
                    | "abs"
                    | "exp"
                    | "min"
                    | "max"
                    | "lt"
                    | "le"
                    | "eq"
                    | "select"
                    | "convert_element_type"
            )
            || !vars
                .into_iter()
                .all(|v| symbolic.get(&v).is_some_and(|i| i.shape() == shape))
        {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!("{op}: no bounded-axis lowering for this signature"),
            ));
        }
    }
    Ok(())
}

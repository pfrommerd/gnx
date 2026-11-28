use super::{invalid, unsupported};
use gnx_base::ops::reduction::ReduceAddOperation;
use gnx_expr::{Eqn, Expr, ExprBuilder, VarInfoRef};
use gnx_math::{
    ArrayInfo, ArrayInfoData, DType, Dim,
    ops::{
        BroadcastInDimOperation, OnesOperation, ReshapeOperation, TransposeOperation,
        ZerosOperation,
    },
};
use std::{collections::HashMap, io};
use tch::{CModule, Kind, Tensor};

thread_local! { static TRACE_DEVICE: std::cell::Cell<tch::Device> = const { std::cell::Cell::new(tch::Device::Cpu) }; }

/// Device selected for tensors created by a lowering, including nullary ops.
/// Like LibTorch's tracer, the scope is synchronous and thread-local.
pub fn tracing_device() -> tch::Device {
    TRACE_DEVICE.get()
}

struct TraceScope(tch::Device);
impl Drop for TraceScope {
    fn drop(&mut self) {
        TRACE_DEVICE.set(self.0);
    }
}

/// Emit traceable, pure tensor operations. A rule may inspect static metadata,
/// but must not read operand data or mutate its inputs.
pub trait TorchLowering {
    fn lower_torch(&self, eqn: Eqn<'_>, inputs: &[Tensor]) -> io::Result<Vec<Tensor>>;

    /// Lower an equation with child regions. Each slice holds one region's
    /// captures in closure order; explicit operands are supplied separately.
    fn lower_torch_regions(
        &self,
        eqn: Eqn<'_>,
        inputs: &[Tensor],
        _captures: &[Vec<Tensor>],
    ) -> io::Result<Vec<Tensor>> {
        self.lower_torch(eqn, inputs)
    }
}

#[macro_export]
macro_rules! register_torch_lowering {
    ($($op:ty),+ $(,)?) => {
        $crate::__gnx_expr::register_op_interface!(dyn $crate::TorchLowering: $($op),+);
    };
}

pub(super) fn metadata(info: VarInfoRef<'_>) -> io::Result<ArrayInfo> {
    info.downcast::<ArrayInfoData>()
        .map(ArrayInfoData::to_native)
        .ok_or_else(|| unsupported("Torch requires array values"))
}

pub(super) fn layout(info: &ArrayInfo) -> io::Result<(Vec<i64>, Kind)> {
    let dims = info
        .shape()
        .dims()
        .iter()
        .map(|d| match d {
            Dim::Fixed(n) => i64::try_from(*n).map_err(invalid),
            Dim::Bounded { max, .. } => Ok(i64::from(*max)),
            _ => Err(unsupported("Torch requires fixed dimensions")),
        })
        .collect::<io::Result<_>>()?;
    let kind = match info.dtype() {
        DType::Bool => Kind::Bool,
        DType::F32 => Kind::Float,
        DType::F64 => Kind::Double,
        DType::F16 => Kind::Half,
        DType::BF16 => Kind::BFloat16,
        DType::U8 => Kind::Uint8,
        DType::I8 => Kind::Int8,
        DType::I16 => Kind::Int16,
        DType::I32 | DType::U32 => Kind::Int,
        DType::I64 => Kind::Int64,
        DType::KeyThreeFry2x32 | DType::KeyPhilox4x32 => Kind::Int64,
        DType::C64 => Kind::ComplexFloat,
        DType::C128 => Kind::ComplexDouble,
        other => return Err(unsupported(format!("unsupported Torch dtype {other:?}"))),
    };
    Ok((dims, kind))
}

pub(super) fn check(t: &Tensor, info: &ArrayInfo, device: tch::Device) -> io::Result<()> {
    let (dims, kind) = layout(info)?;
    if t.device() != device || t.size() != dims || t.f_kind().map_err(io::Error::other)? != kind {
        return Err(invalid(format!(
            "Torch tensor does not match {device:?} array {info:?}"
        )));
    }
    Ok(())
}

fn emit(body: Expr<'_>, inputs: &[Tensor]) -> io::Result<Tensor> {
    let [output] = emit_results(body, inputs)?
        .try_into()
        .map_err(|_| invalid("expected one region result"))?;
    Ok(output)
}

fn emit_results(body: Expr<'_>, inputs: &[Tensor]) -> io::Result<Vec<Tensor>> {
    if inputs.len() != body.closure_inputs().len() + body.inputs().len() {
        return Err(invalid("region input arity mismatch"));
    }
    let mut values: HashMap<_, _> = body
        .closure_inputs()
        .chain(body.inputs())
        .zip(inputs)
        .map(|(p, t)| (p.var(), t.shallow_clone()))
        .collect();
    // tch has no uint32 Kind. Preserve its four physical bytes on transport
    // and structural ops; never send unsigned comparisons, etc. to signed kernels.
    let mut unsigned = std::collections::HashSet::new();
    for p in body.closure_inputs().chain(body.inputs()) {
        if metadata(p.info_ref())?.dtype() == DType::U32 {
            unsigned.insert(p.var());
        }
    }
    for (i, eqn) in body.eqns().enumerate() {
        let op = eqn.op();
        let rule = op
            .interface::<dyn TorchLowering>()
            .map_err(|e| unsupported(e.to_string()))?;
        let operands: Vec<_> = eqn
            .inputs()
            .iter()
            .map(|v| values[v].shallow_clone())
            .collect();
        let captures = eqn
            .subexprs()
            .map(|region| {
                region
                    .provided_vars()
                    .iter()
                    .map(|v| values[v].shallow_clone())
                    .collect()
            })
            .collect::<Vec<_>>();
        let u32_input = eqn.inputs().iter().any(|v| unsigned.contains(v));
        let u32_output = eqn.outputs().any(|p| {
            p.info_ref()
                .downcast::<ArrayInfoData>()
                .is_some_and(|a| a.dtype() == DType::U32)
        });
        let results = if (u32_input || u32_output)
            && op
                .get()
                .downcast_ref::<gnx_math::ops::dtype::ConvertElementTypeOperation>()
                .is_some()
        {
            let mut x = operands[0].shallow_clone();
            if u32_input {
                x = x
                    .f_to_kind(Kind::Int64)
                    .and_then(|x| x.f_bitwise_and(u32::MAX as i64))
                    .map_err(io::Error::other)?;
            }
            if u32_output {
                x = x.f_to_kind(Kind::Int64).map_err(io::Error::other)?;
            }
            vec![
                x.f_to_kind(result_layout(eqn)?.1)
                    .map_err(io::Error::other)?,
            ]
        } else {
            if (u32_input || u32_output)
                && !(op.get().dialect() == "array"
                    && matches!(
                        op.get().name(),
                        // Int32 addition preserves the same low 32 bits for U32 counters.
                        "add" | "zeros"
                            | "ones"
                            | "constant"
                            | "reshape"
                            | "transpose"
                            | "broadcast_in_dim"
                            | "concatenate"
                            | "slice"
                            | "bitcast_convert_type"
                    ))
            {
                return Err(unsupported(format!(
                    "{op} does not support Torch U32 operands"
                )));
            }
            rule.lower_torch_regions(eqn, &operands, &captures)
                .map_err(|e| io::Error::new(e.kind(), format!("equation {i} ({op}): {e}")))?
        };
        if results.len() != eqn.outputs().len() {
            return Err(invalid(format!("{op}: wrong result count")));
        }
        for (p, t) in eqn.outputs().zip(results) {
            let info = metadata(p.info_ref())?;
            if info.dtype() == DType::U32 {
                unsigned.insert(p.var());
            }
            check(&t, &info, tracing_device())?;
            values.insert(p.var(), t);
        }
    }
    Ok(body
        .outputs()
        .iter()
        .map(|var| values[var].shallow_clone())
        .collect())
}

pub(super) fn trace(expr: Expr<'_>, device: tch::Device, joint: bool) -> io::Result<Vec<CModule>> {
    let _scope = TraceScope(TRACE_DEVICE.replace(device));
    let examples = expr
        .closure_inputs()
        .chain(expr.inputs())
        .map(|p| {
            let (dims, kind) = layout(&metadata(p.info_ref())?)?;
            Tensor::f_zeros(&dims, (kind, tracing_device())).map_err(io::Error::other)
        })
        .collect::<io::Result<Vec<_>>>()?;
    // Export either individually callable output modules or one shared graph.
    let group_size = if joint {
        expr.outputs().len().max(1)
    } else {
        1
    };
    expr.outputs()
        .chunks(group_size)
        .map(|outputs| {
            let params = |p: gnx_expr::Param<'_>| (p.var(), p.info_ref().to_owned());
            let body = ExprBuilder::from_parts(
                expr.closure_symbols().to_vec(),
                expr.closure_inputs().map(params).collect(),
                expr.inputs().map(params).collect(),
                expr.eqns().map(|e| e.to_source()).collect(),
                outputs.to_vec(),
            )
            .build()
            .map_err(invalid)?;
            let body = gnx_base::pass::dce(body.view())
                .build()
                .map_err(invalid)?;
            let mut error = None;
            let module = CModule::create_by_tracing("Gnx", "forward", &examples, &mut |inputs| {
                match emit_results(body.view(), inputs) {
                    Ok(t) => t,
                    // Always return to tch so it can close the active trace.
                    Err(e) => {
                        error = Some(e);
                        vec![]
                    }
                }
            });
            if let Some(e) = error {
                return Err(e);
            }
            module.map_err(io::Error::other)
        })
        .collect()
}

fn result_layout(eqn: Eqn<'_>) -> io::Result<(Vec<i64>, Kind)> {
    let mut outputs = eqn.outputs();
    let (Some(output), None) = (outputs.next(), outputs.next()) else {
        return Err(invalid("expected one result"));
    };
    layout(&metadata(output.info_ref())?)
}
fn axes(a: &[usize]) -> io::Result<Vec<i64>> {
    a.iter()
        .map(|&n| i64::try_from(n).map_err(invalid))
        .collect()
}

// Built-in single-result rules share arity checking and native error handling.
macro_rules! rule {
    ($ty:ty, |$op:ident, $eqn:ident, [$($arg:ident),*]| $body:block) => {
        impl TorchLowering for $ty {
            fn lower_torch(&self, $eqn: Eqn<'_>, inputs: &[Tensor]) -> io::Result<Vec<Tensor>> {
                let $op = self;
                let [$($arg),*] = inputs else { return Err(invalid("incorrect Torch operand count")) };
                Ok(vec![$body.map_err(io::Error::other)?])
            }
        }
        crate::register_torch_lowering!($ty);
    }
}
mod beta;
mod bitwise;
mod contraction;
mod control_flow;
pub use control_flow::{trace_if, trace_while};
mod convolution;
mod elementwise;
mod igamma;
mod indexing;
mod linalg;
mod math;
mod ordering;
mod precision;
mod quadrature;
mod reduction;
mod schur;
mod structural;
mod tridiagonal_solve;
rule!(ZerosOperation, |_op, eqn, []| {
    let (dims, k) = result_layout(eqn)?;
    Tensor::f_zeros(dims, (k, tracing_device()))
});
rule!(OnesOperation, |_op, eqn, []| {
    let (dims, k) = result_layout(eqn)?;
    Tensor::f_ones(dims, (k, tracing_device()))
});
rule!(ReshapeOperation, |_op, eqn, [x]| {
    x.f_reshape(result_layout(eqn)?.0)
});
rule!(TransposeOperation, |op, _eqn, [x]| {
    x.f_permute(axes(&op.perm())?)
});
rule!(BroadcastInDimOperation, |op, eqn, [x]| {
    let (target, _) = result_layout(eqn)?;
    let mut shape = vec![1; target.len()];
    let source = x.size();
    let dims = op.dims();
    if dims.len() != source.len()
        || dims.iter().any(|&d| d >= shape.len())
        || dims.windows(2).any(|w| w[0] >= w[1])
    {
        return Err(invalid("invalid broadcast dimensions"));
    }
    for (&d, s) in dims.iter().zip(source) {
        shape[d] = s;
    }
    x.f_reshape(shape).and_then(|t| t.f_expand(target, false))
});
rule!(ReduceAddOperation, |op, _eqn, [x]| {
    let dims = axes(&op.axes())?;
    if dims.is_empty() {
        return Ok(vec![x.shallow_clone()]);
    }
    x.f_sum_dim_intlist(dims.as_slice(), false, x.kind())
});

/// Serialize a traced module, in the format `CModule::load_data` reads.
pub(super) fn save(module: &CModule) -> io::Result<Vec<u8>> {
    use std::ffi::{c_char, c_void};
    unsafe extern "C" {
        fn gnx_torch_save_module(
            module: *const c_void,
            sink: unsafe extern "C" fn(*mut c_void, *const c_char, usize),
            ctx: *mut c_void,
            error: *mut c_char,
            capacity: usize,
        ) -> bool;
    }
    unsafe extern "C" fn sink(ctx: *mut c_void, data: *const c_char, len: usize) {
        // SAFETY: ctx is the Vec below, borrowed for this call; data points
        // to len bytes that live until this function returns.
        unsafe {
            let bytes = &mut *ctx.cast::<Vec<u8>>();
            bytes.extend_from_slice(std::slice::from_raw_parts(data.cast(), len));
        }
    }
    // tch keeps the native module pointer private. The crate is pinned to a
    // version where a CModule is exactly that pointer.
    const _: () = assert!(size_of::<CModule>() == size_of::<*const c_void>());
    // SAFETY: see above; the pointer is only read.
    let native = unsafe { std::ptr::from_ref(module).cast::<*const c_void>().read() };
    let mut bytes = Vec::new();
    let mut error = [0 as c_char; 1024];
    // SAFETY: the module outlives the call; native writes a NUL-terminated
    // diagnostic on failure and catches C++ exceptions.
    let saved = unsafe {
        gnx_torch_save_module(
            native,
            sink,
            std::ptr::from_mut(&mut bytes).cast(),
            error.as_mut_ptr(),
            error.len(),
        )
    };
    if !saved {
        let message = unsafe { std::ffi::CStr::from_ptr(error.as_ptr()) };
        return Err(io::Error::other(message.to_string_lossy().into_owned()));
    }
    Ok(bytes)
}

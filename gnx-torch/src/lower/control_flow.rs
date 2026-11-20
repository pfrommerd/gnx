use super::*;
use gnx_transforms::cf::{CondOperation, SwitchOperation, WhileOperation};
use std::ffi::{CStr, c_char, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};

type Emit = unsafe extern "C" fn(
    *mut c_void,
    usize,
    usize,
    *const *const c_void,
    *mut *const c_void,
) -> bool;
unsafe extern "C" {
    fn gnx_torch_trace_control(
        is_loop: bool,
        pred: *const c_void,
        count: usize,
        carry: *const *const c_void,
        outputs: *const *mut c_void,
        emit: Emit,
        ctx: *mut c_void,
        error: *mut c_char,
        capacity: usize,
    ) -> bool;
}
type LowerBlock<'a> = dyn FnMut(usize, &[Tensor]) -> io::Result<Vec<Tensor>> + 'a;
struct Callback<'a> {
    lower: &'a mut LowerBlock<'a>,
    count: usize,
    values: Vec<Tensor>,
    error: Option<io::Error>,
}
unsafe extern "C" fn emit_block(
    ctx: *mut c_void,
    i: usize,
    n: usize,
    params: *const *const c_void,
    out: *mut *const c_void,
) -> bool {
    // SAFETY: trace_control passes an exclusively borrowed Callback for the
    // lifetime of this synchronous call. C++ never retains its context.
    let state = unsafe { &mut *ctx.cast::<Callback<'_>>() };
    let result = catch_unwind(AssertUnwindSafe(|| {
        let args = (0..n)
            .map(|j| {
                // SAFETY: the bridge lends n live at::Tensor pointers; the new
                // shallow Rust handles do not take ownership of the C++ objects.
                unsafe { Tensor::clone_from_ptr((*params.add(j)).cast_mut().cast()) }
            })
            .collect::<Vec<_>>();
        state.values = (state.lower)(i, &args)?;
        if state.values.len() != state.count {
            return Err(invalid("control-flow block result count mismatch"));
        }
        for (j, v) in state.values.iter().enumerate() {
            // SAFETY: native allocated count slots. state.values owns the
            // handles through registration of this block's outputs.
            unsafe { out.add(j).write(v.as_ptr().cast()) };
        }
        Ok(())
    }));
    match result {
        Ok(Ok(())) => true,
        Ok(Err(e)) => {
            state.error = Some(e);
            false
        }
        Err(_) => {
            state.error = Some(io::Error::other("panic in Torch block lowering"));
            false
        }
    }
}
fn bool_scalar(t: &Tensor) -> io::Result<()> {
    if t.dim() != 0 || t.kind() != Kind::Bool {
        return Err(invalid("control flow requires a scalar boolean predicate"));
    }
    Ok(())
}
fn trace_control(
    pred: &Tensor,
    count: usize,
    carry: Option<&[Tensor]>,
    mut lower: impl FnMut(usize, &[Tensor]) -> io::Result<Vec<Tensor>>,
) -> io::Result<Vec<Tensor>> {
    bool_scalar(pred)?;
    if carry.is_some_and(|c| c.len() != count) {
        return Err(invalid("loop carry count mismatch"));
    }
    let is_loop = carry.is_some();
    let carried: Vec<_> = carry
        .unwrap_or_default()
        .iter()
        .map(|t| t.as_ptr().cast())
        .collect();
    let mut callback = Callback {
        lower: &mut lower,
        count: count + usize::from(is_loop),
        values: vec![],
        error: None,
    };
    let mut outputs: Vec<_> = (0..count).map(|_| Tensor::new()).collect();
    let pointers: Vec<_> = outputs.iter_mut().map(|v| v.as_mut_ptr().cast()).collect();
    let mut message = [0 as c_char; 2048];
    // SAFETY: all handles and pointer arrays live for the entire call. C++
    // assigns into existing output tensors and catches all exceptions.
    let ok = unsafe {
        gnx_torch_trace_control(
            is_loop,
            pred.as_ptr().cast(),
            count,
            carried.as_ptr(),
            pointers.as_ptr(),
            emit_block,
            (&mut callback as *mut Callback<'_>).cast(),
            message.as_mut_ptr(),
            message.len(),
        )
    };
    if let Some(e) = callback.error {
        return Err(e);
    }
    if !ok {
        // SAFETY: native writes a NUL-terminated diagnostic on failure.
        return Err(io::Error::other(
            unsafe { CStr::from_ptr(message.as_ptr()) }
                .to_string_lossy()
                .into_owned(),
        ));
    }
    Ok(outputs)
}
/// Trace a TorchScript conditional during a lowering. Both closures are traced;
/// on replay only the branch selected by the scalar boolean `pred` runs. Each
/// branch must return `output_count` tensors with matching output metadata.
pub fn trace_if(
    pred: &Tensor,
    output_count: usize,
    mut then_fn: impl FnMut() -> io::Result<Vec<Tensor>>,
    mut else_fn: impl FnMut() -> io::Result<Vec<Tensor>>,
) -> io::Result<Vec<Tensor>> {
    trace_control(pred, output_count, None, |branch, _| match branch {
        0 => then_fn(),
        1 => else_fn(),
        _ => Err(invalid("unexpected conditional branch index")),
    })
}

/// Trace a TorchScript while loop during a lowering. `cond_fn` returns a scalar
/// boolean; `body_fn` returns the next carried tensors in the order of `initial`,
/// preserving their metadata. On replay a false initial condition skips the body.
pub fn trace_while(
    initial: &[Tensor],
    mut cond_fn: impl FnMut(&[Tensor]) -> io::Result<Tensor>,
    mut body_fn: impl FnMut(&[Tensor]) -> io::Result<Vec<Tensor>>,
) -> io::Result<Vec<Tensor>> {
    let first = cond_fn(initial)?;
    trace_control(&first, initial.len(), Some(initial), |_, carry| {
        let mut next = body_fn(carry)?;
        if next.len() != initial.len() {
            return Err(invalid(
                "loop body must return one tensor per carried value",
            ));
        }
        let pred = cond_fn(&next)?;
        bool_scalar(&pred)?;
        next.insert(0, pred); // TorchScript block outputs: continuation, then carries.
        Ok(next)
    })
}

struct Regions<'a> {
    eqn: &'a Eqn,
    captures: &'a [Vec<Tensor>],
}
impl Regions<'_> {
    fn emit(&self, index: usize, args: &[Tensor]) -> io::Result<Vec<Tensor>> {
        let params = self.captures[index]
            .iter()
            .chain(args)
            .map(Tensor::shallow_clone)
            .collect::<Vec<_>>();
        emit_results(self.eqn.subexprs()[index].expr(), &params)
    }
    fn count(&self) -> usize {
        self.eqn.outputs().len()
    }
    fn cond(&self, inputs: &[Tensor]) -> io::Result<Vec<Tensor>> {
        let [pred, args @ ..] = inputs else {
            return Err(invalid("missing cond predicate"));
        };
        trace_if(
            pred,
            self.count(),
            || self.emit(0, args),
            || self.emit(1, args),
        )
    }
    fn switch(&self, inputs: &[Tensor]) -> io::Result<Vec<Tensor>> {
        let [index, args @ ..] = inputs else {
            return Err(invalid("missing switch index"));
        };
        if index.dim() != 0 || index.kind() != Kind::Int {
            return Err(invalid("switch requires a scalar i32 selector"));
        }
        self.switch_range(index, args, 0, self.captures.len())
    }
    // A balanced tree gives logarithmic runtime dispatch. Out-of-range
    // selectors naturally follow the path to the first or last leaf.
    fn switch_range(
        &self,
        index: &Tensor,
        args: &[Tensor],
        lo: usize,
        hi: usize,
    ) -> io::Result<Vec<Tensor>> {
        if hi - lo == 1 {
            return self.emit(lo, args);
        }
        let mid = lo + (hi - lo) / 2;
        let pred = index
            .f_lt(i64::try_from(mid).map_err(invalid)?)
            .map_err(io::Error::other)?;
        trace_if(
            &pred,
            self.count(),
            || self.switch_range(index, args, lo, mid),
            || self.switch_range(index, args, mid, hi),
        )
    }
    fn condition(&self, carry: &[Tensor]) -> io::Result<Tensor> {
        let [pred] = self
            .emit(0, carry)?
            .try_into()
            .map_err(|_| invalid("expected one loop predicate"))?;
        Ok(pred)
    }
    fn while_loop(&self, inputs: &[Tensor]) -> io::Result<Vec<Tensor>> {
        if inputs.len() != self.count() {
            return Err(invalid("loop carry count mismatch"));
        }
        trace_while(
            inputs,
            |carry| self.condition(carry),
            |carry| self.emit(1, carry),
        )
    }
}
macro_rules! region_op {
    ($ty:ty,$method:ident) => {
        impl TorchLowering for $ty {
            fn lower_torch(&self, e: &Eqn, x: &[Tensor]) -> io::Result<Vec<Tensor>> {
                self.lower_torch_regions(
                    e,
                    x,
                    &(0..e.subexprs().len())
                        .map(|_| Vec::new())
                        .collect::<Vec<_>>(),
                )
            }
            fn lower_torch_regions(
                &self,
                e: &Eqn,
                x: &[Tensor],
                captures: &[Vec<Tensor>],
            ) -> io::Result<Vec<Tensor>> {
                Regions { eqn: e, captures }.$method(x)
            }
        }
        crate::register_torch_lowering!($ty);
    };
}
region_op!(CondOperation, cond);
region_op!(SwitchOperation, switch);
region_op!(WhileOperation, while_loop);

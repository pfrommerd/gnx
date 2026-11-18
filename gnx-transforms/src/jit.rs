use crate::{Callable, Jitable};

pub use gnx_derive::{jit, jit_fn, jittable};

use gnx_expr::{
    Capture, CellKey, CellTrace, InputSpec, ObjectTracer, Ownership, Trace, TraceContext, TracerKey,
};
use gnx_graph::util::GraphExt;
use gnx_graph::{AsSource, Builder, Graph, GraphContext, Of, SourceError};
use std::collections::BTreeMap;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Identity {
    Value(TracerKey),
    Cell(CellKey),
}

impl Identity {
    fn of(object: &ObjectTracer) -> Self {
        match object {
            ObjectTracer::Ref(value) => Self::Value(TracerKey::from(value)),
            ObjectTracer::Cell(cell) => Self::Cell(CellKey::from(cell)),
        }
    }
}

/// Callable wrapper used by the function attribute and closure macro.
///
/// Each invocation traces the Rust body once with alias-preserving placeholders,
/// captures and lowers its HExpr, then emits an immutable call in the caller's
/// context. Boundary descriptors reconstruct results and apply cell updates;
/// static graph data is preserved. No specialization cache or backend compilation
/// is performed yet. Non-traced Rust side effects run during capture.
/// Capture, lowering, call, or graph reconstruction errors currently panic.
///
/// ```
/// #[gnx_transforms::jit]
/// fn increment(value: u32) -> u32 { value + 1 }
/// let offset = 2;
/// let callable = gnx_transforms::jit_fn!(|value: u32| increment(value) + offset);
/// assert_eq!(callable(3), 6);
/// ```
///
/// The attribute currently accepts no options:
/// ```compile_fail
/// #[gnx_transforms::jit(unsupported)]
/// fn example(value: u32) -> u32 { value }
/// ```
pub struct Jit<F> {
    func: F,
}

impl<I: Graph, F: Jitable<I>> Callable<I> for Jit<F> {
    type Output = <F as Callable<I>>::Output;
    fn invoke(&self, input: I) -> Self::Output {
        let filter = &Of::<ObjectTracer>::filter();
        let guard = TraceContext::enter();
        let mut identities = BTreeMap::new();
        let mut arguments = Vec::new();
        let mut placeholders = Vec::new();
        let placeholder_graph = input.to_dag(filter, |view| {
            let object = view.to_owned();
            let index = *identities.entry(Identity::of(&object)).or_insert_with(|| {
                let placeholder = match &object {
                    ObjectTracer::Ref(value) => {
                        ObjectTracer::Ref(Trace::placeholder(value.info().clone()))
                    }
                    ObjectTracer::Cell(cell) => {
                        ObjectTracer::Cell(CellTrace::placeholder(cell.type_info().clone()))
                    }
                };
                let index = arguments.len();
                arguments.push(object);
                placeholders.push(placeholder);
                index
            });
            placeholders[index].clone()
        });
        let owned_input = input
            .replace(
                filter,
                AsSource::new(&placeholder_graph, filter),
                &mut GraphContext::new(),
            )
            .expect("JIT could not reconstruct placeholder arguments");
        let output = self.func.invoke_owned(owned_input);
        let mut outputs = Vec::new();
        let mut output_indices = BTreeMap::new();
        output.to_dag(filter, |view| {
            let object = view.to_owned();
            output_indices
                .entry(Identity::of(&object))
                .or_insert_with(|| {
                    let index = outputs.len();
                    outputs.push(object);
                    index
                });
        });
        let builder = output
            .builder::<ObjectTracer, _, SourceError>(filter, &mut GraphContext::new())
            .expect("JIT could not build output graph");
        let capture = Capture::from_context(guard.context(), &placeholders, &outputs)
            .expect("JIT could not capture traced function");
        // Emit the call and cell boundary in the caller's context, not capture's.
        drop(guard);
        let closure_bindings = capture
            .closure()
            .iter()
            .cloned()
            .map(|object| (object, Ownership::Borrow))
            .collect::<Vec<_>>();
        let input_bindings = arguments
            .iter()
            .cloned()
            .map(|object| (object, Ownership::Borrow))
            .collect::<Vec<_>>();
        let spec = InputSpec::from_args(&closure_bindings, &input_bindings);
        let lowered = capture
            .expr()
            .lower(&spec)
            .expect("JIT could not lower captured function");
        let results = lowered
            .call(capture.closure(), &arguments)
            .expect("JIT could not call captured function");
        let result_graph = output.to_dag(filter, |view| {
            results[output_indices[&Identity::of(&view.to_owned())]].clone()
        });
        Builder::<ObjectTracer>::build(
            builder,
            AsSource::new(&result_graph, filter),
            &mut GraphContext::new(),
        )
        .expect("JIT could not reconstruct function results")
    }
}

pub fn jit<I: Graph, F: Jitable<I>>(func: F) -> Jit<F> {
    Jit { func }
}

use proc_macro::TokenStream as ProcTokenStream;

mod dag;
mod graph;
mod graph_traits;
mod leaf;
mod lifetime_free;
mod paths;
mod transform_fn;
mod transforms;

#[proc_macro]
pub fn impl_leaf(input: ProcTokenStream) -> ProcTokenStream {
    leaf::impl_leaf(input)
}

#[proc_macro]
pub fn impl_lifetime_free(input: ProcTokenStream) -> ProcTokenStream {
    lifetime_free::impl_lifetime_free(input)
}

/// Wrap an infallible function or closure in a JIT that returns its output directly.
#[proc_macro]
pub fn jit_fn(input: ProcTokenStream) -> ProcTokenStream {
    transforms::jit_fn(input, false)
}

/// Wrap a function or closure in a JIT that always returns a `Result`.
#[proc_macro]
pub fn try_jit_fn(input: ProcTokenStream) -> ProcTokenStream {
    transforms::jit_fn(input, true)
}

#[proc_macro]
pub fn jittable(input: ProcTokenStream) -> ProcTokenStream {
    transforms::jittable(input)
}

/// Construct a DAG. In `dag!(as T; ...)`, convert each payload into `T`;
/// keys, shared bindings, and empty leaves (`*`) keep their usual meaning.
#[proc_macro]
pub fn dag(input: ProcTokenStream) -> ProcTokenStream {
    dag::dag(input)
}

#[proc_macro_attribute]
pub fn jit(attr: ProcTokenStream, input: ProcTokenStream) -> ProcTokenStream {
    transforms::jit(attr, input)
}

#[proc_macro_derive(Leaf)]
pub fn derive_leaf(input: ProcTokenStream) -> ProcTokenStream {
    leaf::derive_leaf(input)
}

#[proc_macro_derive(LifetimeFree)]
pub fn derive_lifetime_free(input: ProcTokenStream) -> ProcTokenStream {
    lifetime_free::derive_lifetime_free(input)
}

/// Derive named-field graph traversal, reconstruction, hashing, and equality
/// for a concrete struct of owned graph values.
#[proc_macro_derive(Graph)]
pub fn derive_graph(input: ProcTokenStream) -> ProcTokenStream {
    graph::derive(input)
}

#[proc_macro_derive(GraphHash)]
pub fn derive_graph_hash(input: ProcTokenStream) -> ProcTokenStream {
    graph_traits::derive_graph_hash(input)
}

#[proc_macro_derive(GraphPartialEq)]
pub fn derive_graph_partial_eq(input: ProcTokenStream) -> ProcTokenStream {
    graph_traits::derive_graph_partial_eq(input)
}

/// Derive builder equality for a graph backed by `LeafBuilder`.
/// Graphs with other builders should implement `GraphEq` directly.
#[proc_macro_derive(GraphEq)]
pub fn derive_graph_eq(input: ProcTokenStream) -> ProcTokenStream {
    graph_traits::derive_graph_eq(input)
}

/// Vectorize a closure like `jit_fn!`: `map!(|x: Array, w: Array| .., in_axes = (0, *))`.
/// Options (all optional, `dag!` syntax): `in_axes`, `out_axes`; and `axis_size = <expr>`.
#[proc_macro]
pub fn map(input: ProcTokenStream) -> ProcTokenStream {
    transform_fn::transform_fn(input, &transform_fn::MAP)
}

/// Forward-mode derivative of a closure, called with `(primals, tangents)`.
/// Selector: `wrt = filter`, `wrt_arg = index`, or `wrt_args = indices`.
#[proc_macro]
pub fn jvp(input: ProcTokenStream) -> ProcTokenStream {
    transform_fn::transform_fn(input, &transform_fn::JVP)
}

/// Reverse-mode derivative of a closure, called with `(primals, cotangents)`.
/// Selector: `wrt = filter`, `wrt_arg = index`, or `wrt_args = indices`.
#[proc_macro]
pub fn vjp(input: ProcTokenStream) -> ProcTokenStream {
    transform_fn::transform_fn(input, &transform_fn::VJP)
}

/// Gradient of a scalar closure. Selector: `wrt = filter`, `wrt_arg = index`, or `wrt_args = indices`.
#[proc_macro]
pub fn grad(input: ProcTokenStream) -> ProcTokenStream {
    transform_fn::transform_fn(input, &transform_fn::GRAD)
}

/// Value and gradient of a scalar closure. Selector: `wrt = filter`, `wrt_arg = index`, or `wrt_args = indices`.
#[proc_macro]
pub fn value_and_grad(input: ProcTokenStream) -> ProcTokenStream {
    transform_fn::transform_fn(input, &transform_fn::VALUE_AND_GRAD)
}

/// Hessian of a scalar closure with respect to one array argument.
/// Selector: `wrt = filter`, `wrt_arg = index`, or `wrt_args = indices`.
#[proc_macro]
pub fn hessian(input: ProcTokenStream) -> ProcTokenStream {
    transform_fn::transform_fn(input, &transform_fn::HESSIAN)
}

use proc_macro::TokenStream as ProcTokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput};

mod leaf;
mod lifetime_free;
mod paths;
mod transforms;

#[proc_macro]
pub fn impl_leaf(input: ProcTokenStream) -> ProcTokenStream {
    leaf::impl_leaf(input)
}

#[proc_macro]
pub fn impl_lifetime_free(input: ProcTokenStream) -> ProcTokenStream {
    lifetime_free::impl_lifetime_free(input)
}

#[proc_macro]
pub fn jit(input: ProcTokenStream) -> ProcTokenStream {
    transforms::jit(input)
}

#[proc_macro_attribute]
pub fn transform(attr: ProcTokenStream, input: ProcTokenStream) -> ProcTokenStream {
    match attr.to_string().as_str() {
        "jit" => transforms::jit_fn(attr, input),
        _ => panic!("Invalid transform"),
    }
}

#[proc_macro_derive(Leaf)]
pub fn derive_leaf(input: ProcTokenStream) -> ProcTokenStream {
    leaf::derive_leaf(input)
}

#[proc_macro_derive(LifetimeFree)]
pub fn derive_lifetime_free(input: ProcTokenStream) -> ProcTokenStream {
    lifetime_free::derive_lifetime_free(input)
}

#[proc_macro_derive(Graph)]
pub fn derive_graph(input: ProcTokenStream) -> ProcTokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let g = paths::graph_crate();
    let err = paths::require_graph_crate();
    let expanded = quote! {
        #err
        impl #g::Graph for #name {
        }
    };
    ProcTokenStream::from(expanded)
}

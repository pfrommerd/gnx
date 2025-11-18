use proc_macro::TokenStream as ProcTokenStream;
use quote::quote;
use syn::{DeriveInput, parse_macro_input};

mod transforms;

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
    // Parse the input into a syntax tree
    let input = parse_macro_input!(input as DeriveInput);
    // let ty = Type::Path(TypePath {
    //     qself: None,
    //     path: Path::from(input.ident),
    // });
    ProcTokenStream::from(quote! {})
}

#[proc_macro_derive(Graph)]
pub fn derive_graph(input: ProcTokenStream) -> ProcTokenStream {
    // Parse the input into a syntax tree
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    // Generate code
    let expanded = quote! {
        impl ::gnx::graph::Graph for #name {
        }
    };
    ProcTokenStream::from(expanded)
}
use proc_macro::TokenStream as ProcTokenStream;
use proc_macro2::TokenStream;
use quote::quote;
use syn::{
    Block, Ident, Token, Type,
    parse::{Parse, ParseStream, Result},
};
use syn::{DeriveInput, parse_macro_input};

fn impl_leaf_graph(name: Ident) -> TokenStream {
    quote! {
        impl ::gnx::graph::Graph for #name {
            type GraphDef<I> = ::gnx::graph::LeafDef<I>;

        }
    }
}

#[proc_macro_derive(Leaf)]
pub fn derive_leaf(input: ProcTokenStream) -> ProcTokenStream {
    // Parse the input into a syntax tree
    let input = parse_macro_input!(input as DeriveInput);
    ProcTokenStream::from(impl_leaf_graph(input.ident))
}

#[proc_macro]
pub fn impl_leaf(input: ProcTokenStream) -> ProcTokenStream {
    let name = parse_macro_input!(input as Ident);
    ProcTokenStream::from(impl_leaf_graph(name))
}

struct UnionVariant {
    name: Ident,
    ty: Option<Type>,
}
struct LeafUnionInput {
    variants: Vec<UnionVariant>,
}

#[proc_macro_derive(LeafUnion)]
pub fn derive_leaf_union(input: ProcTokenStream) -> ProcTokenStream {
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

struct GraphStruct {}

#[proc_macro_derive(Graph)]
pub fn derive_owned_graph(input: ProcTokenStream) -> ProcTokenStream {
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

struct BuildSharedArgs {
    ctx: Ident,
    _comma1: Token![,],
    tp: Type,
    _comma2: Token![,],
    id: Ident,
    _comma3: Token![,],
    body: Block,
}

impl Parse for BuildSharedArgs {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(BuildSharedArgs {
            ctx: input.parse()?,
            _comma1: input.parse()?,
            tp: input.parse()?,
            _comma2: input.parse()?,
            id: input.parse()?,
            _comma3: input.parse()?,
            body: input.parse()?,
        })
    }
}

#[proc_macro]
pub fn ctx_build_shared(input: ProcTokenStream) -> ProcTokenStream {
    let input = parse_macro_input!(input as BuildSharedArgs);
    let (ctx, tp, id, body) = (input.ctx, input.tp, input.id, input.body);
    let expanded = quote! {
        {
            let res = #ctx._reserve::<#tp>(#id);
            match res {
                Err(e) => Err(e.into()),
                Ok(opt) => match opt {
                    Some(v) => Ok(v),
                    None => {
                        let value: Result<#tp, _> = #body;
                        match value {
                            Ok(v) => {
                                let fv = #ctx._finish::<#tp>(#id, v.clone());
                                match fv {
                                    Ok(_) => Ok(v),
                                    Err(e) => Err(e.into()),
                                }
                            },
                            Err(e) => Err(e)
                        }
                    }
                }
            }
        }
    };
    ProcTokenStream::from(expanded)
}

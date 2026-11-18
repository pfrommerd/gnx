use proc_macro_crate::{FoundCrate, crate_name};
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::Path;

pub fn graph_crate() -> Path {
    match crate_name("gnx-graph") {
        Ok(FoundCrate::Itself) => syn::parse_quote!(crate),
        Ok(FoundCrate::Name(_)) => syn::parse_quote!(::gnx_graph),
        Err(_) => match crate_name("gnx") {
            Ok(FoundCrate::Itself) | Ok(FoundCrate::Name(_)) => syn::parse_quote!(::gnx::graph),
            Err(_) => syn::parse_quote!(::gnx_graph),
        },
    }
}

pub fn graph_util_mod() -> Path {
    match crate_name("gnx-graph") {
        Ok(FoundCrate::Itself) => syn::parse_quote!(crate::util),
        Ok(FoundCrate::Name(_)) => syn::parse_quote!(::gnx_graph::util),
        Err(_) => match crate_name("gnx") {
            Ok(FoundCrate::Itself) | Ok(FoundCrate::Name(_)) => syn::parse_quote!(::gnx::util),
            Err(_) => syn::parse_quote!(::gnx_graph::util),
        },
    }
}

pub fn cast() -> TokenStream2 {
    match crate_name("gnx-graph") {
        Ok(FoundCrate::Itself) => quote!(crate::cast),
        Ok(FoundCrate::Name(_)) => quote!(::gnx_graph::cast),
        Err(_) => match crate_name("gnx") {
            Ok(FoundCrate::Itself) | Ok(FoundCrate::Name(_)) => quote!(::gnx::util::cast),
            Err(_) => quote!(::gnx_graph::cast),
        },
    }
}

pub fn require_graph_crate() -> Option<TokenStream2> {
    if matches!(crate_name("gnx-graph"), Ok(_)) || matches!(crate_name("gnx"), Ok(_)) {
        None
    } else {
        Some(quote! {
            compile_error!("gnx-derive graph macros require `gnx-graph` or `gnx` as a direct dependency");
        })
    }
}

pub fn transforms_callable_as_fn() -> Path {
    match crate_name("gnx-transforms") {
        Ok(FoundCrate::Itself) => syn::parse_quote!(::gnx_transforms::callable::as_fn),
        Ok(FoundCrate::Name(_)) => syn::parse_quote!(::gnx_transforms::callable::as_fn),
        Err(_) => match crate_name("gnx") {
            Ok(_) => syn::parse_quote!(::gnx::callable::as_fn),
            Err(_) => syn::parse_quote!(::gnx_transforms::callable::as_fn),
        },
    }
}

pub fn transforms_jit_fn() -> Path {
    match crate_name("gnx-transforms") {
        Ok(FoundCrate::Itself) => syn::parse_quote!(::gnx_transforms::jit),
        Ok(FoundCrate::Name(_)) => syn::parse_quote!(::gnx_transforms::jit),
        Err(_) => match crate_name("gnx") {
            Ok(_) => syn::parse_quote!(::gnx::jit),
            Err(_) => syn::parse_quote!(::gnx_transforms::jit),
        },
    }
}

pub fn transforms_function_trait(name: &str) -> Path {
    let name = syn::Ident::new(name, proc_macro2::Span::call_site());
    match crate_name("gnx-transforms") {
        Err(_) if crate_name("gnx").is_ok() => syn::parse_quote!(::gnx::#name),
        _ => syn::parse_quote!(::gnx_transforms::#name),
    }
}

pub fn require_transforms_crate() -> Option<TokenStream2> {
    if matches!(crate_name("gnx-transforms"), Ok(_)) || matches!(crate_name("gnx"), Ok(_)) {
        None
    } else {
        Some(quote! {
            compile_error!("gnx-derive transform macros require `gnx-transforms` or `gnx` as a direct dependency");
        })
    }
}

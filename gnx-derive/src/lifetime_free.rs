use proc_macro::TokenStream as ProcTokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{
    DeriveInput, GenericParam, Token, Type, TypeParam,
    parse::Parse, parse_macro_input, punctuated::Punctuated, token::Comma
};

use crate::paths::{graph_util_mod, require_graph_crate};

/// Returns the first lifetime generic parameter on the type, if any.
fn find_lifetime_generic_param(input: &DeriveInput) -> Option<&syn::LifetimeParam> {
    input.generics.params.iter().find_map(|param| {
        if let GenericParam::Lifetime(lt) = param {
            Some(lt)
        } else {
            None
        }
    })
}

/// Add `T: LifetimeFree` for each type parameter on the type definition.
fn add_lifetime_free_bounds(generics: &mut syn::Generics, util: &syn::Path) {
    for param in generics.type_params_mut() {
        let bound: syn::TypeParamBound = syn::parse_quote!(#util::LifetimeFree);
        param.bounds.push(bound);
    }
}

fn impl_type_var_bounds(idents: &[TypeParam], util: &syn::Path) -> TokenStream2 {
    if idents.is_empty() {
        TokenStream2::new()
    } else {
        let bounds = idents.iter().map(|ident| {
            let mut ident = ident.clone();
            ident.bounds.push(syn::parse_quote!(#util::LifetimeFree));
            ident
        });
        quote!(<#(#bounds),*>)
    }
}

pub fn expand_lifetime_free_for_type(ty: &Type, type_var_idents: &[TypeParam]) -> TokenStream2 {
    let err = require_graph_crate();
    let util = graph_util_mod();
    let impl_bounds = impl_type_var_bounds(type_var_idents, &util);

    quote! {
        #err
        // SAFETY: The type has no lifetime parameters; type variables listed in `for<...>` are
        // required to be LifetimeFree.
        unsafe impl #impl_bounds #util::LifetimeFree for #ty {}
    }
}

pub fn expand_lifetime_free_from_derive(input: &DeriveInput) -> TokenStream2 {
    let err = require_graph_crate();
    if let Some(lt) = find_lifetime_generic_param(input) {
        return syn::Error::new_spanned(
            &lt.lifetime,
            "LifetimeFree cannot be derived for types with lifetime parameters",
        )
        .to_compile_error();
    }

    let util = graph_util_mod();
    let ident = &input.ident;

    let mut generics = input.generics.clone();
    add_lifetime_free_bounds(&mut generics, &util);
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    quote! {
        #err
        // SAFETY: No lifetime parameters on the type; each type parameter is required to be
        // LifetimeFree, so all instantiations are lifetime-free.
        unsafe impl #impl_generics #util::LifetimeFree for #ident #ty_generics #where_clause {}
    }
}

/// One entry in `impl_lifetime_free!(...)`: either a concrete type or `for<A, ...> Type`.
struct LifetimeFreeEntry {
    type_params: Vec<TypeParam>,
    ty: Type,
}

impl Parse for LifetimeFreeEntry {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        // `for` is a keyword; in proc-macro input `Token![for]` peek/parse is unreliable,
        // but an identifier comparison on the first token works.
        if input.peek(Token![for]) {
            input.parse::<Token![for]>()?;
            input.parse::<Token![<]>()?;
            let type_params = Punctuated::<TypeParam, Comma>::parse_separated_nonempty(input)?;
            input.parse::<Token![>]>()?;
            let ty = input.parse::<Type>()?;
            Ok(LifetimeFreeEntry {
                type_params: type_params.into_iter().collect(),
                ty,
            })
        } else {
            let ty = input.parse::<Type>()?;
            Ok(LifetimeFreeEntry {
                type_params: Vec::new(),
                ty,
            })
        }

    }
}

pub fn impl_lifetime_free(input: ProcTokenStream) -> ProcTokenStream {
    let entries =
        parse_macro_input!(input with Punctuated::<LifetimeFreeEntry, Comma>::parse_terminated);
    let expanded = entries
        .iter()
        .map(|entry| expand_lifetime_free_for_type(&entry.ty, &entry.type_params))
        .collect::<TokenStream2>();
    ProcTokenStream::from(expanded)
}

pub fn derive_lifetime_free(input: ProcTokenStream) -> ProcTokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    ProcTokenStream::from(expand_lifetime_free_from_derive(&input))
}

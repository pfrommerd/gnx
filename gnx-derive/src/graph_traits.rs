use proc_macro::TokenStream as ProcTokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{Data, DeriveInput, Fields, GenericParam, parse_macro_input};

use crate::paths::{graph_crate, require_graph_crate};

fn add_trait_bounds(generics: &mut syn::Generics, bound: TokenStream2) {
    for param in generics.params.iter_mut() {
        if let GenericParam::Type(param) = param {
            param.bounds.push(syn::parse_quote!(#bound));
        }
    }
}

fn field_hashes(fields: &Fields, self_expr: TokenStream2, g: &syn::Path) -> Vec<TokenStream2> {
    match fields {
        Fields::Named(fields) => fields
            .named
            .iter()
            .map(|field| {
                let ident = field.ident.as_ref().unwrap();
                let key = ident.to_string();
                quote! {
                    #g::GraphHash::graph_hash(
                        &#self_expr.#ident,
                        hasher,
                        filter.child(#g::KeyRef::from(#key)),
                        ctx,
                    );
                }
            })
            .collect(),
        Fields::Unnamed(fields) => fields
            .unnamed
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let index = syn::Index::from(i);
                quote! {
                    #g::GraphHash::graph_hash(
                        &#self_expr.#index,
                        hasher,
                        filter.child(#g::KeyRef::from(#i)),
                        ctx,
                    );
                }
            })
            .collect(),
        Fields::Unit => Vec::new(),
    }
}

fn field_eqs(
    fields: &Fields,
    left: TokenStream2,
    right: TokenStream2,
    g: &syn::Path,
) -> Vec<TokenStream2> {
    match fields {
        Fields::Named(fields) => fields
            .named
            .iter()
            .map(|field| {
                let ident = field.ident.as_ref().unwrap();
                let key = ident.to_string();
                quote! {
                    #g::GraphPartialEq::graph_partial_eq(
                        &#left.#ident,
                        &#right.#ident,
                        filter.child(#g::KeyRef::from(#key)),
                        ctx,
                    )
                }
            })
            .collect(),
        Fields::Unnamed(fields) => fields
            .unnamed
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let index = syn::Index::from(i);
                quote! {
                    #g::GraphPartialEq::graph_partial_eq(
                        &#left.#index,
                        &#right.#index,
                        filter.child(#g::KeyRef::from(#i)),
                        ctx,
                    )
                }
            })
            .collect(),
        Fields::Unit => Vec::new(),
    }
}

pub fn derive_graph_hash(input: ProcTokenStream) -> ProcTokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let err = require_graph_crate();
    let g = graph_crate();
    let name = &input.ident;

    let mut generics = input.generics.clone();
    add_trait_bounds(&mut generics, quote!(#g::GraphHash));
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let body = match &input.data {
        Data::Struct(data) => {
            let field_hashes = field_hashes(&data.fields, quote!(self), &g);
            quote! {
                ::std::hash::Hash::hash(&::core::any::type_name::<Self>(), hasher);
                #(#field_hashes)*
            }
        }
        Data::Enum(data) => {
            let arms = data.variants.iter().enumerate().map(|(i, variant)| {
                let variant_ident = &variant.ident;
                match &variant.fields {
                    Fields::Named(fields) => {
                        let bindings = fields.named.iter().map(|f| f.ident.as_ref().unwrap());
                        let tmp = format_ident!("__gnx_variant");
                        let field_hashes = fields.named.iter().map(|field| {
                            let ident = field.ident.as_ref().unwrap();
                            let key = ident.to_string();
                            quote! {
                                #g::GraphHash::graph_hash(
                                    #ident,
                                    hasher,
                                    filter.child(#g::KeyRef::Attr(#key)),
                                    ctx,
                                );
                            }
                        });
                        quote! {
                            Self::#variant_ident { #(#bindings,)* } => {
                                let #tmp = #i;
                                ::std::hash::Hash::hash(&#tmp, hasher);
                                #(#field_hashes)*
                            }
                        }
                    }
                    Fields::Unnamed(fields) => {
                        let bindings = (0..fields.unnamed.len())
                            .map(|i| format_ident!("__gnx_field_{i}"))
                            .collect::<Vec<_>>();
                        let field_hashes = bindings.iter().enumerate().map(|(i, ident)| {
                            quote! {
                                #g::GraphHash::graph_hash(
                                    #ident,
                                    hasher,
                                    filter.child(#g::KeyRef::Index(#i)),
                                    ctx,
                                );
                            }
                        });
                        quote! {
                            Self::#variant_ident(#(#bindings,)*) => {
                                ::std::hash::Hash::hash(&#i, hasher);
                                #(#field_hashes)*
                            }
                        }
                    }
                    Fields::Unit => {
                        quote! {
                            Self::#variant_ident => {
                                ::std::hash::Hash::hash(&#i, hasher);
                            }
                        }
                    }
                }
            });
            quote! {
                ::std::hash::Hash::hash(&::core::any::type_name::<Self>(), hasher);
                match self {
                    #(#arms)*
                }
            }
        }
        Data::Union(data) => {
            return syn::Error::new_spanned(
                &data.union_token,
                "GraphHash cannot be derived for unions",
            )
            .to_compile_error()
            .into();
        }
    };

    ProcTokenStream::from(quote! {
        #err
        impl #impl_generics #g::GraphHash for #name #ty_generics #where_clause {
            fn graph_hash<H, L, F>(
                &self,
                hasher: &mut H,
                filter: F,
                ctx: &mut #g::GraphHashContext,
            )
            where
                H: ::std::hash::Hasher,
                L: #g::Leaf,
                F: #g::Filter<L>,
            {
                if filter.matches_ref(self).is_ok() {
                    ::std::hash::Hash::hash(&#g::HashMarker::Excluded, hasher);
                    ::std::hash::Hash::hash(&::core::any::type_name::<Self>(), hasher);
                } else {
                    #body
                }
            }

            fn hash_builder<H, L>(
                builder: &Self::Builder<L>,
                hasher: &mut H,
                ctx: &mut #g::GraphHashContext,
            )
            where
                H: ::std::hash::Hasher,
                L: #g::Leaf,
            {
                let _ = (builder, ctx);
                ::std::hash::Hash::hash(&#g::HashMarker::Included, hasher);
                ::std::hash::Hash::hash(&::core::any::type_name::<Self>(), hasher);
            }
        }
    })
}

pub fn derive_graph_partial_eq(input: ProcTokenStream) -> ProcTokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let err = require_graph_crate();
    let g = graph_crate();
    let name = &input.ident;

    let mut generics = input.generics.clone();
    add_trait_bounds(&mut generics, quote!(#g::GraphPartialEq));
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let body = match &input.data {
        Data::Struct(data) => {
            let field_eqs = field_eqs(&data.fields, quote!(self), quote!(other), &g);
            quote! {
                true #(&& #field_eqs)*
            }
        }
        Data::Enum(data) => {
            let arms = data.variants.iter().map(|variant| {
                let variant_ident = &variant.ident;
                match &variant.fields {
                    Fields::Named(fields) => {
                        let left = fields
                            .named
                            .iter()
                            .map(|f| f.ident.as_ref().unwrap())
                            .collect::<Vec<_>>();
                        let right = left
                            .iter()
                            .map(|id| format_ident!("__gnx_other_{id}"))
                            .collect::<Vec<_>>();
                        let field_eqs = left.iter().zip(&right).map(|(left, right)| {
                            let key = left.to_string();
                            quote! {
                                #g::GraphPartialEq::graph_partial_eq(
                                    #left,
                                    #right,
                                    filter.child(#g::KeyRef::Attr(#key)),
                                    ctx,
                                )
                            }
                        });
                        quote! {
                            (
                                Self::#variant_ident { #(#left,)* },
                                Self::#variant_ident { #(#left: #right,)* },
                            ) => true #(&& #field_eqs)*,
                        }
                    }
                    Fields::Unnamed(fields) => {
                        let left = (0..fields.unnamed.len())
                            .map(|i| format_ident!("__gnx_self_{i}"))
                            .collect::<Vec<_>>();
                        let right = (0..fields.unnamed.len())
                            .map(|i| format_ident!("__gnx_other_{i}"))
                            .collect::<Vec<_>>();
                        let field_eqs =
                            left.iter()
                                .zip(&right)
                                .enumerate()
                                .map(|(i, (left, right))| {
                                    quote! {
                                        #g::GraphPartialEq::graph_partial_eq(
                                            #left,
                                            #right,
                                            filter.child(#g::KeyRef::Index(#i)),
                                            ctx,
                                        )
                                    }
                                });
                        quote! {
                            (
                                Self::#variant_ident(#(#left,)*),
                                Self::#variant_ident(#(#right,)*),
                            ) => true #(&& #field_eqs)*,
                        }
                    }
                    Fields::Unit => {
                        quote! {
                            (Self::#variant_ident, Self::#variant_ident) => true,
                        }
                    }
                }
            });
            quote! {
                match (self, other) {
                    #(#arms)*
                    _ => false,
                }
            }
        }
        Data::Union(data) => {
            return syn::Error::new_spanned(
                &data.union_token,
                "GraphPartialEq cannot be derived for unions",
            )
            .to_compile_error()
            .into();
        }
    };

    ProcTokenStream::from(quote! {
        #err
        impl #impl_generics #g::GraphPartialEq for #name #ty_generics #where_clause {
            fn graph_partial_eq<L, F>(
                &self,
                other: &Self,
                filter: F,
                ctx: &mut #g::GraphEqContext,
            ) -> bool
            where
                L: #g::Leaf,
                F: #g::Filter<L>,
            {
                if filter.matches_ref(self).is_ok() && filter.matches_ref(other).is_ok() {
                    true
                } else {
                    #body
                }
            }
        }
    })
}

pub fn derive_graph_eq(input: ProcTokenStream) -> ProcTokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let err = require_graph_crate();
    let g = graph_crate();
    let name = &input.ident;

    let mut generics = input.generics.clone();
    add_trait_bounds(&mut generics, quote!(#g::GraphEq));
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    ProcTokenStream::from(quote! {
        #err
        impl #impl_generics #g::GraphEq for #name #ty_generics #where_clause {}
    })
}

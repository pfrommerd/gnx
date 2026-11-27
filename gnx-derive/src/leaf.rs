use proc_macro::TokenStream as ProcTokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{DeriveInput, Generics, Type, parse_macro_input, punctuated::Punctuated, token::Comma};

use crate::lifetime_free::expand_lifetime_free_from_derive;
use crate::paths::{cast, graph_crate, require_graph_crate};

fn leaf_impl(ty: &TokenStream2, generics: &Generics) -> TokenStream2 {
    let g = graph_crate();
    let cast = cast();
    let (impl_generics, _, where_clause) = generics.split_for_impl();
    quote! {
        impl #impl_generics #g::Leaf for #ty #where_clause {
            type Ref<'l> = &'l Self
                where Self: 'l;
            fn as_ref<'l>(&'l self) -> Self::Ref<'l> { self }
            fn clone_ref(v: Self::Ref<'_>) -> Self { v.clone() }
            fn try_from_value<V>(g: V) -> Result<Self, V> {
                #cast!(g, Self)
            }
            fn try_from_ref<'v, V>(graph: &'v V) -> Result<Self::Ref<'v>, &'v V> {
                #cast!(graph, Self::Ref<'v>)
            }
            fn try_into_value<V: 'static>(self) -> Result<V, Self> {
                #cast!(self, V)
            }
        }
    }
}

fn leaf_graph_impl(ty: &TokenStream2, generics: &Generics) -> TokenStream2 {
    let g = graph_crate();
    let (impl_generics, _, where_clause) = generics.split_for_impl();
    quote! {
        impl #impl_generics #g::Graph for #ty #where_clause {
            type Owned = Self;
            type Builder<L: #g::Leaf> = #g::LeafBuilder<Self>;

            fn replace<'g, L: #g::Leaf, F: #g::Filter<L>, S: #g::GraphSource<L::Ref<'g>, L>>(
                &'g self, filter: F, source: S, _ctx: &mut #g::GraphContext,
            ) -> Result<Self::Owned, S::Error> {
                match filter.matches_ref(self) {
                    Ok(r) => Ok(source.leaf(r)?
                        .try_into_value()
                        .or_else(<Self as #g::Leaf>::try_from_value)
                        .map_err(|_| <S::Error as #g::Error>::invalid_leaf())?),
                    Err(_) => Ok(self.clone())
                }
            }
            fn builder<'g, L: #g::Leaf, F: #g::Filter<L>, E: #g::Error>(
                &'g self, filter: F, _ctx: &mut #g::GraphContext
            ) -> Result<Self::Builder<L>, E> {
                match filter.matches_ref(self) {
                    Ok(_) => Ok(#g::LeafBuilder::Leaf),
                    Err(_) => Ok(#g::LeafBuilder::Static(self.clone())),
                }
            }

            fn visit<'g, L: #g::Leaf, F: #g::Filter<L>, V: #g::GraphVisitor<'g, Self, L, F::Annotation<'g>>>(
                &'g self, filter: F, visitor: V
            ) -> V::Output {
                match filter.matches_ref(self) {
                    Ok(r) => visitor.visit_leaf(r, filter),
                    Err(s) => visitor.visit_static::<Self>(<Self as #g::Leaf>::as_ref(s))
                }
            }
            fn visit_into<L: #g::Leaf, F: #g::Filter<L>, C: #g::GraphConsumer<Self, L>>(
                self, filter: F, consumer: C
            ) -> C::Output {
                match filter.matches_value(self) {
                    Ok(v) => consumer.consume_leaf(v),
                    Err(s) => consumer.consume_static::<Self>(s)
                }
            }
        }
    }
}

pub fn impl_leaf(input: ProcTokenStream) -> ProcTokenStream {
    let types = parse_macro_input!(input with Punctuated::<Type, Comma>::parse_terminated);
    let err = require_graph_crate();
    let impls = types.iter().map(|ty| {
        let ty = quote!(#ty);
        let generics = Generics::default();
        let graph = leaf_graph_impl(&ty, &generics);
        let leaf = leaf_impl(&ty, &generics);
        quote!(#graph #leaf)
    });
    quote!(#err #(#impls)*).into()
}

fn derived_type(input: &DeriveInput) -> TokenStream2 {
    let name = &input.ident;
    let (_, ty_generics, _) = input.generics.split_for_impl();
    quote!(#name #ty_generics)
}

pub fn derive_maybe_leaf(input: ProcTokenStream) -> ProcTokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let err = require_graph_crate();
    let leaf = leaf_impl(&derived_type(&input), &input.generics);
    quote!(#err #leaf).into()
}

pub fn derive_leaf(input: ProcTokenStream) -> ProcTokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let err = require_graph_crate();
    let ty = derived_type(&input);
    let graph = leaf_graph_impl(&ty, &input.generics);
    let leaf = leaf_impl(&ty, &input.generics);
    let lifetime_free = expand_lifetime_free_from_derive(&input);
    quote!(#err #graph #leaf #lifetime_free).into()
}

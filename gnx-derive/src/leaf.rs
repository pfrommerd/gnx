use proc_macro::TokenStream as ProcTokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{parse_macro_input, punctuated::Punctuated, token::Comma, DeriveInput, Type};

use crate::lifetime_free::expand_lifetime_free_from_derive;
use crate::paths::{graph_crate, require_graph_crate, try_specialize};

pub fn expand_leaf(ty: &Type) -> TokenStream2 {
    let err = require_graph_crate();
    let g = graph_crate();
    let try_specialize = try_specialize();

    quote! {
        #err
        impl #g::Graph for #ty {
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

            fn visit<'g, L: #g::Leaf, F: #g::Filter<L>, V: #g::GraphVisitor<'g, Self, L>>(
                &'g self, filter: F, visitor: V
            ) -> V::Output {
                match filter.matches_ref(self) {
                    Ok(r) => visitor.visit_leaf(r),
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
        impl #g::Leaf for #ty {
            type Ref<'l> = &'l Self
                where Self: 'l;
            fn as_ref<'l>(&'l self) -> Self::Ref<'l> { self }
            fn clone_ref(v: Self::Ref<'_>) -> Self { v.clone() }
            fn try_from_value<V>(g: V) -> Result<Self, V> {
                #try_specialize!(g, Self)
            }
            fn try_from_ref<'v, V>(graph: &'v V) -> Result<Self::Ref<'v>, &'v V> {
                #try_specialize!(graph, Self::Ref<'v>)
            }
            fn try_into_value<V: 'static>(self) -> Result<V, Self> {
                #try_specialize!(self, V)
            }
        }
    }
}

pub fn impl_leaf(input: ProcTokenStream) -> ProcTokenStream {
    let types = parse_macro_input!(input with Punctuated::<Type, Comma>::parse_terminated);
    let expanded = types.iter().map(expand_leaf).collect::<TokenStream2>();
    ProcTokenStream::from(expanded)
}

pub fn expand_leaf_from_derive(input: &DeriveInput) -> TokenStream2 {
    let err = require_graph_crate();
    let g = graph_crate();
    let try_specialize = try_specialize();
    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    quote! {
        #err
        impl #impl_generics #g::Graph for #name #ty_generics #where_clause {
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

            fn visit<'g, L: #g::Leaf, F: #g::Filter<L>, V: #g::GraphVisitor<'g, Self, L>>(
                &'g self, filter: F, visitor: V
            ) -> V::Output {
                match filter.matches_ref(self) {
                    Ok(r) => visitor.visit_leaf(r),
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
        impl #impl_generics #g::Leaf for #name #ty_generics #where_clause {
            type Ref<'l> = &'l Self
                where Self: 'l;
            fn as_ref<'l>(&'l self) -> Self::Ref<'l> { self }
            fn clone_ref(v: Self::Ref<'_>) -> Self { v.clone() }
            fn try_from_value<V>(g: V) -> Result<Self, V> {
                #try_specialize!(g, Self)
            }
            fn try_from_ref<'v, V>(graph: &'v V) -> Result<Self::Ref<'v>, &'v V> {
                #try_specialize!(graph, Self::Ref<'v>)
            }
            fn try_into_value<V: 'static>(self) -> Result<V, Self> {
                #try_specialize!(self, V)
            }
        }
    }
}

pub fn derive_leaf(input: ProcTokenStream) -> ProcTokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let leaf_impl = expand_leaf_from_derive(&input);
    let lifetime_free_impl = expand_lifetime_free_from_derive(&input);

    ProcTokenStream::from(quote! {
        #leaf_impl
        #lifetime_free_impl
    })
}

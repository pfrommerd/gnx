use proc_macro::TokenStream as ProcTokenStream;
use proc_macro2::TokenStream;
use quote::quote;
use syn::{
    Block, Ident, Path, Token, Type, TypePath,
    parse::{Parse, ParseStream, Result},
};
use syn::{DeriveInput, parse_macro_input};

fn impl_leaf_graph(name: Type) -> TokenStream {
    quote! {
        impl Graph for #name {
            type Owned = #name;
            type Builder<'g, L: Leaf, F: Filter<L>> = ::gnx::graph::ViewBuilder<'g, #name, F>
                where Self: 'g;
            type OwnedBuilder<L: Leaf> = ::gnx::graph::LeafBuilder<#name>;

            fn builder<'g, L: Leaf, F: Filter<L>>(&'g self, filter: F) -> Self::Builder<'g, L, F> {
                ::gnx::graph::ViewBuilder::new(self, filter)
            }

            fn visit<'g, L: Leaf, F: Filter<L>, V: GraphVisitor<'g, Self, L>>(
                &'g self, filter: F, visitor: V
            ) -> V::Output {
                match filter.matches_ref(self) {
                    Ok(r) => visitor.visit_leaf(r),
                    Err(s) => visitor.visit_static::<Self>(s.as_ref())
                }
            }
            fn mut_visit<'g, L: Leaf, F: Filter<L>, V: GraphMutVisitor<'g, Self, L>>(
                &'g mut self, filter: F, visitor: V
            ) -> V::Output {
                match filter.matches_mut(self) {
                    Ok(r) => visitor.visit_leaf_mut(r),
                    Err(s) => visitor.visit_static_mut::<Self>(s.as_mut())
                }
            }
            fn into_visit<L: Leaf, F: Filter<L>, C: GraphConsumer<Self, L>>(
                self, filter: F, consumer: C
            ) -> C::Output {
                match filter.matches_value(self) {
                    Ok(v) => consumer.consume_leaf(v),
                    Err(s) => consumer.consume_static::<Self>(s)
                }
            }
        }
        impl TypedGraph<#name> for #name {}
        impl Leaf for #name {
            type Ref<'l> = &'l Self
                where Self: 'l;
            type RefMut<'l> = &'l mut Self
                where Self: 'l;
            fn as_ref<'l>(&'l self) -> Self::Ref<'l> { self }
            fn as_mut<'l>(&'l mut self) -> Self::RefMut<'l> { self }
            fn clone_ref(v: Self::Ref<'_>) -> Self { v.clone() }
            fn clone_mut(v: Self::RefMut<'_>) -> Self { v.clone() }
            fn try_from_value<V>(g: V) -> Result<Self, V> {
                ::castaway::cast!(g, Self)
            }
            fn try_from_ref<'v, V>(graph: &'v V) -> Result<Self::Ref<'v>, &'v V> {
                ::castaway::cast!(graph, &Self)
            }
            fn try_from_mut<'v, V>(graph: &'v mut V) -> Result<Self::RefMut<'v>, &'v mut V> {
                ::castaway::cast!(graph, &mut Self)
            }
            fn try_into_value<V: 'static>(self) -> Result<V, Self> {
                ::castaway::cast!(self, V)
            }
        }
    }
}

#[proc_macro_derive(Leaf)]
pub fn derive_leaf(input: ProcTokenStream) -> ProcTokenStream {
    // Parse the input into a syntax tree
    let input = parse_macro_input!(input as DeriveInput);
    let ty = Type::Path(TypePath {
        qself: None,
        path: Path::from(input.ident),
    });
    ProcTokenStream::from(impl_leaf_graph(ty))
}

#[proc_macro]
pub fn impl_leaf(input: ProcTokenStream) -> ProcTokenStream {
    let name = parse_macro_input!(input as Type);
    ProcTokenStream::from(impl_leaf_graph(name))
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

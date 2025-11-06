use proc_macro::TokenStream as ProcTokenStream;
use proc_macro2::TokenStream;
use quote::quote;
use syn::{DeriveInput, parse_macro_input};
use syn::{Path, Type, TypePath};

fn impl_leaf_graph(name: Type) -> TokenStream {
    quote! {
        impl Graph for #name {
            type Owned = Self;
            type Builder<L: Leaf> = ::gnx::graph::LeafBuilder<Self>;

            fn replace<'g, L: Leaf, F: Filter<L>, S: GraphSource<L::Ref<'g>, L>>(
                &'g self, filter: F, source: S, mut ctx: &mut GraphContext,
            ) -> Result<Self::Owned, S::Error> {
                match filter.matches_ref(self) {
                    Ok(r) => Ok(source.leaf(r)?
                        .try_into_value()
                        .or_else(<Self as ::gnx::graph::Leaf>::try_from_value)
                        .map_err(|_| GraphError::InvalidLeaf)?),
                    Err(_) => Ok(self.clone())
                }
            }
            fn builder<'g, L: Leaf, F: Filter<L>>(
                &'g self, filter: F, ctx: &mut GraphContext
            ) -> Result<Self::Builder<L>, GraphError> {
                match filter.matches_ref(self) {
                    Ok(_) => Ok(::gnx::graph::LeafBuilder::Leaf),
                    Err(_) => Ok(::gnx::graph::LeafBuilder::Static(self.clone())),
                }
            }

            fn visit<'g, L: Leaf, F: Filter<L>, V: GraphVisitor<'g, Self, L>>(
                &'g self, filter: F, visitor: V
            ) -> V::Output {
                match filter.matches_ref(self) {
                    Ok(r) => visitor.visit_leaf(r),
                    Err(s) => visitor.visit_static::<Self>(<Self as ::gnx::graph::Leaf>::as_ref(s))
                }
            }
            fn visit_into<L: Leaf, F: Filter<L>, C: GraphConsumer<Self, L>>(
                self, filter: F, consumer: C
            ) -> C::Output {
                match filter.matches_value(self) {
                    Ok(v) => consumer.consume_leaf(v),
                    Err(s) => consumer.consume_static::<Self>(s)
                }
            }
        }
        impl ::gnx::graph::TypedGraph<#name> for #name {}
        impl ::gnx::graph::Leaf for #name {
            type Ref<'l> = &'l Self
                where Self: 'l;
            fn as_ref<'l>(&'l self) -> Self::Ref<'l> { self }
            fn clone_ref(v: Self::Ref<'_>) -> Self { v.clone() }
            fn try_from_value<V>(g: V) -> Result<Self, V> {
                ::gnx::util::try_specialize!(g, Self)
            }
            fn try_from_ref<'v, V>(graph: &'v V) -> Result<Self::Ref<'v>, &'v V> {
                ::gnx::util::try_specialize!(graph, &Self)
            }
            fn try_into_value<V: 'static>(self) -> Result<V, Self> {
                ::gnx::util::try_specialize!(self, V)
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

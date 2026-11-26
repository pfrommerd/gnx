//! Structural graphs for concrete structs of owned graph fields.
use crate::paths;
use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{Data, DeriveInput, Fields, parse_macro_input};

pub fn derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let vis = &input.vis;
    let fields = match &input.data {
        Data::Struct(s) if input.generics.params.is_empty() => match &s.fields {
            Fields::Named(f) if !f.named.is_empty() => Some(&f.named),
            _ => None,
        },
        _ => None,
    };
    let Some(fields) = fields else {
        return syn::Error::new_spanned(
            name,
            "Graph requires a concrete struct with named, owned graph fields",
        )
        .to_compile_error()
        .into();
    };
    let names: Vec<_> = fields.iter().map(|f| f.ident.as_ref().unwrap()).collect();
    let types: Vec<_> = fields.iter().map(|f| &f.ty).collect();
    let keys: Vec<_> = names
        .iter()
        .map(|id| quote!(g::KeyRef::from(stringify!(#id))))
        .collect();
    let graph = paths::graph_crate();
    let builder = format_ident!("__Gnx{}Builder", name);
    quote! {
        const _: () = {
            use #graph as g;
            use g::{Builder as _, ChildrenSource as _, Error as _, Graph as _, GraphHash as _, GraphPartialEq as _};

            #[derive(Clone)]
            #vis struct #builder<__Leaf: g::Leaf> {
                #(#names: <#types as g::Graph>::Builder<__Leaf>,)*
            }
            impl<__Leaf: g::Leaf> g::Builder<__Leaf> for #builder<__Leaf> {
                type Graph = #name;

                fn build<__Source: g::GraphSource<(), __Leaf>>(
                    self, source: __Source, ctx: &mut g::GraphContext,
                ) -> Result<Self::Graph, __Source::Error> {
                    let mut children = source.node()?;
                    Ok(#name {
                        #(#names: self.#names.build(children.expect_child(#keys)?, ctx)?,)*
                    })
                }
            }
            impl g::Graph for #name {
                type Owned = Self;
                type Builder<L: g::Leaf> = g::NodeBuilder<#builder<L>>;

                fn builder<L: g::Leaf, F: g::Filter<L>, E: g::Error>(
                    &self, filter: F, ctx: &mut g::GraphContext,
                ) -> Result<Self::Builder<L>, E> {
                    if filter.matches_ref(self).is_ok() { return Ok(g::NodeBuilder::Leaf); }
                    Ok(g::NodeBuilder::Node(#builder {
                        #(#names: self.#names.builder(filter.child(#keys), ctx)?,)*
                    }))
                }
                fn replace<'a, L: g::Leaf, F: g::Filter<L>, S: g::GraphSource<L::Ref<'a>, L>>(
                    &'a self, filter: F, source: S, ctx: &mut g::GraphContext,
                ) -> Result<Self, S::Error> {
                    if let Ok(value) = filter.matches_ref(self) {
                        return source.leaf(value)?.try_into_value().map_err(|_| S::Error::invalid_leaf());
                    }
                    let mut children = source.node()?;
                    Ok(Self {
                        #(#names: self.#names.replace(filter.child(#keys), children.expect_child(#keys)?, ctx)?,)*
                    })
                }
                fn visit<'a, L: g::Leaf, F: g::Filter<L>, V: g::GraphVisitor<'a, Self, L, F::Annotation<'a>>>(
                    &'a self, filter: F, visitor: V,
                ) -> V::Output {
                    match filter.matches_ref(self) {
                        Ok(value) => visitor.visit_leaf(value, filter),
                        Err(node) => visitor.visit_node(g::View::new(node, filter)),
                    }
                }
                fn visit_into<L: g::Leaf, F: g::Filter<L>, C: g::GraphConsumer<Self, L>>(
                    self, filter: F, consumer: C,
                ) -> C::Output {
                    match filter.matches_value(self) {
                        Ok(value) => consumer.consume_leaf(value),
                        Err(node) => consumer.consume_node(g::Bound::new(node, filter)),
                    }
                }
            }
            impl g::Node for #name {
                fn visit_children<'a, L: g::Leaf, F: g::Filter<L>, V: g::ChildrenVisitor<'a, Self, L, F::Annotation<'a>>>(
                    &'a self, filter: F, mut visitor: V,
                ) -> V::Output {
                    #(visitor.visit_child(#keys, g::View::new(&self.#names, filter.child(#keys)));)*
                    visitor.finish()
                }
                fn visit_into_children<L: g::Leaf, F: g::Filter<L>, C: g::ChildrenConsumer<Self, L>>(
                    self, filter: F, mut consumer: C,
                ) -> C::Output {
                    #(consumer.consume_child(#keys.to_value(), g::Bound::new(self.#names, filter.child(#keys)));)*
                    consumer.finish()
                }
            }
            impl g::GraphHash for #name {
                fn graph_hash<H: g::anyhash::HasherWrite, L: g::Leaf, F: g::Filter<L>>(
                    &self, h: &mut H, filter: F, ctx: &mut g::GraphHashContext,
                ) {
                    g::anyhash::Hash::hash(&::core::any::type_name::<Self>(), h);
                    if filter.matches_ref(self).is_ok() {
                        g::anyhash::Hash::hash(&g::HashMarker::Excluded, h);
                    } else {
                        g::anyhash::Hash::hash(&g::HashMarker::Included, h);
                        #(self.#names.graph_hash(h, filter.child(#keys), ctx);)*
                    }
                }
                fn hash_builder<__Hasher: g::anyhash::HasherWrite, __Leaf: g::Leaf>(
                    builder: &Self::Builder<__Leaf>, h: &mut __Hasher, ctx: &mut g::GraphHashContext,
                ) {
                    g::anyhash::Hash::hash(&::core::any::type_name::<Self>(), h);
                    match builder {
                        g::NodeBuilder::Leaf => g::anyhash::Hash::hash(&g::HashMarker::Excluded, h),
                        g::NodeBuilder::Node(b) => {
                            g::anyhash::Hash::hash(&g::HashMarker::Included, h);
                            #(<#types as g::GraphHash>::hash_builder::<__Hasher, __Leaf>(&b.#names, h, ctx);)*
                        }
                    }
                }
            }
            impl g::GraphPartialEq for #name {
                fn graph_partial_eq<L: g::Leaf, F: g::Filter<L>>(
                    &self, other: &Self, filter: F, ctx: &mut g::GraphEqContext,
                ) -> bool {
                    match (filter.matches_ref(self), filter.matches_ref(other)) {
                        (Ok(_), Ok(_)) => true,
                        (Err(_), Err(_)) => true #(&& self.#names.graph_partial_eq(&other.#names, filter.child(#keys), ctx))*,
                        _ => false,
                    }
                }
            }
            impl g::GraphEq for #name {
                fn builders_eq<__Leaf: g::Leaf>(
                    left: &Self::Builder<__Leaf>, right: &Self::Builder<__Leaf>, ctx: &mut g::GraphEqContext,
                ) -> bool {
                    match (left, right) {
                        (g::NodeBuilder::Leaf, g::NodeBuilder::Leaf) => true,
                        (g::NodeBuilder::Node(a), g::NodeBuilder::Node(b)) => {
                            true #(&& <#types as g::GraphEq>::builders_eq::<__Leaf>(&a.#names, &b.#names, ctx))*
                        }
                        _ => false,
                    }
                }
            }
        };
    }.into()
}

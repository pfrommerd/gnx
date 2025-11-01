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
        impl Graph for #name {
            type GraphDef<
                I: Clone + 'static,
                R: ::gnx::graph::NonLeafRepr
            > = ::gnx::graph::LeafDef<I, #name, R>;
            type Owned = #name;

            fn graph_def<I, L, V, F>(
                &self,
                viewer: V,
                mut map: F,
                _ctx: &mut ::gnx::graph::GraphContext,
            ) -> Result<Self::GraphDef<I, V::NonLeafRepr>, GraphError>
            where
                I: Clone + 'static,
                V: ::gnx::graph::GraphViewer<L>,
                F: FnMut(V::Ref<'_>) -> I,
            {
                match viewer.try_as_leaf(self) {
                    Ok(leaf) => Ok(::gnx::graph::LeafDef::Leaf(map(leaf))),
                    Err(_graph) => Ok(::gnx::graph::LeafDef::NonLeaf(
                        V::NonLeafRepr::try_to_nonleaf(self)?
                    )),
                }
            }
            fn visit<L, V, M>(&self, view: V, visitor: M) -> M::Output
            where
                V: ::gnx::graph::GraphViewer<L>,
                M: ::gnx::graph::GraphVisitor<L, V>,
            {
                match view.try_as_leaf(self) {
                    Ok(leaf) => visitor.leaf(Some(leaf)),
                    Err(_graph) => visitor.leaf(None),
                }
            }
            fn map<L, V, M>(self, view: V, map: M) -> M::Output
            where
                V: ::gnx::graph::GraphViewer<L>,
                M: ::gnx::graph::GraphMap<L, V>,
            {
                match view.try_to_leaf(self) {
                    Ok(leaf) => map.leaf(Some(leaf)),
                    Err(_graph) => map.leaf(None),
                }
            }
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

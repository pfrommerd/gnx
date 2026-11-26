use std::collections::HashSet;

use proc_macro::TokenStream as ProcTokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{
    Expr, Ident, Result, Token, braced, bracketed, parenthesized, parse::Parse, parse::ParseStream,
    parse_macro_input,
};

use crate::paths;

pub fn dag(input: ProcTokenStream) -> ProcTokenStream {
    let input = parse_macro_input!(input as DagInput);
    ProcTokenStream::from(input.expand())
}

pub(crate) enum DagValue {
    Leaf(Expr),
    /// `*`: an empty leaf.
    Empty,
    List(DagContainer),
    Map(DagMap),
}

pub(crate) struct DagContainer {
    bindings: Vec<DagBinding>,
    values: Vec<DagValue>,
}

pub(crate) struct DagMap {
    bindings: Vec<DagBinding>,
    entries: Vec<DagMapEntry>,
}

struct DagBinding {
    ident: Ident,
    value: DagValue,
}

struct DagMapEntry {
    key: Expr,
    value: DagValue,
}

struct DagInput(DagValue, Option<syn::Type>);

impl Parse for DagInput {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let ty = if input.peek(Token![as]) {
            input.parse::<Token![as]>()?;
            let ty = input.parse()?;
            input.parse::<Token![;]>()?;
            Some(ty)
        } else {
            None
        };
        Ok(DagInput(parse_container(input)?, ty))
    }
}

impl DagInput {
    fn expand(self) -> TokenStream2 {
        let g = paths::graph_crate();
        let util = paths::graph_util_mod();
        let mut scope = HashSet::new();
        let dag = self.0.expand_dag(&g, &util, &mut scope, self.1.as_ref());
        if let Some(ty) = self.1 {
            quote!({let dag: #util::Dag<#ty> = #dag; dag})
        } else {
            dag
        }
    }
}

impl DagValue {
    /// Empty-only transform options need the legacy payload type for inference.
    pub(crate) fn has_payload(&self) -> bool {
        match self {
            Self::Leaf(_) => true,
            Self::Empty => false,
            Self::List(list) => list.values.iter().any(Self::has_payload),
            Self::Map(map) => map.entries.iter().any(|e| e.value.has_payload()),
        }
    }

    pub(crate) fn expand_dag(
        &self,
        g: &syn::Path,
        util: &syn::Path,
        scope: &mut HashSet<String>,
        ty: Option<&syn::Type>,
    ) -> TokenStream2 {
        match self {
            DagValue::Leaf(expr) if ty.is_some() => {
                quote!(#util::Dag::Leaf(Some(::core::convert::Into::<#ty>::into(#expr))))
            }
            DagValue::Leaf(expr) => quote!(#util::Dag::Leaf(Some(#expr))),
            DagValue::Empty => quote!(#util::Dag::Leaf(None)),
            DagValue::List(list) => list.expand_dag(g, util, scope, ty),
            DagValue::Map(map) => map.expand_dag(g, util, scope, ty),
        }
    }

    fn expand_child(
        &self,
        g: &syn::Path,
        util: &syn::Path,
        scope: &mut HashSet<String>,
        ty: Option<&syn::Type>,
    ) -> TokenStream2 {
        if let DagValue::Leaf(expr) = self {
            if let Some(ident) = bound_ident(expr, scope) {
                return quote!(#util::DagChild::Shared(#ident.clone()));
            }
        }

        let dag = self.expand_dag(g, util, scope, ty);
        quote!(#util::DagChild::Owned(#dag))
    }
}

impl DagContainer {
    pub(crate) fn expand_dag(
        &self,
        g: &syn::Path,
        util: &syn::Path,
        scope: &mut HashSet<String>,
        ty: Option<&syn::Type>,
    ) -> TokenStream2 {
        let mut local_scope = scope.clone();
        let bindings = expand_bindings(&self.bindings, g, util, &mut local_scope, ty);
        let values = self
            .values
            .iter()
            .map(|value| value.expand_child(g, util, &mut local_scope, ty));

        quote!({
            #(#bindings)*
            #util::Dag::Node(#util::DagNode::List(vec![#(#values),*]))
        })
    }
}

impl DagMap {
    pub(crate) fn expand_dag(
        &self,
        g: &syn::Path,
        util: &syn::Path,
        scope: &mut HashSet<String>,
        ty: Option<&syn::Type>,
    ) -> TokenStream2 {
        let mut local_scope = scope.clone();
        let bindings = expand_bindings(&self.bindings, g, util, &mut local_scope, ty);
        let entries = self.entries.iter().map(|entry| {
            let key = &entry.key;
            let value = entry.value.expand_child(g, util, &mut local_scope, ty);
            quote! {
                __gnx_dag_map.insert(#g::Key::from(#key), #value);
            }
        });

        quote!({
            #(#bindings)*
            let mut __gnx_dag_map = #util::OrderMap::new();
            #(#entries)*
            #util::Dag::Node(#util::DagNode::Map(__gnx_dag_map))
        })
    }
}

fn expand_bindings(
    bindings: &[DagBinding],
    g: &syn::Path,
    util: &syn::Path,
    scope: &mut HashSet<String>,
    ty: Option<&syn::Type>,
) -> Vec<TokenStream2> {
    bindings
        .iter()
        .map(|binding| {
            let ident = &binding.ident;
            let value = binding.value.expand_dag(g, util, scope, ty);
            scope.insert(ident.to_string());
            quote! {
                let #ident = ::std::sync::Arc::new(#value);
            }
        })
        .collect()
}

fn parse_container(input: ParseStream<'_>) -> Result<DagValue> {
    let mut bindings = Vec::new();

    while input.peek(Token![let]) {
        bindings.push(parse_binding(input)?);
    }

    if input.is_empty() {
        return Ok(DagValue::List(DagContainer {
            bindings,
            values: Vec::new(),
        }));
    }

    if starts_map_entry(input)? {
        let mut entries = Vec::new();
        while !input.is_empty() {
            entries.push(parse_map_entry(input)?);
            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            } else {
                break;
            }
        }
        return Ok(DagValue::Map(DagMap { bindings, entries }));
    }

    let mut values = Vec::new();
    let mut saw_comma = false;
    while !input.is_empty() {
        values.push(parse_value(input)?);
        if input.peek(Token![,]) {
            input.parse::<Token![,]>()?;
            saw_comma = true;
        } else {
            break;
        }
    }

    if bindings.is_empty() && values.len() == 1 && !saw_comma {
        Ok(values.pop().unwrap())
    } else {
        Ok(DagValue::List(DagContainer { bindings, values }))
    }
}

/// A lone `*` (followed by `,`, `;`, or the end) is an empty leaf. Anything
/// else starting with `*`, such as `*x`, is an expression leaf.
fn peek_empty(input: ParseStream<'_>) -> bool {
    let fork = input.fork();
    fork.parse::<Token![*]>().is_ok()
        && (fork.is_empty() || fork.peek(Token![,]) || fork.peek(Token![;]))
}

pub(crate) fn parse_value(input: ParseStream<'_>) -> Result<DagValue> {
    if peek_empty(input) {
        input.parse::<Token![*]>()?;
        return Ok(DagValue::Empty);
    }

    if input.peek(syn::token::Bracket) {
        let content;
        bracketed!(content in input);
        return parse_list_contents(&content);
    }

    if input.peek(syn::token::Paren) {
        let content;
        parenthesized!(content in input);
        if content.is_empty() {
            // `()` is the unit value; `*` is the empty leaf.
            return Ok(DagValue::Leaf(syn::parse_quote!(())));
        }
        return parse_container(&content);
    }

    if input.peek(syn::token::Brace) {
        let content;
        braced!(content in input);
        return parse_map_contents(&content);
    }

    input.parse::<Expr>().map(DagValue::Leaf)
}

fn parse_list_contents(input: ParseStream<'_>) -> Result<DagValue> {
    let mut bindings = Vec::new();
    while input.peek(Token![let]) {
        bindings.push(parse_binding(input)?);
    }

    let mut values = Vec::new();
    while !input.is_empty() {
        values.push(parse_value(input)?);
        if input.peek(Token![,]) {
            input.parse::<Token![,]>()?;
        } else {
            break;
        }
    }

    Ok(DagValue::List(DagContainer { bindings, values }))
}

fn parse_map_contents(input: ParseStream<'_>) -> Result<DagValue> {
    let mut bindings = Vec::new();
    while input.peek(Token![let]) {
        bindings.push(parse_binding(input)?);
    }

    let mut entries = Vec::new();
    while !input.is_empty() {
        entries.push(parse_map_entry(input)?);
        if input.peek(Token![,]) {
            input.parse::<Token![,]>()?;
        } else {
            break;
        }
    }

    Ok(DagValue::Map(DagMap { bindings, entries }))
}

fn parse_binding(input: ParseStream<'_>) -> Result<DagBinding> {
    input.parse::<Token![let]>()?;
    let ident = input.parse::<Ident>()?;
    input.parse::<Token![=]>()?;
    let value = parse_value(input)?;
    input.parse::<Token![;]>()?;
    Ok(DagBinding { ident, value })
}

fn parse_map_entry(input: ParseStream<'_>) -> Result<DagMapEntry> {
    let key = input.parse::<Expr>()?;
    input.parse::<Token![:]>()?;
    let value = parse_value(input)?;
    Ok(DagMapEntry { key, value })
}

fn starts_map_entry(input: ParseStream<'_>) -> Result<bool> {
    if peek_empty(input) {
        return Ok(false);
    }
    // A value that isn't an expression (e.g. `[*, 1]`) can't be a map key.
    let fork = input.fork();
    Ok(fork.parse::<Expr>().is_ok() && fork.peek(Token![:]))
}

fn bound_ident<'a>(expr: &'a Expr, scope: &HashSet<String>) -> Option<&'a Ident> {
    let Expr::Path(path) = expr else {
        return None;
    };
    if path.qself.is_some() || path.path.segments.len() != 1 {
        return None;
    }

    let ident = &path.path.segments[0].ident;
    scope.contains(&ident.to_string()).then_some(ident)
}

//! `map!`, `jvp!`, `vjp!`, `grad!`, `value_and_grad!`, and `hessian!`: wrap a
//! closure like `jit_fn!`, apply builder options written in `dag!` syntax, and
//! return a plain function.

use std::collections::HashSet;

use proc_macro::TokenStream as ProcTokenStream;
use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::quote;
use syn::parse::ParseStream;
use syn::{Expr, Ident, Token};

use crate::dag::{DagValue, parse_value};
use crate::paths;
use crate::transforms::{closure_signature, expand_jittable};

/// A transformation macro: the builder function and its options, each with
/// the leaf type of its spec.
pub struct Transform {
    pub function: &'static str,
    pub options: &'static [(&'static str, &'static str)],
}

pub const MAP: Transform = Transform {
    function: "map",
    options: &[
        ("in_axes", "usize"),
        ("out_axes", "usize"),
        ("axis_size", ""),
    ],
};
pub const JVP: Transform = Transform {
    function: "jvp",
    options: &[("wrt", "bool")],
};
pub const VJP: Transform = Transform {
    function: "vjp",
    options: &[("wrt", "bool")],
};
pub const GRAD: Transform = Transform {
    function: "grad",
    options: &[("wrt", "bool")],
};
pub const VALUE_AND_GRAD: Transform = Transform {
    function: "value_and_grad",
    options: &[("wrt", "bool")],
};
pub const HESSIAN: Transform = Transform {
    function: "hessian",
    options: &[("wrt", "bool")],
};

enum OptionValue {
    Spec(DagValue),
    Expr(Expr),
}

struct Input {
    callable: Expr,
    options: Vec<(Ident, OptionValue)>,
}

fn parse_input(input: ParseStream<'_>, transform: &Transform) -> syn::Result<Input> {
    let callable = input.parse::<Expr>()?;
    let mut options = Vec::new();
    let mut seen = HashSet::new();
    while input.peek(Token![,]) {
        input.parse::<Token![,]>()?;
        if input.is_empty() {
            break;
        }
        let name = input.parse::<Ident>()?;
        let Some((_, leaf)) = transform.options.iter().find(|(option, _)| name == option) else {
            let known: Vec<_> = transform.options.iter().map(|(name, _)| *name).collect();
            return Err(syn::Error::new(
                name.span(),
                format!(
                    "`{}!` has no option `{name}` (expected one of: {})",
                    transform.function,
                    known.join(", ")
                ),
            ));
        };
        if !seen.insert(name.to_string()) {
            return Err(syn::Error::new(name.span(), "option given more than once"));
        }
        input.parse::<Token![=]>()?;
        // A flat `wrt = [a, b]` may be argument indices or a legacy mask.
        // Let Rust choose by element type; nested specs and `*` keep DAG syntax.
        let flat_wrt = name == "wrt"
            && input.peek(syn::token::Bracket)
            && input.fork().parse::<syn::ExprArray>().is_ok_and(|a| {
                a.elems
                    .iter()
                    .all(|e| !matches!(e, Expr::Array(_) | Expr::Tuple(_)))
            });
        let value = if flat_wrt {
            let list: syn::ExprArray = input.parse()?;
            if list.elems.is_empty() {
                OptionValue::Expr(syn::parse_quote!([0usize; 0]))
            } else {
                OptionValue::Expr(syn::parse_quote!(&#list[..]))
            }
        } else if leaf.is_empty() {
            OptionValue::Expr(input.parse()?)
        } else {
            OptionValue::Spec(parse_value(input)?)
        };
        options.push((name, value));
    }
    Ok(Input { callable, options })
}

pub fn transform_fn(input: ProcTokenStream, transform: &'static Transform) -> ProcTokenStream {
    let parsed = syn::parse::Parser::parse(
        |input: ParseStream<'_>| parse_input(input, transform),
        input,
    );
    let Input { callable, options } = match parsed {
        Ok(parsed) => parsed,
        Err(error) => return error.to_compile_error().into(),
    };
    let err = paths::require_transforms_crate();
    let as_fn = paths::transforms_callable_as_fn();
    let function = paths::transforms_function_trait(transform.function);
    let graph = paths::graph_crate();
    let util = paths::graph_util_mod();
    let signature = closure_signature(&callable);
    let jittable = expand_jittable(signature, callable);
    let calls = options.iter().map(|(name, value)| {
        let leaf = transform
            .options
            .iter()
            .find(|(option, _)| name == option)
            .map(|(_, leaf)| *leaf)
            .unwrap_or_default();
        match value {
            OptionValue::Spec(DagValue::Leaf(expr)) => quote! { .#name(#expr) },
            OptionValue::Spec(spec) => {
                let dag = spec.expand_dag(&graph, &util, &mut HashSet::new(), None);
                if spec.has_payload() {
                    quote! { .#name(#dag) }
                } else {
                    let leaf = Ident::new(leaf, Span::call_site());
                    quote! { .#name({ let spec: #util::Dag<#leaf> = #dag; spec }) }
                }
            }
            OptionValue::Expr(expr) => quote! { .#name(#expr) },
        }
    });
    let expanded: TokenStream2 = quote! {
        {
            #err
            use #as_fn::*;
            #function(#jittable) #(#calls)* .into_fn()
        }
    };
    expanded.into()
}

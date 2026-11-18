use proc_macro::TokenStream as ProcTokenStream;
use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::spanned::Spanned;
use syn::{
    Expr, ExprClosure, FnArg, Ident, ItemFn, Pat, PatIdent, ReturnType, Token, Type, TypeReference,
    parenthesized, parse_macro_input, parse_quote,
};

use crate::paths::{
    graph_crate, require_transforms_crate, transforms_callable, transforms_callable_as_fn,
    transforms_jit_fn, transforms_jitable,
};

struct JittableInput {
    signature: Option<JittableSignature>,
    callable: Expr,
}

#[derive(Clone)]
struct JittableSignature {
    inputs: Vec<Type>,
    output: Option<Type>,
}

impl Parse for JittableInput {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        if input.peek(Token![fn]) {
            input.parse::<Token![fn]>()?;
            let content;
            parenthesized!(content in input);
            let inputs = content
                .parse_terminated(Type::parse, Token![,])?
                .into_iter()
                .collect();
            let output = if input.peek(Token![->]) {
                input.parse::<Token![->]>()?;
                Some(input.parse()?)
            } else {
                None
            };
            input.parse::<Token![,]>()?;
            let callable = input.parse()?;
            Ok(Self {
                signature: Some(JittableSignature { inputs, output }),
                callable,
            })
        } else {
            Ok(Self {
                signature: None,
                callable: input.parse()?,
            })
        }
    }
}

#[derive(Clone)]
enum InputKind {
    Owned(Type),
    Ref { ty: TypeReference, inner: Type },
}

pub fn jittable(input: ProcTokenStream) -> ProcTokenStream {
    let input = parse_macro_input!(input as JittableInput);
    ProcTokenStream::from(expand_jittable(input.signature, input.callable))
}

pub fn jit_fn(input: ProcTokenStream) -> ProcTokenStream {
    let input = parse_macro_input!(input as Expr);
    let err = require_transforms_crate();
    let as_fn = transforms_callable_as_fn();
    let jit_fn = transforms_jit_fn();
    let signature = closure_signature(&input);
    let jittable = expand_jittable(signature, input);

    let expanded = quote! {
        {
            #err
            use #as_fn::*;
            #jit_fn(#jittable).into_fn()
        }
    };
    ProcTokenStream::from(expanded)
}

pub fn jit(attr: ProcTokenStream, input: ProcTokenStream) -> ProcTokenStream {
    if !attr.is_empty() {
        return syn::Error::new(Span::call_site(), "#[jit] does not accept arguments")
            .to_compile_error()
            .into();
    }
    let input = parse_macro_input!(input as ItemFn);
    let err = require_transforms_crate();
    let jit_fn = transforms_jit_fn();
    let callable = transforms_callable();

    let vis = &input.vis;
    let sig = &input.sig;
    let block = &input.block;
    let inputs = &sig.inputs;

    let mut mod_sig = sig.clone();
    mod_sig
        .inputs
        .iter_mut()
        .enumerate()
        .for_each(|(i, input)| {
            if let FnArg::Typed(pat) = input {
                *pat.pat = Pat::Ident(PatIdent {
                    attrs: vec![],
                    by_ref: None,
                    mutability: None,
                    ident: Ident::new(format!("__arg{}", i).as_str(), pat.span()),
                    subpat: None,
                });
            }
        });
    let args_gathered = inputs
        .iter()
        .enumerate()
        .map(|(i, input)| Ident::new(format!("__arg{}", i).as_str(), input.span()))
        .collect::<Vec<_>>();
    let inputs_ty = inputs
        .iter()
        .map(|input| match input {
            FnArg::Receiver(receiver) => receiver.ty.as_ref().clone(),
            FnArg::Typed(pat) => pat.ty.as_ref().clone(),
        })
        .collect::<Vec<_>>();
    let output_ty = match &sig.output {
        ReturnType::Default => parse_quote! { () },
        ReturnType::Type(_, ty) => ty.as_ref().clone(),
    };
    let f = quote! {
        |#inputs| -> #output_ty #block
    };
    let jittable = expand_jittable(
        Some(JittableSignature {
            inputs: inputs_ty.clone(),
            output: Some(output_ty),
        }),
        parse_quote! { #f },
    );

    let expanded = quote! {
        #vis #mod_sig {
            #err
            let jitted_fn = #jit_fn(#jittable);
            #callable::invoke(&jitted_fn, (#(#args_gathered,)*))
        }
    };

    ProcTokenStream::from(expanded)
}

fn closure_signature(input: &Expr) -> Option<JittableSignature> {
    let Expr::Closure(ExprClosure { inputs, output, .. }) = input else {
        return None;
    };
    let inputs = inputs
        .iter()
        .map(|pat| match pat {
            Pat::Type(pat) => pat.ty.as_ref().clone(),
            _ => parse_quote! { _ },
        })
        .collect::<Vec<_>>();
    let output = match output {
        ReturnType::Default => None,
        ReturnType::Type(_, ty) => Some(ty.as_ref().clone()),
    };
    Some(JittableSignature { inputs, output })
}

fn expand_jittable(signature: Option<JittableSignature>, callable: Expr) -> TokenStream2 {
    let err = require_transforms_crate();
    let impls = match signature {
        Some(signature) => expand_signature_impls(signature),
        None => expand_owned_impls(),
    };

    quote! {
        {
            #err
            struct _FnContainer<F>(F);
            #impls
            _FnContainer(#callable)
        }
    }
}

fn expand_owned_impls() -> TokenStream2 {
    let callable_trait = transforms_callable();
    let jitable_trait = transforms_jitable();
    let graph = graph_crate();
    (0..=12)
        .map(|arity| {
            let type_params = generic_idents(arity, Span::call_site());
            let vars = (0..arity)
                .map(|i| Ident::new(&format!("__arg{}", i), Span::call_site()))
                .collect::<Vec<_>>();
            quote! {
                impl<F, OUT: #graph::Graph, #(#type_params: #graph::Graph,)*>
                    #callable_trait<(#(#type_params,)*)> for _FnContainer<F>
                where
                    F: Fn(#(#type_params),*) -> OUT,
                {
                    type Output = OUT;

                    fn invoke(&self, input: (#(#type_params,)*)) -> Self::Output {
                        let (#(#vars,)*) = input;
                        (self.0)(#(#vars),*)
                    }
                }

                impl<F, OUT: #graph::Graph<Owned = OUT>, #(#type_params: #graph::Graph<Owned = #type_params>,)*>
                    #jitable_trait<(#(#type_params,)*)> for _FnContainer<F>
                where
                    F: Fn(#(#type_params),*) -> OUT,
                {
                    type OwnedOutput = OUT;

                    fn invoke_owned(
                        &self,
                        input: <(#(#type_params,)*) as #graph::Graph>::Owned,
                    ) -> Self::Output {
                        let (#(#vars,)*) = input;
                        (self.0)(#(#vars),*)
                    }
                }
            }
        })
        .collect()
}

fn expand_signature_impls(signature: JittableSignature) -> TokenStream2 {
    let callable_trait = transforms_callable();
    let jitable_trait = transforms_jitable();
    let graph = graph_crate();
    let span = Span::call_site();
    let mut generic_types = Vec::new();
    let mut ref_lifetimes = Vec::new();
    let mut kinds = Vec::new();

    for (i, input) in signature.inputs.into_iter().enumerate() {
        let ty = normalize_infer(input, i, &mut generic_types);
        match ty {
            Type::Reference(mut ty_ref) => {
                let lifetime = syn::Lifetime::new(&format!("'__jittable{}", i), span);
                ty_ref.lifetime = Some(lifetime.clone());
                let inner = normalize_infer(*ty_ref.elem, i, &mut generic_types);
                ty_ref.elem = Box::new(inner.clone());
                ref_lifetimes.push(lifetime);
                kinds.push(InputKind::Ref { ty: ty_ref, inner });
            }
            ty => kinds.push(InputKind::Owned(ty)),
        }
    }

    let input_types = kinds
        .iter()
        .map(|kind| match kind {
            InputKind::Owned(ty) => quote! { #ty },
            InputKind::Ref { ty, .. } => quote! { #ty },
        })
        .collect::<Vec<_>>();
    let mut call_lifetimes = Vec::new();
    let fn_input_types = kinds
        .iter()
        .enumerate()
        .map(|(i, kind)| match kind {
            InputKind::Owned(ty) => quote! { #ty },
            InputKind::Ref { ty, .. } => {
                let mut ty = ty.clone();
                let lifetime = syn::Lifetime::new(&format!("'__jittable_call{}", i), span);
                ty.lifetime = Some(lifetime.clone());
                call_lifetimes.push(lifetime);
                quote! { #ty }
            }
        })
        .collect::<Vec<_>>();
    let output_type = signature
        .output
        .map(|ty| normalize_output(ty, &mut generic_types))
        .unwrap_or_else(|| OutputType {
            ty: quote! { __JittableOutput },
            needs_generic: true,
        });
    let mut impl_generics = ref_lifetimes
        .iter()
        .map(|lifetime| quote! { #lifetime })
        .collect::<Vec<_>>();
    impl_generics.push(quote! { F });
    if output_type.needs_generic {
        impl_generics.push(quote! { __JittableOutput });
    }
    impl_generics.extend(generic_types.iter().map(|ident| quote! { #ident }));
    let vars = (0..input_types.len())
        .map(|i| Ident::new(&format!("__arg{}", i), span))
        .collect::<Vec<_>>();
    let callable_bounds = kinds
        .iter()
        .map(|kind| match kind {
            InputKind::Owned(ty) => quote! { #ty: #graph::Graph },
            InputKind::Ref { inner, .. } => quote! { #inner: #graph::Graph },
        })
        .collect::<Vec<_>>();
    let owned_bounds = kinds
        .iter()
        .map(|kind| match kind {
            InputKind::Owned(ty) => quote! { #ty: #graph::Graph<Owned = #ty> },
            InputKind::Ref { inner, .. } => quote! { #inner: #graph::Graph<Owned = #inner> },
        })
        .collect::<Vec<_>>();
    let owned_args = kinds
        .iter()
        .zip(vars.iter())
        .map(|(kind, var)| match kind {
            InputKind::Owned(_) => quote! { #var },
            InputKind::Ref { .. } => quote! { #var.as_ref() },
        })
        .collect::<Vec<_>>();
    let fn_bound = if call_lifetimes.is_empty() {
        quote! { F: Fn(#(#fn_input_types),*) -> #output_type }
    } else {
        quote! { F: for<#(#call_lifetimes),*> Fn(#(#fn_input_types),*) -> #output_type }
    };
    let callable_impl = quote! {
        impl<#(#impl_generics),*>
            #callable_trait<(#(#input_types,)*)> for _FnContainer<F>
        where
            #(#callable_bounds,)*
            #output_type: #graph::Graph,
            #fn_bound,
        {
            type Output = #output_type;

            fn invoke(&self, input: (#(#input_types,)*)) -> Self::Output {
                let (#(#vars,)*) = input;
                (self.0)(#(#vars),*)
            }
        }
    };

    quote! {
        #callable_impl

        impl<#(#impl_generics),*>
            #jitable_trait<(#(#input_types,)*)> for _FnContainer<F>
        where
            #(#owned_bounds,)*
            #output_type: #graph::Graph<Owned = #output_type>,
            #fn_bound,
        {
            type OwnedOutput = #output_type;

            fn invoke_owned(
                &self,
                input: <(#(#input_types,)*) as #graph::Graph>::Owned,
            ) -> Self::Output {
                let (#(#vars,)*) = input;
                (self.0)(#(#owned_args),*)
            }
        }
    }
}

fn normalize_infer(ty: Type, index: usize, generic_types: &mut Vec<Ident>) -> Type {
    if matches!(ty, Type::Infer(_)) {
        let ident = Ident::new(&format!("__JittableInput{}", index), Span::call_site());
        push_unique(generic_types, ident.clone());
        parse_quote! { #ident }
    } else {
        ty
    }
}

struct OutputType {
    ty: TokenStream2,
    needs_generic: bool,
}

impl quote::ToTokens for OutputType {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        tokens.extend(self.ty.clone());
    }
}

fn normalize_output(ty: Type, generic_types: &mut Vec<Ident>) -> OutputType {
    if matches!(ty, Type::Infer(_)) {
        OutputType {
            ty: quote! { __JittableOutput },
            needs_generic: true,
        }
    } else {
        let ty = normalize_infer(ty, usize::MAX, generic_types);
        OutputType {
            ty: quote! { #ty },
            needs_generic: false,
        }
    }
}

fn push_unique(generic_types: &mut Vec<Ident>, ident: Ident) {
    if !generic_types.iter().any(|existing| existing == &ident) {
        generic_types.push(ident);
    }
}

fn generic_idents(arity: usize, span: Span) -> Vec<Ident> {
    (0..arity)
        .map(|i| Ident::new(&format!("__JittableInput{}", i), span))
        .collect()
}

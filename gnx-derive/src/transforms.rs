use proc_macro::TokenStream as ProcTokenStream;
use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::quote;
use syn::ext::IdentExt;
use syn::parse::{Parse, ParseStream};
use syn::spanned::Spanned;
use syn::{
    Expr, ExprClosure, FnArg, Ident, ItemFn, Pat, PatIdent, ReturnType, Token, Type, TypeReference,
    parenthesized, parse_macro_input, parse_quote,
};

use crate::paths::{
    graph_crate, require_transforms_crate, transforms_callable_as_fn, transforms_function_trait,
    transforms_jit_fn,
};

struct JittableInput {
    signature: Option<JittableSignature>,
    callable: Expr,
}

#[derive(Clone)]
pub(crate) struct JittableSignature {
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

pub fn jit_fn(input: ProcTokenStream, fallible: bool) -> ProcTokenStream {
    let input = parse_macro_input!(input as Expr);
    let err = require_transforms_crate();
    let as_fn = transforms_callable_as_fn();
    let jit_fn = transforms_jit_fn();
    let signature = closure_signature(&input);
    let jittable = expand_jittable(signature, input);
    let into_fn = if fallible {
        quote!(into_try_fn)
    } else {
        quote!(into_fn)
    };

    let expanded = quote! {
        {
            #err
            use #as_fn::*;
            #jit_fn(#jittable).#into_fn()
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
    let graph = graph_crate();

    let vis = &input.vis;
    let sig = &input.sig;
    let name = sig.ident.unraw().to_string();
    let mut jit_type = jit_fn;
    jit_type.segments.last_mut().unwrap().ident = Ident::new("Jit", Span::call_site());
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
    let signature = JittableSignature {
        inputs: inputs_ty.clone(),
        output: Some(output_ty.clone()),
    };
    let invoke = if sig.generics.params.is_empty() {
        let impls = expand_signature_impls(signature);
        quote! {
            struct _FnContainer<F>(F);
            #impls
            static __JIT: #jit_type<(#(#inputs_ty,)*), _FnContainer<fn(#(#inputs_ty),*) -> #output_ty>> =
                #jit_type::new_with_hint(_FnContainer((#f) as fn(#(#inputs_ty),*) -> #output_ty), ::std::borrow::Cow::Borrowed(#name));
            #graph::util::GraphResult::from_result(__JIT.try_invoke_compatible((#(#args_gathered,)*)))
        }
    } else {
        // A local static can't name a function's generic parameters.
        let jittable = expand_jittable(Some(signature), parse_quote! { #f });
        quote! {
            let jitted_fn = #jit_type::new_with_hint(#jittable, ::std::borrow::Cow::Borrowed(#name));
            #graph::util::GraphResult::from_result(jitted_fn.try_invoke_compatible((#(#args_gathered,)*)))
        }
    };

    let expanded = quote! {
        #vis #mod_sig {
            #err
            #invoke
        }
    };

    ProcTokenStream::from(expanded)
}

pub(crate) fn closure_signature(input: &Expr) -> Option<JittableSignature> {
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

pub(crate) fn expand_jittable(
    signature: Option<JittableSignature>,
    callable: Expr,
) -> TokenStream2 {
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
    let function = transforms_function_trait("Function");
    let callable = transforms_function_trait("Callable");
    let try_callable = transforms_function_trait("TryCallable");
    let jitable = transforms_function_trait("Jitable");
    let try_jitable = transforms_function_trait("TryJitable");
    let graph = graph_crate();
    quote! {
        impl<I: #graph::Graph, F: #function<I>> #function<I> for _FnContainer<F> {
            type Output = F::Output;
        }

        impl<I: #graph::Graph, F: #callable<I>> #callable<I> for _FnContainer<F> {
            fn invoke(&self, input: I) -> Self::Output {
                self.0.invoke(input)
            }
        }

        impl<I: #graph::Graph, F: #try_callable<I>> #try_callable<I> for _FnContainer<F> {
            type Error = F::Error;

            fn try_invoke(&self, input: I) -> ::core::result::Result<Self::Output, Self::Error> {
                self.0.try_invoke(input)
            }
        }

        impl<I: #graph::Graph, F: #jitable<I>> #jitable<I> for _FnContainer<F> {
            fn invoke_owned(&self, input: I::Owned) -> Self::Output {
                self.0.invoke_owned(input)
            }
        }

        impl<I: #graph::Graph, F: #try_jitable<I>> #try_jitable<I> for _FnContainer<F> {
            fn try_invoke_owned(&self, input: I::Owned) -> ::core::result::Result<Self::Output, Self::Error> {
                self.0.try_invoke_owned(input)
            }
        }
    }
}

fn expand_signature_impls(signature: JittableSignature) -> TokenStream2 {
    let function = transforms_function_trait("Function");
    let callable = transforms_function_trait("Callable");
    let try_callable = transforms_function_trait("TryCallable");
    let jitable = transforms_function_trait("Jitable");
    let try_jitable = transforms_function_trait("TryJitable");
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
    impl_generics.extend(generic_types.iter().map(|ident| quote! { #ident }));
    let adapter_generics = impl_generics.clone();
    if output_type.needs_generic {
        impl_generics.push(quote! { __JittableOutput });
    }
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
    quote! {
        impl<#(#impl_generics),*> #function<(#(#input_types,)*)> for _FnContainer<F>
        where
            #(#callable_bounds,)*
            #output_type: #graph::util::GraphResult,
            #fn_bound,
        {
            type Output = <#output_type as #graph::util::GraphResult>::Output;
        }

        impl<#(#impl_generics),*> #try_callable<(#(#input_types,)*)> for _FnContainer<F>
        where
            #(#callable_bounds,)*
            #output_type: #graph::util::GraphResult,
            #fn_bound,
        {
            type Error = <#output_type as #graph::util::GraphResult>::Error;

            fn try_invoke(&self, input: (#(#input_types,)*)) -> ::core::result::Result<Self::Output, Self::Error> {
                let (#(#vars,)*) = input;
                #graph::util::GraphResult::into_result((self.0)(#(#vars),*))
            }
        }

        impl<#(#adapter_generics),*> #callable<(#(#input_types,)*)> for _FnContainer<F>
        where
            #(#callable_bounds,)*
            Self: #try_callable<(#(#input_types,)*), Error = ::core::convert::Infallible>,
        {
            fn invoke(&self, input: (#(#input_types,)*)) -> Self::Output {
                <Self as #try_callable<(#(#input_types,)*)>>::try_invoke(self, input)
                    .unwrap_or_else(|never| match never {})
            }
        }

        impl<#(#impl_generics),*> #try_jitable<(#(#input_types,)*)> for _FnContainer<F>
        where
            #(#owned_bounds,)*
            #output_type: #graph::util::GraphResult,
            <#output_type as #graph::util::GraphResult>::Output: #graph::OwnedGraph,
            #fn_bound,
        {
            fn try_invoke_owned(
                &self,
                input: <(#(#input_types,)*) as #graph::Graph>::Owned,
            ) -> ::core::result::Result<Self::Output, Self::Error> {
                let (#(#vars,)*) = input;
                #graph::util::GraphResult::into_result((self.0)(#(#owned_args),*))
            }
        }

        impl<#(#adapter_generics),*> #jitable<(#(#input_types,)*)> for _FnContainer<F>
        where
            #(#owned_bounds,)*
            Self: #callable<(#(#input_types,)*)>
                + #try_jitable<(#(#input_types,)*), Error = ::core::convert::Infallible>,
        {
            fn invoke_owned(&self, input: <(#(#input_types,)*) as #graph::Graph>::Owned) -> Self::Output {
                <Self as #try_jitable<(#(#input_types,)*)>>::try_invoke_owned(self, input)
                    .unwrap_or_else(|never| match never {})
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

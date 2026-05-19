use proc_macro::TokenStream as ProcTokenStream;
use quote::quote;
use syn::{
    parse_macro_input, Expr, FnArg, Ident, ItemFn, Pat, PatIdent, ReturnType, Type,
};
use syn::spanned::Spanned;

use crate::paths::{
    require_transforms_crate, transforms_callable, transforms_callable_as_fn,
    transforms_jit_fn, transforms_jit_type,
};

pub fn jit(input: ProcTokenStream) -> ProcTokenStream {
    let input = parse_macro_input!(input as Expr);
    let err = require_transforms_crate();
    let as_fn = transforms_callable_as_fn();
    let jit_fn = transforms_jit_fn();

    let expanded = quote! {
        {
            #err
            use #as_fn::*;
            #jit_fn(#input).into_fn()
        }
    };
    ProcTokenStream::from(expanded)
}

pub fn jit_fn(_attr: ProcTokenStream, input: ProcTokenStream) -> ProcTokenStream {
    let input = parse_macro_input!(input as ItemFn);
    let err = require_transforms_crate();
    let jit_type = transforms_jit_type();
    let jit_fn = transforms_jit_fn();
    let callable = transforms_callable();

    let vis = &input.vis;
    let sig = &input.sig;
    let block = &input.block;
    let inputs = &sig.inputs;

    let mut mod_sig = sig.clone();
    mod_sig.inputs.iter_mut().enumerate().for_each(|(i, mut input)| {
        if let FnArg::Typed(pat) = &mut input {
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
            FnArg::Receiver(receiver) => receiver.ty.clone(),
            FnArg::Typed(pat) => pat.ty.clone(),
        })
        .collect::<Vec<_>>();
    let output_ty = match &sig.output {
        ReturnType::Default => Type::Verbatim(quote! { () }),
        ReturnType::Type(_, ty) => ty.as_ref().clone(),
    };

    let expanded = quote! {
        #vis #mod_sig {
            #err
            static jit_fn : ::std::sync::LazyLock<
              #jit_type< fn(#(#inputs_ty,)*) -> #output_ty >
            > =
                ::std::sync::LazyLock::new(|| {
                    let f: fn(#(#inputs_ty,)*) -> #output_ty = |#inputs| #block;
                    #jit_fn(f)
                });
            let jitted_fn = &*jit_fn;
            #callable::invoke(jitted_fn, (#(#args_gathered,)*))
        }
    };

    ProcTokenStream::from(expanded)
}

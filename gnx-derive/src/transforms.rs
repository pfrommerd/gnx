use proc_macro::TokenStream as ProcTokenStream;
use quote::quote;
use syn::{Expr, FnArg, Ident, ItemFn, Pat, PatIdent, ReturnType, Type, parse_macro_input};
use syn::spanned::Spanned;

pub fn jit(input: ProcTokenStream) -> ProcTokenStream {
    let input = parse_macro_input!(input as Expr);

    let expanded = quote! {
        {
            use ::gnx::callable::as_fn::*;
            ::gnx::transforms::jit(#input).into_fn()
        }
    };
    ProcTokenStream::from(expanded)
}

// Will wrap a function item with a jit wrapper.
pub fn jit_fn(_attr: ProcTokenStream, input: ProcTokenStream) -> ProcTokenStream {
    let input = parse_macro_input!(input as ItemFn);
    let vis = &input.vis;
    let sig = &input.sig;
    let block = &input.block;
    let inputs = &sig.inputs;

    // Modify the signature to replace all
    // input patterns with __arg0, __arg1, etc.
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
    let args_gathered = inputs.iter().enumerate().map(|(i, input)| {
        Ident::new(format!("__arg{}", i).as_str(), input.span())
    }).collect::<Vec<_>>();
    let inputs_ty = inputs.iter().map(|input| {
        match input {
            FnArg::Receiver(receiver) => receiver.ty.clone(),
            FnArg::Typed(pat) => pat.ty.clone(),
        }
    }).collect::<Vec<_>>();
    let output_ty = match &sig.output {
        ReturnType::Default => Type::Verbatim(quote! { () }),
        ReturnType::Type(_, ty) => ty.as_ref().clone(),
    };

    let expanded = quote! {
        #vis #mod_sig {
            static jit_fn : ::std::sync::LazyLock<
              ::gnx::transforms::Jit< fn(#(#inputs_ty,)*) -> #output_ty >
            > =
                ::std::sync::LazyLock::new(|| {
                    let f: fn(#(#inputs_ty,)*) -> #output_ty = |#inputs| #block;
                    ::gnx::jit(f)
                });
            let jitted_fn = &*jit_fn;
            ::gnx::Callable::invoke(jitted_fn, (#(#args_gathered,)*))
        }
    };

    ProcTokenStream::from(expanded)
}
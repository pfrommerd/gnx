//! GNX-native implementations of the Serde derive data model.
use proc_macro::TokenStream;
use proc_macro_crate::{FoundCrate, crate_name};
use proc_macro2::TokenStream as Tokens;
use quote::{format_ident, quote};
use serde_derive_internals::{
    Ctxt, Derive,
    ast::{Container, Data, Field},
    attr::{Default, TagType},
};
use syn::{DeriveInput, parse_macro_input, parse_quote};

mod de;
mod identifier;
mod ser;

#[proc_macro_derive(Serialize, attributes(serde))]
pub fn serialize(input: TokenStream) -> TokenStream {
    expand(parse_macro_input!(input), false)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}
#[proc_macro_derive(Deserialize, attributes(serde))]
pub fn deserialize(input: TokenStream) -> TokenStream {
    expand(parse_macro_input!(input), true)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

fn expand(mut input: DeriveInput, de: bool) -> syn::Result<Tokens> {
    serde_derive_internals::replace_receiver(&mut input);
    let cx = Ctxt::new();
    let c = Container::from_ast(
        &cx,
        &input,
        if de {
            Derive::Deserialize
        } else {
            Derive::Serialize
        },
    );
    cx.check()?;
    let c = c.unwrap();
    let a = &c.attrs;
    let io = if let Some(path) = a.custom_serde_path() {
        quote!(#path)
    } else {
        let name = match crate_name("gnx-io") {
            Ok(FoundCrate::Name(n)) => format_ident!("{n}"),
            _ => format_ident!("gnx_io"),
        };
        quote!(::#name)
    };
    let mut generics = input.generics.clone();
    if de {
        if let Some(l) = generics.lifetimes().find(|p| p.lifetime.ident == "de") {
            return Err(syn::Error::new_spanned(
                l,
                "cannot derive Deserialize with a lifetime named 'de",
            ));
        }
        generics.params.insert(0, parse_quote!('de));
    }
    let global = if de { a.de_bound() } else { a.ser_bound() };
    let predicates = &mut generics.make_where_clause().predicates;
    if let Some(bounds) = global {
        predicates.extend(bounds.iter().cloned());
    }
    if de && matches!(a.default(), Default::Default) {
        let name = &c.ident;
        let (_, ty, _) = c.generics.split_for_impl();
        predicates.push(parse_quote!(#name #ty: ::core::default::Default));
    }
    let mut add = |f: &Field, infer: bool| {
        let t = f.ty;
        let a = &f.attrs;
        let skip = if de {
            a.skip_deserializing()
        } else {
            a.skip_serializing()
        };
        if !de && c.attrs.is_packed() && !skip {
            predicates.push(parse_quote!(#t: ::core::marker::Copy));
        }
        if de && matches!(a.default(), Default::Default) {
            predicates.push(parse_quote!(#t: ::core::default::Default));
        }
        if de {
            for l in a.borrowed_lifetimes() {
                predicates.push(parse_quote!('de: #l));
            }
        }
        if let Some(bounds) = if de { a.de_bound() } else { a.ser_bound() } {
            predicates.extend(bounds.iter().cloned());
        } else if infer
            && global.is_none()
            && !skip
            && (if de {
                a.deserialize_with().is_none()
            } else {
                a.serialize_with().is_none()
            })
        {
            // Bound T (or its associated type), not a recursive field like Vec<Node<T>>.
            struct Used<'a>(&'a syn::Generics, Vec<syn::TypePath>);
            impl<'ast> syn::visit::Visit<'ast> for Used<'_> {
                fn visit_type_path(&mut self, ty: &'ast syn::TypePath) {
                    let path = &ty.path;
                    if path
                        .segments
                        .last()
                        .is_some_and(|s| s.ident == "PhantomData")
                    {
                        return;
                    }
                    if path
                        .segments
                        .first()
                        .is_some_and(|s| self.0.type_params().any(|p| p.ident == s.ident))
                        || ty.qself.is_some()
                    {
                        self.1.push(ty.clone());
                    } else {
                        syn::visit::visit_type_path(self, ty);
                    }
                }
            }
            let mut used = Used(c.generics, Vec::new());
            syn::visit::Visit::visit_type(&mut used, t);
            for t in used.1 {
                predicates.push(if de {
                    parse_quote!(#t: __io::Deserialize<'de>)
                } else {
                    parse_quote!(#t: __io::Serialize)
                });
            }
        }
    };
    match &c.data {
        Data::Struct(_, fs) => {
            for f in fs {
                add(f, true);
            }
        }
        Data::Enum(vs) => {
            for v in vs {
                let infer = if de {
                    v.attrs.de_bound().is_none() && v.attrs.deserialize_with().is_none()
                } else {
                    v.attrs.ser_bound().is_none() && v.attrs.serialize_with().is_none()
                };
                if !(if de {
                    v.attrs.skip_deserializing()
                } else {
                    v.attrs.skip_serializing()
                }) {
                    for f in &v.fields {
                        add(f, infer);
                    }
                }
            }
        }
    }
    if let Data::Enum(vs) = &c.data {
        for v in vs {
            if let Some(b) = if de {
                v.attrs.de_bound()
            } else {
                v.attrs.ser_bound()
            } {
                predicates.extend(b.iter().cloned());
            }
        }
    }
    let body = if de {
        de::expand(&c, &generics)?
    } else {
        ser::expand(&c, &generics)?
    };
    Ok(
        quote!(const _: () = { use #io as __io; #[allow(unused_imports)] use __io::{Serializer as _, Deserializer as _}; #body };),
    )
}

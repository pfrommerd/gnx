use super::*;
use serde_derive_internals::ast::Style;

pub(super) fn expand(c: &Container, g: &syn::Generics) -> syn::Result<Tokens> {
    let name = &c.ident;
    let wire = c.attrs.name().serialize_name();
    let (imp, _, bounds) = g.split_for_impl();
    let (_, ty, _) = c.generics.split_for_impl();
    let path = c
        .attrs
        .remote()
        .map(|p| quote!(#p))
        .unwrap_or_else(|| quote!(#name));
    let receiver = if c.attrs.remote().is_some() {
        quote!(__value)
    } else {
        quote!(self)
    };
    let mut helpers = Tokens::new();
    let mut payload = |style: Style,
                       fs: &[Field],
                       variant: Option<(usize, &str)>,
                       external: bool| {
        let mut fields = Vec::new();
        for (i, f) in fs.iter().enumerate() {
            if f.attrs.skip_serializing() {
                continue;
            }
            let member = &f.member;
            let binding = format_ident!("__f{i}");
            let raw = if variant.is_some() {
                quote!(#binding)
            } else if let Some(getter) = f.attrs.getter() {
                quote!(&#getter(#receiver))
            } else if c.attrs.is_packed() {
                quote!(&{ #receiver.#member })
            } else {
                quote!(&#receiver.#member)
            };
            let value = if let Some(hook) = f.attrs.serialize_with() {
                let wrapper = format_ident!("__Hook{}_{}", variant.map_or(0, |v| v.0), i);
                let t = f.ty;
                helpers.extend(quote!{
                    struct #wrapper<'a,T:?Sized,C>(&'a T,::core::marker::PhantomData<C>);
                    impl #imp __io::Serialize for #wrapper<'_,#t,#name #ty> #bounds {
                        fn serialize<__S:__io::Serializer>(&self,s:__S)->::core::result::Result<__S::Ok,__S::Error>{#hook(self.0,s)}
                    }
                });
                quote!(&#wrapper(#raw,::core::marker::PhantomData::<#name #ty>))
            } else {
                raw.clone()
            };
            let include = f
                .attrs
                .skip_serializing_if()
                .map(|p| quote!(!#p(#raw)))
                .unwrap_or(quote!(true));
            fields.push((f, value, include));
        }
        let v_args = variant.filter(|_| external).map(|(i, n)| {
            let i = i as u32;
            quote!(#i,#n,)
        });
        if c.attrs.transparent() {
            let (_, v, _) = fields
                .iter()
                .find(|(f, _, _)| f.attrs.transparent())
                .unwrap();
            return quote!(__io::Serialize::serialize(#v,s));
        }
        let v = variant.is_some() && external;
        if matches!(style, Style::Struct) && fields.iter().any(|(f, _, _)| f.attrs.flatten()) {
            let writes = fields.iter().map(|(f, val, flag)| {
                let k = f.attrs.name().serialize_name();
                let action = if f.attrs.flatten() {
                    quote!(__io::Serialize::serialize(#val,__io::flat::Flat(&mut state))?)
                } else {
                    quote!(__io::SerializeMap::serialize_entry(&mut state,#k,#val)?)
                };
                quote!(if #flag { #action; })
            });
            return quote!({let mut state=s.serialize_map(None)?;#(#writes)* __io::SerializeMap::end(state)});
        }
        if matches!(style, Style::Unit) || matches!(style, Style::Newtype) && fields.is_empty() {
            let method = if v {
                format_ident!("serialize_unit_variant")
            } else {
                format_ident!("serialize_unit_struct")
            };
            return quote!(s.#method(#wire,#v_args));
        }
        if matches!(style, Style::Newtype) {
            let value = &fields[0].1;
            let method = if v {
                format_ident!("serialize_newtype_variant")
            } else {
                format_ident!("serialize_newtype_struct")
            };
            return quote!(s.#method(#wire,#v_args #value));
        }
        let named = matches!(style, Style::Struct);
        let (method, tr) = match (named, v) {
            (true, true) => ("serialize_struct_variant", "SerializeStructVariant"),
            (true, false) => ("serialize_struct", "SerializeStruct"),
            (false, true) => ("serialize_tuple_variant", "SerializeTupleVariant"),
            (false, false) => ("serialize_tuple_struct", "SerializeTupleStruct"),
        };
        let method = format_ident!("{method}");
        let tr = format_ident!("{tr}");
        let flags: Vec<_> = (0..fields.len())
            .map(|i| format_ident!("__include{i}"))
            .collect();
        let includes = fields.iter().map(|f| &f.2);
        let stmts = fields.iter().zip(&flags).map(|((f, value, _), flag)| {
            let key = f.attrs.name().serialize_name();
            let key = named.then(|| quote!(#key,));
            quote!(if #flag { __io::#tr::serialize_field(&mut state,#key #value)?; })
        });
        quote!({ #(let #flags=#includes;)* let mut state=s.#method(#wire,#v_args 0 #(+ #flags as usize)*)?; #(#stmts)* __io::#tr::end(state) })
    };
    let mut wrappers = Tokens::new();
    let body = if let Some(target) = c.attrs.type_into() {
        quote!(__io::Serialize::serialize(&<#target as ::core::convert::From<#path #ty>>::from(#receiver.clone()),s))
    } else {
        match &c.data {
            Data::Struct(style, fields) => {
                let body = payload(*style, fields, None, false);
                if let TagType::Internal { tag } = c.attrs.tag() {
                    quote!({let mut state=s.serialize_map(None)?;__io::SerializeMap::serialize_entry(&mut state,#tag,#wire)?;{let s=__io::flat::Flat(&mut state);#body}?;__io::SerializeMap::end(state)})
                } else {
                    body
                }
            }
            Data::Enum(vs) => {
                let mut arms = Vec::new();
                for (i, v) in vs.iter().enumerate() {
                    let id = &v.ident;
                    let key = v.attrs.name().serialize_name();
                    let members = v.fields.iter().map(|f| &f.member);
                    let vars: Vec<_> = (0..v.fields.len())
                        .map(|i| format_ident!("__f{i}"))
                        .collect();
                    let pat = quote!(#path::#id { #(#members: #vars),* });
                    let tag = if v.attrs.untagged() {
                        &TagType::None
                    } else {
                        c.attrs.tag()
                    };
                    if v.attrs.skip_serializing() {
                        arms.push(quote!(#pat => Err(__io::Error::custom(concat!("skipped variant: ",#key)))));
                        continue;
                    }
                    if matches!(tag, TagType::External)
                        && v.attrs.serialize_with().is_none()
                        && !v.fields.iter().any(|f| f.attrs.flatten())
                    {
                        let expr = payload(v.style, &v.fields, Some((i, key)), true);
                        arms.push(quote!(#pat => #expr));
                        continue;
                    }
                    let body = if let Some(hook) = v.attrs.serialize_with() {
                        quote!(#hook(#(#vars,)* s))
                    } else {
                        payload(v.style, &v.fields, Some((i, key)), false)
                    };
                    let expr = match tag {
                        TagType::None => body,
                        TagType::Internal { tag } => quote!({
                            let mut state = s.serialize_map(None)?;
                            __io::SerializeMap::serialize_entry(&mut state, #tag, #key)?;
                            { let s = __io::flat::Flat(&mut state); #body }?;
                            __io::SerializeMap::end(state)
                        }),
                        TagType::External | TagType::Adjacent { .. } => {
                            // A compound serializer accepts a Serialize value, so expose just
                            // this variant's payload through a private, borrowing wrapper.
                            let helper = format_ident!("__Payload{i}");
                            wrappers.extend(quote! {
                                struct #helper<'a,T>(&'a T);
                                impl #imp __io::Serialize for #helper<'_, #path #ty> #bounds {
                                    fn serialize<__S: __io::Serializer>(&self, s: __S) -> ::core::result::Result<__S::Ok, __S::Error> {
                                        match self.0 { #pat => #body, #[allow(unreachable_patterns)] _ => unreachable!() }
                                    }
                                }
                            });
                            if let TagType::Adjacent { tag, content } = tag {
                                let unit = matches!(v.style, Style::Unit)
                                    && v.attrs.serialize_with().is_none();
                                let field = (!unit).then(|| quote!(__io::SerializeStruct::serialize_field(&mut state, #content, &#helper(#receiver))?;));
                                let n = if unit { 1usize } else { 2 };
                                quote!({
                                    let mut state = s.serialize_struct(#wire,#n)?;
                                    __io::SerializeStruct::serialize_field(&mut state, #tag, #key)?;
                                    #field
                                    __io::SerializeStruct::end(state)
                                })
                            } else {
                                let idx = i as u32;
                                quote!(s.serialize_newtype_variant(#wire, #idx, #key, &#helper(#receiver)))
                            }
                        }
                    };
                    arms.push(quote!(#pat => #expr));
                }
                if c.attrs.remote().is_some() && c.attrs.non_exhaustive() {
                    arms.push(quote!(_ => Err(__io::Error::custom("unknown remote variant"))));
                }
                if vs.is_empty() {
                    quote!(match *#receiver {})
                } else {
                    quote!(match #receiver { #(#arms),* })
                }
            }
        }
    };
    if c.attrs.remote().is_some() {
        let vis = &c.original.vis;
        return Ok(
            quote!(#helpers #wrappers impl #imp #name #ty #bounds {#vis fn serialize<__S:__io::Serializer>(__value:&#path #ty,s:__S)->::core::result::Result<__S::Ok,__S::Error>{#body}}),
        );
    }
    Ok(
        quote! { #helpers #wrappers impl #imp __io::Serialize for #name #ty #bounds { fn serialize<__S:__io::Serializer>(&self,s:__S)->::core::result::Result<__S::Ok,__S::Error>{#body} } },
    )
}

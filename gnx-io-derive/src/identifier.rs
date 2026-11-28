use super::*;
use serde_derive_internals::ast::Style;

pub fn body(c: &Container, helpers: &mut Tokens, g: &syn::Generics) -> Tokens {
    let Data::Enum(vs) = &c.data else {
        unreachable!()
    };
    let name = &c.ident;
    let (imp, _, bounds) = g.split_for_impl();
    let (_, ty, _) = c.generics.split_for_impl();
    let mut names = Vec::new();
    let mut numbers = Vec::new();
    let mut fallback = quote!(Err(__E::custom("unknown identifier")));
    let mut numeric_fallback = fallback.clone();
    for (i, v) in vs
        .iter()
        .filter(|v| !v.attrs.skip_deserializing())
        .enumerate()
    {
        let id = &v.ident;
        let key = v.attrs.name().deserialize_name();
        let aliases = v.attrs.aliases().iter().filter(|a| a.as_str() != key);
        let index = i as u64;
        if matches!(v.style, Style::Newtype) {
            fallback = quote!(__io::Deserialize::deserialize(__io::content::Content::Str(value.into()).replay::<__E>()).map(#name::#id));
            numeric_fallback = quote!(__io::Deserialize::deserialize(__io::content::Content::U(value.into()).replay::<__E>()).map(#name::#id));
        } else {
            names.push(quote!(#key #(|#aliases)* =>Ok(#name::#id)));
            numbers.push(quote!(#index=>Ok(#name::#id)));
            if v.attrs.other() {
                fallback = quote!(Ok(#name::#id));
                numeric_fallback = fallback.clone();
            }
        }
    }
    helpers.extend(quote!(impl #imp __io::DataVisitor<'de> for __Visitor<#name #ty,0> #bounds {type Value=#name #ty;
        fn visit_str<__E:__io::Error>(self,value:&str)->::core::result::Result<Self::Value,__E>{let value=value.to_owned();match value.as_str(){#(#names,)* _=>#fallback}}
        fn visit_borrowed_str<__E:__io::Error>(self,value:&'de str)->::core::result::Result<Self::Value,__E>{match value{#(#names,)* _=>#fallback}}
        fn visit_bytes<__E:__io::Error>(self,value:&[u8])->::core::result::Result<Self::Value,__E>{self.visit_str(::core::str::from_utf8(value).map_err(__E::custom)?)}
        fn visit_u64<__E:__io::Error>(self,value:u64)->::core::result::Result<Self::Value,__E>{match value{#(#numbers,)* _=>#numeric_fallback}}
    }));
    quote!(d.deserialize_identifier(__Visitor::<#name #ty,0>(::core::marker::PhantomData)))
}

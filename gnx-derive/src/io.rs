use proc_macro::TokenStream;
use proc_macro_crate::{FoundCrate, crate_name};
use quote::{format_ident, quote};
use syn::{Data, DeriveInput, Fields, parse_macro_input, parse_quote};

pub fn derive(input: TokenStream, deserialize: bool) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let Data::Struct(data) = &input.data else {
        return unsupported(name);
    };
    let Fields::Named(fields) = &data.fields else {
        return unsupported(name);
    };
    let io = match crate_name("gnx-io") {
        Ok(FoundCrate::Name(n)) => {
            let n = format_ident!("{n}");
            quote!(::#n)
        }
        _ => quote!(::gnx_io),
    };
    let names: Vec<_> = fields
        .named
        .iter()
        .map(|f| f.ident.as_ref().unwrap())
        .collect();
    let keys: Vec<_> = names
        .iter()
        .map(|id| syn::ext::IdentExt::unraw(*id).to_string())
        .collect();
    let slots: Vec<_> = (0..names.len())
        .map(|i| format_ident!("__field_{i}"))
        .collect();
    let n = names.len();
    let mut generics = input.generics.clone();
    let (_, ty, _) = input.generics.split_for_impl();
    if deserialize {
        generics.params.insert(0, parse_quote!('__gnx_de));
    }
    for f in &fields.named {
        let t = &f.ty;
        generics
            .make_where_clause()
            .predicates
            .push(if deserialize {
                parse_quote!(#t: #io::Deserialize<'__gnx_de>)
            } else {
                parse_quote!(#t: #io::Serialize)
            });
    }
    let (imp, _, bounds) = generics.split_for_impl();
    if !deserialize {
        return quote! {
            impl #imp #io::Serialize for #name #ty #bounds {
                fn serialize<__S: #io::Serializer>(&self, s: __S) -> ::core::result::Result<__S::Ok, __S::Error> {
                    let mut obj = s.serialize_struct(stringify!(#name), #n)?;
                    #(#io::SerializeStruct::serialize_field(&mut obj, #keys, &self.#names)?;)*
                    #io::SerializeStruct::end(obj)
                }
            }
        }
        .into();
    }
    quote! {
        const _: () = {
            struct __Visitor<T>(::core::marker::PhantomData<T>);
            impl<T> #io::Expecting for __Visitor<T> {
                fn expected(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result { f.write_str(stringify!(#name)) }
            }
            impl #imp #io::DataVisitor<'__gnx_de> for __Visitor<#name #ty> #bounds {
                type Value = #name #ty;
                fn visit_map<__A: #io::MapAccess<'__gnx_de>>(self, mut map: __A) -> ::core::result::Result<Self::Value, __A::Error> {
                    #(let mut #slots = None;)*
                    while let Some(key) = map.next_key::<::std::string::String>()? {
                        match key.as_str() {
                            #(#keys => {
                                if #slots.is_some() { return Err(#io::Error::custom(concat!("duplicate field: ", #keys))); }
                                #slots = Some(map.next_value()?);
                            })*
                            _ => return Err(#io::Error::custom(format!("unknown field: {}", key))),
                        }
                    }
                    Ok(#name { #(#names: #slots.ok_or_else(|| <__A::Error as #io::Error>::custom(concat!("missing field: ", #keys)))?,)* })
                }
            }
            impl #imp #io::Deserialize<'__gnx_de> for #name #ty #bounds {
                fn deserialize<__D: #io::Deserializer<'__gnx_de>>(d: __D) -> ::core::result::Result<Self, __D::Error> {
                    d.deserialize_struct(stringify!(#name), &[#(#keys),*], __Visitor::<Self>(::core::marker::PhantomData))
                }
            }
        };
    }.into()
}
fn unsupported(name: &syn::Ident) -> TokenStream {
    syn::Error::new_spanned(name, "GNX I/O derives require a struct with named fields")
        .into_compile_error()
        .into()
}

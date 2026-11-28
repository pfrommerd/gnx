use super::*;
use serde_derive_internals::ast::Style;

pub(super) fn expand(c: &Container, g: &syn::Generics) -> syn::Result<Tokens> {
    let name = &c.ident;
    let wire = c.attrs.name().deserialize_name();
    let (imp, _, bounds) = g.split_for_impl();
    let (_, ty, _) = c.generics.split_for_impl();
    let path = c
        .attrs
        .remote()
        .filter(|_| !c.data.has_getter())
        .map(|p| quote!(#p))
        .unwrap_or_else(|| quote!(#name));
    let target = quote!(#path #ty);
    let expected = c.attrs.expecting().unwrap_or(wire);
    let mut helpers = Tokens::new();
    let mut payload = |style: Style, fs: &[Field], index: usize, variant: Option<&syn::Ident>| {
        let ctor = variant.map_or_else(|| quote!(#path), |v| quote!(#path::#v));
        let mut slots = Vec::new();
        let mut seq = Vec::new();
        let mut arms = Vec::new();
        let mut finish = Vec::new();
        let mut seed_values = Vec::new();
        let mut defaults = Vec::new();
        let mut keys = Vec::new();
        let mut primary = Vec::new();
        let container_default = match c.attrs.default() {
            Default::None => None,
            Default::Default => Some(quote!(::core::default::Default::default())),
            Default::Path(p) => Some(quote!(#p())),
        };
        let init = container_default
            .as_ref()
            .map(|x| quote!(let __default:#target=#x;));
        for (i, f) in fs.iter().enumerate() {
            let t = f.ty;
            let member = &f.member;
            let slot = format_ident!("__f{i}");
            let key = f.attrs.name().deserialize_name();
            let default = match f.attrs.default() {
                Default::Default => Some(quote!(::core::default::Default::default())),
                Default::Path(p) => Some(quote!(#p())),
                Default::None => container_default
                    .as_ref()
                    .map(|_| quote!(__default.#member)),
            };
            defaults.push(default.clone());
            if f.attrs.skip_deserializing() {
                let val = default.unwrap_or(quote!(::core::default::Default::default()));
                seq.push(quote!(let #slot=#val;));
                finish.push(quote!(let #slot=#val;));
                continue;
            }
            let seed = if let Some(hook) = f.attrs.deserialize_with() {
                let hook = if hook
                    .path
                    .segments
                    .iter()
                    .map(|s| s.ident.to_string())
                    .take(3)
                    .eq(["_serde", "__private", "de"].map(str::to_owned))
                {
                    let id = &hook.path.segments.last().unwrap().ident;
                    quote!(__io::borrow::#id)
                } else {
                    quote!(#hook)
                };
                let helper = format_ident!("__Seed{index}_{i}");
                helpers.extend(quote!{
                struct #helper<C>(::core::marker::PhantomData<C>);
                impl #imp __io::DeserializeSeed<'de> for #helper<#target> #bounds { type Value=#t; fn deserialize<__D:__io::Deserializer<'de>>(self,d:__D)->::core::result::Result<Self::Value,__D::Error>{ #hook(d) } }
            });
                quote!(#helper(::core::marker::PhantomData::<#target>))
            } else {
                quote!(::core::marker::PhantomData::<#t>)
            };
            seed_values.push((i, seed.clone()));
            let absent = default.clone().unwrap_or_else(|| {
                quote!(
                    return Err(__io::Error::custom(
                        concat!("missing sequence element: ",#key)
                    ))
                )
            });
            seq.push(
                quote!(let #slot=match a.next_element_seed(#seed)? { Some(v)=>v,None=>#absent };),
            );
            if f.attrs.flatten() {
                finish.push(quote!(let #slot=__io::DeserializeSeed::deserialize(#seed,__io::flat_input::FlatInput::<__A::Error>(&mut __extra,::core::marker::PhantomData))?;));
                continue;
            }
            let missing = default.unwrap_or_else(|| {
                if f.attrs.deserialize_with().is_some() {
                    quote!(return Err(__io::Error::custom(concat!("missing field: ",#key))))
                } else {
                    quote!(<#t as __io::Deserialize>::missing_field(#key)?)
                }
            });
            finish.push(quote!(let #slot=match #slot {Some(v)=>v,None=>#missing};));
            slots.push(slot.clone());
            keys.push(key);
            primary.push(key);
            keys.extend(
                f.attrs
                    .aliases()
                    .iter()
                    .map(String::as_str)
                    .filter(|a| *a != key),
            );
            let aliases = f.attrs.aliases().iter().filter(|a| a.as_str() != key);
            arms.push(quote!(#key #(| #aliases)* => { if #slot.is_some(){return Err(__io::Error::custom(concat!("duplicate field: ",#key)));} #slot=Some(a.next_value_seed(#seed)?); }));
        }
        let members = fs.iter().map(|f| &f.member);
        let vars: Vec<_> = (0..fs.len()).map(|i| format_ident!("__f{i}")).collect();
        let construct = quote!(#ctor { #(#members:#vars),* });
        let has_flat = fs.iter().any(|f| f.attrs.flatten());
        let n = slots.len();
        let vis = quote!(__Visitor::<#target,#index>(::core::marker::PhantomData));
        if c.attrs.transparent() {
            let vals = fs.iter().enumerate().map(|(i, f)| {
                if f.attrs.transparent() {
                    let seed = &seed_values.iter().find(|(j, _)| *j == i).unwrap().1;
                    quote!(__io::DeserializeSeed::deserialize(#seed,d)?)
                } else {
                    defaults[i]
                        .clone()
                        .unwrap_or(quote!(::core::default::Default::default()))
                }
            });
            let members = fs.iter().map(|f| &f.member);
            return quote!(Ok(#ctor {#(#members:#vals),*}));
        }
        let is_variant = variant.is_some();
        let (methods, call) = match style {
            Style::Unit | Style::Newtype if n == 0 => {
                let methods = quote!(fn visit_unit<__E:__io::Error>(self)->::core::result::Result<Self::Value,__E>{#init #(#seq)* Ok(#construct)});
                let call = if is_variant {
                    quote!({__io::VariantAccess::unit_variant(v)?; #init #(#seq)* Ok(#construct)})
                } else {
                    quote!(d.deserialize_unit_struct(#wire,#vis))
                };
                (methods, call)
            }
            Style::Newtype => {
                let seed = &seed_values[0].1;
                (
                    quote!(fn visit_newtype_struct<__D:__io::Deserializer<'de>>(self,d:__D)->::core::result::Result<Self::Value,__D::Error>{Ok(#ctor(__io::DeserializeSeed::deserialize(#seed,d)?))}),
                    if is_variant {
                        quote!(Ok(#ctor(__io::VariantAccess::newtype_variant_seed(v,#seed)?)))
                    } else {
                        quote!(d.deserialize_newtype_struct(#wire,#vis))
                    },
                )
            }
            Style::Unit => unreachable!(),
            Style::Tuple | Style::Struct => {
                let seq = quote!(fn visit_seq<__A:__io::SeqAccess<'de>>(self,mut a:__A)->::core::result::Result<Self::Value,__A::Error>{ #init #(#seq)* if a.next_element::<__io::IgnoredAny>()?.is_some(){return Err(__io::Error::custom("too many sequence elements"));} Ok(#construct) });
                if matches!(style, Style::Struct) {
                    let extra = has_flat.then(|| quote!(let mut __extra=::std::vec::Vec::new();));
                    let unknown = if has_flat {
                        quote!(
                            __extra.push(Some((__io::content::Content::Str(key), a.next_value()?)))
                        )
                    } else if c.attrs.deny_unknown_fields() {
                        quote!(return Err(__io::Error::custom(format!("unknown field: {}", key))))
                    } else {
                        quote!({
                            a.next_value::<__io::IgnoredAny>()?;
                        })
                    };
                    let leftover = (has_flat && c.attrs.deny_unknown_fields()).then(|| quote! {
                        if __extra.iter().any(|slot| slot.is_some()) { return Err(__io::Error::custom("unrecognized flattened field")); }
                    });
                    (
                        quote! {
                            #seq
                            fn visit_map<__A: __io::MapAccess<'de>>(self, mut a: __A) -> ::core::result::Result<Self::Value, __A::Error> {
                                #extra
                                #(let mut #slots = None;)*
                                while let Some(key) = a.next_key_seed(__io::KeySeed(&[#(#primary),*]))? {
                                    match key.as_ref() { #(#arms,)* _ => #unknown }
                                }
                                #init
                                #(#finish)*
                                #leftover
                                Ok(#construct)
                            }
                        },
                        if is_variant {
                            quote!(__io::VariantAccess::struct_variant(v,&[#(#keys),*],#vis))
                        } else if has_flat {
                            quote!(d.deserialize_map(#vis))
                        } else {
                            quote!(d.deserialize_struct(#wire,&[#(#keys),*],#vis))
                        },
                    )
                } else {
                    (
                        seq,
                        if is_variant {
                            quote!(__io::VariantAccess::tuple_variant(v,#n,#vis))
                        } else {
                            quote!(d.deserialize_tuple_struct(#wire,#n,#vis))
                        },
                    )
                }
            }
        };
        helpers.extend(quote!(impl #imp __io::DataVisitor<'de> for __Visitor<#target,#index> #bounds {type Value=#target; #methods}));
        call
    };
    let mut variant_helpers = Tokens::new();
    let body = if !matches!(
        c.attrs.identifier(),
        serde_derive_internals::attr::Identifier::No
    ) {
        super::identifier::body(c, &mut helpers, g)
    } else if let Some(t) = c.attrs.type_from() {
        quote!(Ok(<#t as __io::Deserialize>::deserialize(d)?.into()))
    } else if let Some(t) = c.attrs.type_try_from() {
        quote!(<#t as __io::Deserialize>::deserialize(d)?.try_into().map_err(__io::Error::custom))
    } else {
        match &c.data {
            Data::Struct(style, fs) => payload(*style, fs, 0, None),
            Data::Enum(vs) => {
                let mut arms = Vec::new();
                let mut names = Vec::new();
                let mut trials = Vec::new();
                let mut tagged = Vec::new();
                let mut tagged_other = None;
                let mut unknown = quote!(Err(__io::Error::custom(format!(
                    "unknown variant: {}",
                    key
                ))));
                for (i, v) in vs.iter().enumerate() {
                    if v.attrs.skip_deserializing() {
                        continue;
                    }
                    let key = v.attrs.name().deserialize_name();
                    let aliases = v.attrs.aliases().iter().filter(|a| a.as_str() != key);
                    let call = if let Some(hook) = v.attrs.deserialize_with() {
                        let seed = format_ident!("__VariantSeed{i}");
                        let id = &v.ident;
                        let vars: Vec<_> = (0..v.fields.len())
                            .map(|j| format_ident!("__f{j}"))
                            .collect();
                        let members = v.fields.iter().map(|f| &f.member);
                        variant_helpers.extend(quote!(struct #seed<C>(::core::marker::PhantomData<C>);impl #imp __io::DeserializeSeed<'de> for #seed<#target> #bounds{type Value=#target;fn deserialize<__D:__io::Deserializer<'de>>(self,d:__D)->::core::result::Result<Self::Value,__D::Error>{let (#(#vars),*)=#hook(d)?;Ok(#path::#id{#(#members:#vars),*})}}));
                        quote!(__io::VariantAccess::newtype_variant_seed(v,#seed::<#target>(::core::marker::PhantomData)))
                    } else {
                        payload(v.style, &v.fields, i + 1, Some(&v.ident))
                    };
                    if v.attrs.untagged() || matches!(c.attrs.tag(), TagType::None) {
                        trials.push(quote!({let v=__io::content::Payload::<__D::Error>(Some(&__input),::core::marker::PhantomData);if let Ok(value)=(|| -> ::core::result::Result<#target,__D::Error>{#call})(){return Ok(value);}}));
                        continue;
                    }
                    let aliases: Vec<_> = aliases.collect();
                    let pat = quote!(#key #(|#aliases)*);
                    let unit = matches!(v.style, Style::Unit);
                    let prepare = if unit && matches!(c.attrs.tag(), TagType::Internal { .. }) {
                        quote!(None)
                    } else {
                        quote!(__data.as_ref())
                    };
                    let direct = quote!({let v=__io::content::Payload::<__D::Error>(#prepare,::core::marker::PhantomData);#call});
                    if v.attrs.other() {
                        unknown = call.clone();
                        tagged_other = Some(direct.clone());
                    }
                    tagged.push(quote!(#pat=>#direct));
                    arms.push(quote!(#pat => #call));
                    names.push(key);
                }
                helpers.extend(quote!(impl #imp __io::DataVisitor<'de> for __Visitor<#target,0> #bounds {type Value=#target; fn visit_enum<__A:__io::EnumAccess<'de>>(self,a:__A)->::core::result::Result<Self::Value,__A::Error>{let (key,v)=a.variant_seed(__io::KeySeed(&[#(#names),*]))?;match key.as_ref(){#(#arms,)* _=>#unknown}}}));
                let recognized = vs
                    .iter()
                    .filter(|v| !v.attrs.skip_deserializing() && !v.attrs.untagged())
                    .flat_map(|v| {
                        std::iter::once(v.attrs.name().deserialize_name()).chain(
                            v.attrs
                                .aliases()
                                .iter()
                                .map(String::as_str)
                                .filter(|name| *name != v.attrs.name().deserialize_name()),
                        )
                    });
                let external = quote!(d.deserialize_enum(#wire,&[#(#recognized),*],__Visitor::<#target,0>(::core::marker::PhantomData)));
                if matches!(c.attrs.tag(), TagType::External) && trials.is_empty() {
                    external
                } else {
                    let dispatch = match c.attrs.tag() {
                        TagType::None => None,
                        TagType::External => {
                            Some(quote!({let d=__input.replay::<__D::Error>();#external}))
                        }
                        TagType::Internal { tag } | TagType::Adjacent { tag, .. } => {
                            let data = if let TagType::Adjacent { content, .. } = c.attrs.tag() {
                                let check=c.attrs.deny_unknown_fields().then(||quote!(if matches!(&input,__io::content::Content::Map(v) if !v.is_empty()){return Err(__io::Error::custom("unknown field in tagged enum"));}));
                                quote!({let data=input.take_field::<__D::Error>(#content)?;#check data})
                            } else {
                                quote!(Some(input))
                            };
                            let fallback = tagged_other.unwrap_or(quote!(Err(
                                __io::Error::custom(format!("unknown variant: {}", key))
                            )));
                            Some(
                                quote!({let mut input=__input;let key=match input.take_field::<__D::Error>(#tag)? {Some(__io::content::Content::Str(k))=>k,_=>return Err(__io::Error::custom(concat!("missing or invalid tag: ",#tag)))};let __data=#data;match key.as_ref(){#(#tagged,)* _=>#fallback}}),
                            )
                        }
                    };
                    let read = quote!(let __input=<__io::content::Content as __io::Deserialize>::deserialize(d)?;);
                    if trials.is_empty() {
                        quote!({#read #dispatch})
                    } else {
                        let attempt=dispatch.map(|expr|quote!({let __input=__input.clone();if let Ok(x)=(||->::core::result::Result<#target,__D::Error>{#expr})(){return Ok(x);}}));
                        quote!({#read #attempt #(#trials)* Err(__io::Error::custom(concat!("no matching variant for ",#wire)))})
                    }
                }
            }
        }
    };
    let implementation = if let Some(remote) = c.attrs.remote() {
        let vis = &c.original.vis;
        let body = if c.data.has_getter() {
            quote!((||->::core::result::Result<#target,__D::Error>{#body})().map(Into::into))
        } else {
            body
        };
        quote!(impl #imp #name #ty #bounds {#vis fn deserialize<__D:__io::Deserializer<'de>>(d:__D)->::core::result::Result<#remote #ty,__D::Error>{#body}})
    } else {
        quote!(impl #imp __io::Deserialize<'de> for #target #bounds {fn deserialize<__D:__io::Deserializer<'de>>(d:__D)->::core::result::Result<Self,__D::Error>{#body}})
    };
    Ok(
        quote!(struct __Visitor<T,const I:usize>(::core::marker::PhantomData<T>);impl<T,const I:usize> __io::Expecting for __Visitor<T,I>{fn expected(&self,f:&mut ::core::fmt::Formatter)->::core::fmt::Result{f.write_str(#expected)}} #helpers #variant_helpers #implementation),
    )
}

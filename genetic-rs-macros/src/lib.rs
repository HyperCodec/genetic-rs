extern crate proc_macro;

use darling::util::PathList;
use darling::FromAttributes;
use darling::FromMeta;
use proc_macro::TokenStream;
use quote::quote;
use quote::quote_spanned;
use quote::ToTokens;
use syn::parse_quote;
use syn::spanned::Spanned;
use syn::{parse_macro_input, Data, DeriveInput};

#[proc_macro_derive(RandomlyMutable, attributes(randmut))]
pub fn randmut_derive(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);

    let (def_ctx, mut ctx_ident) =
        create_context_helper(&ast, parse_quote!(RandomlyMutable), parse_quote!(randmut));
    let custom_context = ctx_ident.is_some();

    let name = ast.ident;

    match ast.data {
        Data::Struct(s) => {
            let mut inner = Vec::new();

            for (i, field) in s.fields.into_iter().enumerate() {
                let ty = field.ty;
                let span = ty.span();

                if ctx_ident.is_none() {
                    ctx_ident = Some(
                        quote_spanned! {span=> <#ty as genetic_rs_common::prelude::RandomlyMutable>::Context },
                    );
                }

                if let Some(field_name) = field.ident {
                    if custom_context {
                        inner.push(quote_spanned! {span=>
                            <#ty as genetic_rs_common::prelude::RandomlyMutable>::mutate(&mut self.#field_name, &ctx.#field_name, rate, rng);
                        });
                    } else {
                        inner.push(quote_spanned! {span=>
                            <#ty as genetic_rs_common::prelude::RandomlyMutable>::mutate(&mut self.#field_name, ctx, rate, rng);
                        });
                    }
                } else if custom_context {
                    inner.push(quote_spanned! {span=>
                        <#ty as genetic_rs_common::prelude::RandomlyMutable>::mutate(&mut self.#i, &ctx.#i, rate, rng);
                    });
                } else {
                    inner.push(quote_spanned! {span=>
                        <#ty as genetic_rs_common::prelude::RandomlyMutable>::mutate(&mut self.#i, ctx, rate, rng);
                    });
                }
            }

            let inner: proc_macro2::TokenStream = inner.into_iter().collect();

            quote! {
                #[automatically_derived]
                impl genetic_rs_common::prelude::RandomlyMutable for #name {
                    type Context = #ctx_ident;

                    fn mutate(&mut self, ctx: &Self::Context, rate: f32, rng: &mut impl rand::Rng) {
                        #inner
                    }
                }

                #def_ctx
            }
            .into()
        }
        Data::Enum(_e) => {
            panic!("enums not yet supported");
        }
        Data::Union(_u) => {
            panic!("unions not yet supported");
        }
    }
}

#[derive(FromAttributes)]
#[darling(attributes(mitosis))]
struct MitosisSettings {
    use_randmut: Option<bool>,

    // darling is annoyingly restrictive and doesn't
    // let me just ignore extra fields
    #[darling(rename = "create_context")]
    _create_context: Option<CreateContext>,

    #[darling(rename = "with_context")]
    _with_context: Option<syn::Path>,
}

#[proc_macro_derive(Mitosis, attributes(mitosis))]
pub fn mitosis_derive(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);

    let name = &ast.ident;

    let mitosis_settings = MitosisSettings::from_attributes(&ast.attrs).unwrap();
    if mitosis_settings.use_randmut.is_some() && mitosis_settings.use_randmut.unwrap() {
        quote! {
            #[automatically_derived]
            impl genetic_rs_common::prelude::Mitosis for #name {
                type Context = <Self as genetic_rs_common::prelude::RandomlyMutable>::Context;

                fn divide(&self, ctx: &Self::Context, rate: f32, rng: &mut impl rand::Rng) -> Self {
                    let mut child = self.clone();
                    <Self as genetic_rs_common::prelude::RandomlyMutable>::mutate(&mut child, ctx, rate, rng);
                    child
                }
            }
        }
        .into()
    } else {
        let (def_ctx, mut ctx_ident) =
            create_context_helper(&ast, parse_quote!(RandomlyMutable), parse_quote!(mitosis));
        let custom_context = ctx_ident.is_some();

        let name = ast.ident;

        match ast.data {
            Data::Struct(s) => {
                let mut is_tuple_struct = false;
                let mut inner = Vec::new();

                for (i, field) in s.fields.into_iter().enumerate() {
                    let ty = field.ty;
                    let span = ty.span();

                    if ctx_ident.is_none() {
                        ctx_ident = Some(
                            quote_spanned! {span=> <#ty as genetic_rs_common::prelude::Mitosis>::Context },
                        );
                    }

                    if let Some(field_name) = field.ident {
                        if custom_context {
                            inner.push(quote_spanned! {span=>
                                #field_name: <#ty as genetic_rs_common::prelude::Mitosis>::divide(&self.#field_name, &ctx.#field_name, rate, rng),
                            });
                        } else {
                            inner.push(quote_spanned! {span=>
                                #field_name: <#ty as genetic_rs_common::prelude::Mitosis>::divide(&self.#field_name, ctx, rate, rng),
                            });
                        }
                    } else if custom_context {
                        is_tuple_struct = true;
                        inner.push(quote_spanned! {span=>
                            <#ty as genetic_rs_common::prelude::Mitosis>::divide(&self.#i, &ctx.#i, rate, rng),
                        });
                    } else {
                        is_tuple_struct = true;
                        inner.push(quote_spanned! {span=>
                            <#ty as genetic_rs_common::prelude::Mitosis>::divide(&self.#i, ctx, rate, rng),
                        });
                    }
                }

                let inner: proc_macro2::TokenStream = inner.into_iter().collect();
                let child = if is_tuple_struct {
                    quote! {
                        Self(#inner)
                    }
                } else {
                    quote! {
                        Self {
                            #inner
                        }
                    }
                };

                quote! {
                    #[automatically_derived]
                    impl genetic_rs_common::prelude::Mitosis for #name {
                        type Context = #ctx_ident;

                        fn divide(&self, ctx: &Self::Context, rate: f32, rng: &mut impl rand::Rng) -> Self {
                            #child
                        }
                    }

                    #def_ctx
                }
                .into()
            }
            Data::Enum(_e) => {
                panic!("enums not yet supported");
            }
            Data::Union(_u) => {
                panic!("unions not yet supported");
            }
        }
    }
}

#[derive(FromMeta)]
struct ContextArgs {
    with_context: Option<syn::Path>,
    create_context: Option<CreateContext>,
}

#[derive(FromMeta)]
struct CreateContext {
    name: syn::Ident,
    derive: Option<PathList>,
}

fn create_context_helper(
    ast: &DeriveInput,
    trait_name: syn::Ident,
    attr_path: syn::Path,
) -> (
    Option<proc_macro2::TokenStream>,
    Option<proc_macro2::TokenStream>,
) {
    let name = &ast.ident;
    let doc =
        quote! { #[doc = concat!("Autogenerated context struct for [`", stringify!(#name), "`]")] };

    let vis = ast.vis.to_token_stream();

    let attr = ast.attrs.iter().find(|a| a.path() == &attr_path);
    if attr.is_none() {
        return (None, None);
    }

    let meta = &attr.unwrap().meta;

    let args = ContextArgs::from_meta(meta).unwrap();

    if args.create_context.is_some() && args.with_context.is_some() {
        panic!("cannot have both create_context and with_context");
    }

    if let Some(create_ctx) = args.create_context {
        let ident = &create_ctx.name;
        let derives = create_ctx.derive.map(|paths| {
            quote! {
                #[derive(#(#paths,)*)]
            }
        });

        match &ast.data {
            Data::Struct(s) => {
                let mut inner = Vec::<proc_macro2::TokenStream>::new();
                let mut tuple_struct = false;

                for field in &s.fields {
                    let ty = &field.ty;
                    let ty_span = ty.span();

                    if let Some(field_name) = &field.ident {
                        inner.push(quote_spanned! {ty_span=>
                            #vis #field_name: <#ty as genetic_rs_common::prelude::#trait_name>::Context,
                        });
                    } else {
                        tuple_struct = true;
                        inner.push(quote_spanned! {ty_span=>
                            #vis <#ty as genetic_rs_common::prelude::#trait_name>::Context,
                        });
                    }
                }

                let inner: proc_macro2::TokenStream = inner.into_iter().collect();

                if tuple_struct {
                    return (
                        Some(quote! { #doc #derives #vis struct #ident (#inner);}),
                        Some(ident.to_token_stream()),
                    );
                }

                return (
                    Some(quote! { #doc #derives #vis struct #ident {#inner};}),
                    Some(ident.to_token_stream()),
                );
            }
            Data::Enum(_) => panic!("enums not supported"),
            Data::Union(_) => panic!("unions not supported"),
        }
    }

    if let Some(ident) = args.with_context {
        return (None, Some(ident.to_token_stream()));
    }

    (None, None)
}

#[cfg(feature = "crossover")]
#[proc_macro_derive(Crossover, attributes(crossover))]
pub fn crossover_derive(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);

    let (def_ctx, mut context) =
        create_context_helper(&ast, parse_quote!(Crossover), parse_quote!(crossover));
    let custom_context = context.is_some();

    let name = ast.ident;

    match ast.data {
        Data::Struct(s) => {
            let mut inner = Vec::new();
            let mut tuple_struct = false;

            for (i, field) in s.fields.into_iter().enumerate() {
                let ty = field.ty;
                let span = ty.span();

                if context.is_none() {
                    context =
                        Some(quote! { <#ty as genetic_rs_common::prelude::Crossover>::Context });
                }

                if let Some(field_name) = field.ident {
                    if custom_context {
                        inner.push(quote_spanned! {span=>
                            #field_name: <#ty as genetic_rs_common::prelude::Crossover>::crossover(&self.#field_name, &other.#field_name, &ctx.#field_name, rate, rng),
                        });
                    } else {
                        inner.push(quote_spanned! {span=>
                            #field_name: <#ty as genetic_rs_common::prelude::Crossover>::crossover(&self.#field_name, &other.#field_name, ctx, rate, rng),
                        });
                    }
                } else {
                    tuple_struct = true;

                    if custom_context {
                        inner.push(quote_spanned! {span=>
                            <#ty as genetic_rs_common::prelude::Crossover>::crossover(&self.#i, &other.#i, &ctx.#i, rate, rng),
                        })
                    } else {
                        inner.push(quote_spanned! {span=>
                            <#ty as genetic_rs_common::prelude::Crossover>::crossover(&self.#i, &other.#i, ctx, rate, rng),
                        });
                    }
                }
            }

            let inner: proc_macro2::TokenStream = inner.into_iter().collect();

            if tuple_struct {
                quote! {
                    #def_ctx

                    #[automatically_derived]
                    impl genetic_rs_common::prelude::Crossover for #name {
                        type Context = #context;

                        fn crossover(&self, other: &Self, ctx: &Self::Context, rate: f32, rng: &mut impl rand::Rng) -> Self {
                            Self(#inner)
                        }
                    }
                }.into()
            } else {
                quote! {
                    #def_ctx

                    #[automatically_derived]
                    impl genetic_rs_common::prelude::Crossover for #name {
                        type Context = #context;

                        fn crossover(&self, other: &Self, ctx: &Self::Context, rate: f32, rng: &mut impl rand::Rng) -> Self {
                            Self {
                                #inner
                            }
                        }
                    }
                }.into()
            }
        }
        Data::Enum(_e) => {
            panic!("enums not yet supported");
        }
        Data::Union(_u) => {
            panic!("unions not yet supported");
        }
    }
}

#[cfg(feature = "genrand")]
#[proc_macro_derive(GenerateRandom)]
pub fn genrand_derive(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);

    let name = ast.ident;

    match ast.data {
        Data::Struct(s) => {
            let mut inner = Vec::new();
            let mut tuple_struct = false;

            for field in s.fields {
                let ty = field.ty;
                let span = ty.span();

                if let Some(field_name) = field.ident {
                    inner.push(quote_spanned! {span=>
                        #field_name: <#ty as genetic_rs_common::prelude::GenerateRandom>::gen_random(rng),
                    });
                } else {
                    tuple_struct = true;
                    inner.push(quote_spanned! {span=>
                        <#ty as genetic_rs_common::prelude::GenerateRandom>::gen_random(rng),
                    });
                }
            }

            let inner: proc_macro2::TokenStream = inner.into_iter().collect();
            if tuple_struct {
                quote! {
                    #[automatically_derived]
                    impl genetic_rs_common::prelude::GenerateRandom for #name {
                        fn gen_random(rng: &mut impl rand::Rng) -> Self {
                            Self(#inner)
                        }
                    }
                }
                .into()
            } else {
                quote! {
                    #[automatically_derived]
                    impl genetic_rs_common::prelude::GenerateRandom for #name {
                        fn gen_random(rng: &mut impl rand::Rng) -> Self {
                            Self {
                                #inner
                            }
                        }
                    }
                }
                .into()
            }
        }
        Data::Enum(_e) => {
            panic!("enums not yet supported");
        }
        Data::Union(_u) => {
            panic!("unions not yet supported");
        }
    }
}

extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;
use quote::quote_spanned;
use syn::spanned::Spanned;
use syn::{parse_macro_input, Data, DeriveInput};

#[proc_macro_derive(RandomlyMutable)]
pub fn randmut_derive(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);

    let name = ast.ident;

    match ast.data {
        Data::Struct(s) => {
            let mut inner = Vec::new();

            for (i, field) in s.fields.into_iter().enumerate() {
                let ty = field.ty;
                let span = ty.span();

                if let Some(field_name) = field.ident {
                    inner.push(quote_spanned! {span=>
                        <#ty as genetic_rs_common::prelude::RandomlyMutable>::mutate(&mut self.#field_name, rate, rng);
                    });
                } else {
                    inner.push(quote_spanned! {span=>
                        <#ty as genetic_rs_common::prelude::RandomlyMutable>::mutate(&mut self.#i, rate, rng);
                    });
                }
            }

            let inner: proc_macro2::TokenStream = inner.into_iter().collect();

            quote! {
                impl genetic_rs_common::prelude::RandomlyMutable for #name {
                    fn mutate(&mut self, rate: f32, rng: &mut impl rand::Rng) {
                        #inner
                    }
                }
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

#[proc_macro_derive(Mitosis)]
pub fn mitosis_derive(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);
    let name = &ast.ident;
    let span = name.span();

    quote_spanned! {span=>
        impl genetic_rs_common::prelude::Mitosis for #name {
            type Context = <Self as genetic_rs_common::prelude::RandomlyMutable>::Context;

            fn mitosis(&self, rate: f32, rng: &mut impl rand::Rng, ctx: Self::Context) -> Self {
                let mut child = self.clone();
                <Self as genetic_rs_common::prelude::RandomlyMutable>::mutate(&mut child, rate, ctx, rng);
                child
            }
        }
    }
    .into()
}

#[cfg(feature = "crossover")]
#[proc_macro_derive(Crossover)]
pub fn crossover_derive(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);

    let name = ast.ident;

    match ast.data {
        Data::Struct(s) => {
            let mut inner = Vec::new();
            let mut tuple_struct = false;

            for (i, field) in s.fields.into_iter().enumerate() {
                let ty = field.ty;
                let span = ty.span();

                if let Some(field_name) = field.ident {
                    inner.push(quote_spanned! {span=>
                        #field_name: <#ty as genetic_rs_common::prelude::Crossover>::crossover(&self.#field_name, &other.#field_name, rate, rng),
                    });
                } else {
                    tuple_struct = true;
                    inner.push(quote_spanned! {span=>
                        <#ty as genetic_rs_common::prelude::Crossover>::crossover(&self.#i, &other.#i, rate, rng),
                    });
                }
            }

            let inner: proc_macro2::TokenStream = inner.into_iter().collect();

            if tuple_struct {
                quote! {
                    impl genetic_rs_common::prelude::Crossover for #name {
                        fn crossover(&self, other: &Self, rate: f32, rng: &mut impl rand::Rng) -> Self {
                            Self(#inner)
                        }
                    }
                }.into()
            } else {
                quote! {
                    impl genetic_rs_common::prelude::Crossover for #name {
                        fn crossover(&self, other: &Self, rate: f32, rng: &mut impl rand::Rng) -> Self {
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
                    impl genetic_rs_common::prelude::GenerateRandom for #name {
                        fn gen_random(rng: &mut impl rand::Rng) -> Self {
                            Self(#inner)
                        }
                    }
                }
                .into()
            } else {
                quote! {
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

extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Fields};
use quote::quote_spanned;
use syn::spanned::Spanned;

#[proc_macro_derive(RandomlyMutable)]
pub fn randmut_derive(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);

    let mut inner_mutate = quote!();

    if let Data::Struct(data) = ast.data {
        match &data.fields {
            Fields::Named(named) => {
                for field in named.named.iter() {
                    let name = field.ident.clone().unwrap();
                    inner_mutate
                        .extend(quote!(genetic_rs_common::prelude::RandomlyMutable::mutate(&mut self.#name, rate, rng);));
                }
            }
            _ => unimplemented!(),
        }
    } else {
        panic!("Cannot derive RandomlyMutable for an enum.");
    }

    let name = &ast.ident;
    quote! {
        impl genetic_rs_common::prelude::RandomlyMutable for #name {
            fn mutate(&mut self, rate: f32, rng: &mut impl genetic_rs_common::Rng) {
                #inner_mutate
            }
        }
    }
    .into()
}

#[proc_macro_derive(DivisionReproduction)]
pub fn divrepr_derive(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);
    let name = &ast.ident;

    quote! {
        impl genetic_rs_common::prelude::DivisionReproduction for #name {}
    }
    .into()
}

#[cfg(feature = "crossover")]
#[proc_macro_derive(CrossoverReproduction)]
pub fn cross_repr_derive(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);

    let mut inner_crossover_return = quote!();

    if let Data::Struct(data) = ast.data {
        match &data.fields {
            Fields::Named(named) => {
                for field in named.named.iter() {
                    let name = field.ident.clone().unwrap();
                    inner_crossover_return.extend(quote!(#name: genetic_rs_common::prelude::CrossoverReproduction::crossover(&self.#name, &other.#name, rate, rng),));
                }
            }
            _ => unimplemented!(),
        }
    } else {
        panic!("Cannot derive CrossoverReproduction for an enum.");
    }

    let name = &ast.ident;

    quote! {
        impl genetic_rs_common::prelude::CrossoverReproduction for #name {
            fn crossover(&self, other: &Self, rate: f32, rng: &mut impl genetic_rs_common::Rng) -> Self {
                Self { #inner_crossover_return }
            }
        }
    }
    .into()
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
                        #field_name: <#ty as GenerateRandom>::gen_random(rng),
                    });
                } else {
                    tuple_struct = true;
                    inner.push(quote_spanned! {span=>
                        <#ty as GenerateRandom>::gen_random(rng),
                    });
                }
            }

            let inner: proc_macro2::TokenStream = inner.into_iter().collect();
            if tuple_struct {
                quote! {
                    impl GenerateRandom for #name {
                        fn gen_random(rng: &mut impl rand::Rng) -> Self {
                            Self(#inner)
                        }
                    }
                }.into()
            } else {
                quote! {
                    impl GenerateRandom for #name {
                        fn gen_random(rng: &mut impl rand::Rng) -> Self {
                            Self {
                                #inner
                            }
                        }
                    }
                }.into()
            }
        },
        Data::Enum(_e) => {
            panic!("enums not yet supported");
        },
        Data::Union(_u) => {
            panic!("unions not yet supported");
        }
    }
}

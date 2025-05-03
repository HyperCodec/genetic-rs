extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Fields};

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
                    inner_crossover_return.extend(quote!(#name: genetic_rs_common::prelude::CrossoverReproduction::crossover(&self.#name, &other.#name, rng),));
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
            fn crossover(&self, other: &Self, rng: &mut impl genetic_rs_common::Rng) -> Self {
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

    let mut genrand_inner_return = quote!();

    if let Data::Struct(data) = ast.data {
        match &data.fields {
            Fields::Named(named) => {
                for field in named.named.iter() {
                    let name = field.ident.clone().unwrap();
                    let ty = field.ty.clone();
                    genrand_inner_return
                        .extend(quote!(#name: <#ty as genetic_rs_common::prelude::GenerateRandom>::gen_random(rng),));
                }
            }
            _ => unimplemented!(),
        }
    }

    let name = &ast.ident;
    quote! {
        impl GenerateRandom for #name {
            fn gen_random(rng: &mut impl genetic_rs_common::Rng) -> Self {
                Self {
                    #genrand_inner_return
                }
            }
        }
    }
    .into()
}

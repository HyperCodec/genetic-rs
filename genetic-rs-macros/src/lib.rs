extern crate proc_macro;

use genetic_rs_common::prelude::*;
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Fields};

#[proc_macro_derive(RandomlyMutable)]
pub fn randmut_derive(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);

    let mut inner_mutate = quote!();

    if let Data::Struct(data) = ast.data {
        match &data.fields {
            Fields::Named(named) => for field in named.named.iter() {
                let name = field.ident.clone().unwrap();
                inner_mutate.extend(quote!(self.#name.mutate(rate, rng);));
            }
            _ => unimplemented!(),
        }
    
    }

    let name = &ast.ident;
    quote! {
        impl RandomlyMutable for #name {
            fn mutate(&mut self, rate: f32, rng: &mut impl rand::Rng) {
                #inner_mutate
            }
        }
    }.into()
}

#[proc_macro_derive(DivisionReproduction)]
pub fn divrepr_derive(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);
    
    let mut inner_divide_return = quote!();

    if let Data::Struct(data) = ast.data {
        match &data.fields {
            Fields::Named(named) => for field in named.named.iter() {
                let name = field.ident.clone().unwrap();
                inner_divide_return.extend(quote!(#name: self.#name.divide(rng),));
            },
            _ => unimplemented!()
        }
    }

    let name = &ast.ident;

    quote! {
        impl DivisionReproduction for #name {
            fn divide(&self, &mut impl rand::Rng) -> Self {
                Self {
                    #inner_divide_return
                }
            }
        }
    }.into()
}

#[proc_macro_derive(Prunable)]
pub fn prunable_derive(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);

    let mut inner_despawn = quote!();

    if let Data::Struct(data) = ast.data {
        match &data.fields {
            Fields::Named(named) => for field in named.named.iter() {
                let name = field.ident.clone().unwrap();
                inner_despawn.extend(quote!(self.#name.despawn();));
            },
            _ => unimplemented!()
        }
    }

    let name = &ast.ident;

    quote! {
        impl Prunable for #name {
            fn despawn(self) {
                #inner_despawn
            }
        }
    }.into()
}
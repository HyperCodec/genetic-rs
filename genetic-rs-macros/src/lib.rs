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
            Fields::Named(named) => for field in named.named.iter() {
                let name = field.ident.clone().unwrap();
                inner_mutate.extend(quote!(<self.#name as RandomlyMutable>::mutate(rate, rng);));
            }
            _ => unimplemented!(),
        }
    
    } else {
        panic!("Cannot derive RandomlyMutable for an enum.");
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
                inner_divide_return.extend(quote!(#name: <self.#name as DivisionReproduction>::divide(rng),));
            },
            _ => unimplemented!()
        }
    } else {
        panic!("Cannot derive DivisionReproduction for an enum.");
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

#[cfg(feature = "crossover")]
#[proc_macro_derive(CrossoverReproduction)]
pub fn cross_repr_derive(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);

    let mut inner_crossover_return = quote!();

    if let Data::Struct(data) = ast.data {
        match &data.fields {
            Fields::Named(named) => for field in named.named.iter() {
                let name = field.ident.clone().unwrap();
                inner_crossover_return.extend(quote!(#name: <self.#name as CrossoverReproduction>::crossover(&other.#name),));
            },
            _ => unimplemented!(),
        }
    } else {
        panic!("Cannot derive CrossoverReproduction for an enum.");
    }


    // doing it this way to avoid 'mutation_rate possibly uninitialized'
    let mutation_rate = || -> proc_macro2::TokenStream {
        for attr in ast.attrs.iter() {
            if attr.path().is_ident("mutation_rate") {
                return attr.parse_args().expect("Failed to parse `mutation_rate` attr");
            }
        }
        panic!("Missing `mutation_rate` attr");
    }();
    

    let name = &ast.ident;

    quote! {
        impl CrossoverReproduction for #name {
            fn crossover(&self, other: &Self, rng: &mut impl rand::Rng) -> Self {
                let mut child = Self { #inner_crossover_return };
                child.mutate(#mutation_rate, rng)
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
                inner_despawn.extend(quote!(<self.#name as Prunable>::despawn();));
            },
            _ => unimplemented!()
        }
    } else {
        panic!("Cannot derive Prunable for an enum.");
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
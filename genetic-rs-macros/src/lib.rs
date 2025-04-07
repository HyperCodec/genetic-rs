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
                        .extend(quote!(RandomlyMutable::mutate(&mut self.#name, rate, rng);));
                }
            }
            _ => unimplemented!(),
        }
    } else {
        panic!("Cannot derive RandomlyMutable for an enum.");
    }

    let name = &ast.ident;
    quote! {
        impl RandomlyMutable for #name {
            fn mutate(&mut self, rate: f32, rng: &mut impl Rng) {
                #inner_mutate
            }
        }
    }
    .into()
}

#[proc_macro_derive(DivisionReproduction)]
pub fn divrepr_derive(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);

    let mut inner_divide_return = quote!();

    if let Data::Struct(data) = ast.data {
        match &data.fields {
            Fields::Named(named) => {
                for field in named.named.iter() {
                    let name = field.ident.clone().unwrap();
                    inner_divide_return
                        .extend(quote!(#name: DivisionReproduction::divide(&self.#name, rng),));
                }
            }
            _ => unimplemented!(),
        }
    } else {
        panic!("Cannot derive DivisionReproduction for an enum.");
    }

    let name = &ast.ident;

    quote! {
        impl DivisionReproduction for #name {
            fn divide(&self, rng: &mut impl Rng) -> Self {
                Self {
                    #inner_divide_return
                }
            }
        }
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
                    inner_crossover_return.extend(quote!(#name: CrossoverReproduction::crossover(&self.#name, &other.#name, rng),));
                }
            }
            _ => unimplemented!(),
        }
    } else {
        panic!("Cannot derive CrossoverReproduction for an enum.");
    }

    let name = &ast.ident;

    quote! {
        impl CrossoverReproduction for #name {
            fn crossover(&self, other: &Self, rng: &mut impl Rng) -> Self {
                Self { #inner_crossover_return }
            }
        }
    }
    .into()
}

#[proc_macro_derive(Prunable)]
pub fn prunable_derive(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);

    let mut inner_despawn = quote!();

    if let Data::Struct(data) = ast.data {
        match &data.fields {
            Fields::Named(named) => {
                for field in named.named.iter() {
                    let name = field.ident.clone().unwrap();
                    inner_despawn.extend(quote!(Prunable::despawn(self.#name);));
                }
            }
            _ => unimplemented!(),
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
                        .extend(quote!(#name: <#ty as GenerateRandom>::gen_random(rng),));
                }
            }
            _ => unimplemented!(),
        }
    }

    let name = &ast.ident;
    quote! {
        impl GenerateRandom for #name {
            fn gen_random(rng: &mut impl Rng) -> Self {
                Self {
                    #genrand_inner_return
                }
            }
        }
    }
    .into()
}

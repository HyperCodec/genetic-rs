#![allow(dead_code)]

use genetic_rs::prelude::*;

#[derive(Clone, Debug, Default)]
struct Context1;

#[derive(Clone, Debug, Default)]
struct Context2;

#[derive(Clone, Mitosis, Debug)]
#[mitosis(use_randmut = true)]
struct Foo1(f32);

impl RandomlyMutable for Foo1 {
    type Context = Context1;

    fn mutate(&mut self, _ctx: &Self::Context, rate: f32, rng: &mut impl rand::Rng) {
        self.0 += rng.random_range(-rate..rate);
    }
}

#[derive(Clone, Mitosis, Debug)]
#[mitosis(use_randmut = true)]
struct Foo2(f32);

impl RandomlyMutable for Foo2 {
    type Context = Context2;

    fn mutate(&mut self, _ctx: &Self::Context, rate: f32, rng: &mut impl rand::Rng) {
        self.0 += rng.random_range(-rate..rate);
    }
}

#[derive(Clone, RandomlyMutable, Mitosis, Debug)]
#[randmut(create_context(name = BarCtx, derive(Clone, Debug, Default)))]
#[mitosis(with_context = BarCtx)]
struct Bar {
    a: Foo1,
    b: Foo2,
}

fn main() {
    println!("check `cargo expand --package genetic-rs --example derive --all-features`");
}

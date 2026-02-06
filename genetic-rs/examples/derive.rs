use genetic_rs::prelude::*;

struct Context1;
struct Context2;

#[derive(Clone, Mitosis, Debug)]
struct Foo1(f32);

impl RandomlyMutable for Foo1 {
    type Context = Context1;

    fn mutate(&mut self, _ctx: &Self::Context, rate: f32, rng: &mut impl rand::Rng) {
        self.0 += rng.random_range(-rate..rate);
    }
}

#[derive(Clone, Mitosis, Debug)]
struct Foo2(f32);

impl RandomlyMutable for Foo2 {
    type Context = Context2;

    fn mutate(&mut self, _ctx: &Self::Context, rate: f32, rng: &mut impl rand::Rng) {
        self.0 += rng.random_range(-rate..rate);
    }
}

#[derive(Clone, RandomlyMutable, Mitosis, Debug)]
#[randmut(create_context = BarCtx)]
struct Bar {
    a: Foo1,
    b: Foo2,
}

fn main() {
    println!("check `cargo expand --package genetic-rs --example derive --all-features`");
}

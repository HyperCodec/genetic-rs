use genetic_rs::prelude::*;

#[derive(Clone, Debug)]
struct TestGene {
    a: f32,
}

impl RandomlyMutable for TestGene {
    type Context = ();

    fn mutate(&mut self, _: &(), rate: f32, rng: &mut impl Rng) {
        self.a += rng.random_range(-1.0..1.0) * rate;
    }
}

#[cfg(feature = "crossover")]
impl Crossover for TestGene {
    type Context = ();

    fn crossover(&self, other: &Self, _: &(), rate: f32, rng: &mut impl Rng) -> Self {
        let mut child = Self {
            a: self.a + other.a / 2.0,
        };
        child.mutate(&(), rate, rng);
        child
    }
}

impl GenerateRandom for TestGene {
    fn gen_random(rng: &mut impl Rng) -> Self {
        Self {
            a: rng.random_range(-1.0..1.0),
        }
    }
}

// using the derive macros here is only useful if the fields are not related to each other
#[derive(RandomlyMutable, Mitosis, GenerateRandom, Clone, Debug)]
#[cfg_attr(feature = "crossover", derive(Crossover))]
struct MyDNA {
    g1: TestGene,
    g2: TestGene,
}

fn main() {
    println!("check `cargo expand --package genetic-rs --example derive --all-features`");
}

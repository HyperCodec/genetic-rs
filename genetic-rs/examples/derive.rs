use genetic_rs::prelude::*;

#[derive(Clone, Debug)]
struct TestGene {
    a: f32,
}

impl RandomlyMutable for TestGene {
    fn mutate(&mut self, rate: f32, rng: &mut impl Rng) {
        self.a += rng.random_range(-1.0..1.0) * rate;
    }
}

#[cfg(feature = "crossover")]
impl Crossover for TestGene {
    fn crossover(&self, other: &Self, rng: &mut impl Rng) -> Self {
        Self {
            a: (self.a + other.a + rng.random_range(-0.5..0.5)) / 2.,
        }
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
#[derive(RandomlyMutable, DivisionReproduction, GenerateRandom, Clone, Debug)]
#[cfg_attr(feature = "crossover", derive(CrossoverReproduction))]
struct MyDNA {
    g1: TestGene,
    g2: TestGene,
}

fn main() {
    println!("check `cargo expand --package genetic-rs --example derive --all-features`");
}

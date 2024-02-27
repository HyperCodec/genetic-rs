use genetic_rs::prelude::*;

#[derive(Clone)]
struct TestGene {
    a: f32,
}

impl RandomlyMutable for TestGene {
    fn mutate(&mut self, rate: f32, rng: &mut impl rand::Rng) {
        self.a += rng.gen_range(-1.0..1.0) * rate;
    }
}

impl DivisionReproduction for TestGene {
    fn divide(&self, rng: &mut impl rand::Rng) -> Self {
        let mut child = self.clone();
        child.mutate(0.25, rng);
        child
    }
}

#[cfg(feature = "crossover")]
impl CrossoverReproduction for TestGene {
    fn crossover(&self, other: &Self, rng: &mut impl rand::Rng) -> Self {
        Self { a: (self.a + other.a + rng.gen_range(-0.5..0.5)) / 2. }
    }
}

impl GenerateRandom for TestGene {
    fn gen_random(rng: &mut impl rand::Rng) -> Self {
        Self { a: rng.gen_range(-1.0..1.0) }
    }
}

// using the derive macros here is only useful if the fields are not related to each other
#[derive(RandomlyMutable, DivisionReproduction, GenerateRandom, Clone)]
#[cfg_attr(feature = "crossover", derive(CrossoverReproduction))]
struct MyDNA {
    g1: TestGene,
    g2: TestGene,
}

fn main() {
    println!("check `cargo expand --package genetic-rs --example derive --all-features`");
}
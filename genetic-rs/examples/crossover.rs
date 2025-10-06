use genetic_rs::prelude::*;

#[derive(Clone, Debug, PartialEq)]
struct MyGenome {
    val: f32,
}

impl RandomlyMutable for MyGenome {
    fn mutate(&mut self, rate: f32, rng: &mut impl Rng) {
        self.val += rng.random::<f32>() * rate;
    }
}

impl Crossover for MyGenome {
    fn crossover(&self, other: &Self, rate: f32, rng: &mut impl Rng) -> Self {
        let mut child = Self {
            val: (self.val + other.val) / 2.,
        };
        child.mutate(rate, rng);
        child
    }
}

impl GenerateRandom for MyGenome {
    fn gen_random(rng: &mut impl Rng) -> Self {
        Self {
            val: rng.random::<f32>() * 1000.,
        }
    }
}

#[cfg(not(feature = "rayon"))]
fn main() {
    let mut rng = rand::rng();

    let magic_number = rng.random::<f32>() * 1000.;
    let fitness = move |e: &MyGenome| -> f32 { 1 / (magic_number - e.val).abs().max(1e-7) };

    let mut sim = GeneticSim::new(
        Vec::gen_random(&mut rng, 100),
        FitnessEliminator::new_with_default(fitness),
        CrossoverRepopulator::new(0.25), // 25% mutation rate
    );

    sim.perform_generations(100);

    dbg!(sim.genomes, magic_number);
}

#[cfg(feature = "rayon")]
fn main() {
    let mut rng = rand::rng();
    let magic_number = rng.random::<f32>() * 1000.;
    let fitness = move |e: &MyGenome| -> f32 { -(magic_number - e.val).abs() };

    let mut sim = GeneticSim::new(
        Vec::gen_random(100),
        FitnessEliminator::new_with_default(fitness),
        CrossoverRepopulator::new(0.25), // 25% mutation rate
    );

    for _ in 0..100 {
        sim.next_generation();
    }

    dbg!(sim.genomes, magic_number);
}

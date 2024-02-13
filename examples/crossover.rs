use genetic_rs::prelude::*;
use rand::prelude::*;

#[derive(Clone, Debug, PartialEq)]
struct MyGenome {
    val: f32,
}

impl RandomlyMutable for MyGenome {
    fn mutate(&mut self, rate: f32, rng: &mut impl rand::Rng) {
        self.val += rng.gen::<f32>() * rate;
    }
}

impl CrossoverReproduction for MyGenome {
    fn crossover(&self, other: &Self, rng: &mut impl rand::Rng) -> Self {
        let mut child = Self {
            val: (self.val + other.val) / 2.,
        };
        child.mutate(0.25, rng);
        child
    }
}

impl Prunable for MyGenome {}

impl GenerateRandom for MyGenome {
    fn gen_random(rng: &mut impl Rng) -> Self {
        Self {
            val: rng.gen::<f32>() * 1000.,
        }
    }
}

#[cfg(not(feature = "rayon"))]
fn main() {
    let mut rng = rand::thread_rng();

    let magic_number = rng.gen::<f32>() * 1000.;
    let fitness = move |e: &MyGenome| -> f32 { -(magic_number - e.val).abs() };

    let mut sim = GeneticSim::new(
        Vec::gen_random(&mut rng, 100),
        fitness,
        crossover_pruning_nextgen,
    );

    for _ in 0..100 {
        sim.next_generation();
    }

    dbg!(sim.genomes, magic_number);
}

#[cfg(feature = "rayon")]
fn main() {
    let mut rng = rand::thread_rng();
    let magic_number = rng.gen::<f32>() * 1000.;
    let fitness = move |e: &MyGenome| -> f32 { -(magic_number - e.val).abs() };

    let mut sim = GeneticSim::new(Vec::gen_random(100), fitness, crossover_pruning_nextgen);

    for _ in 0..100 {
        sim.next_generation();
    }

    dbg!(sim.genomes, magic_number);
}

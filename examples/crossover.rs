use genetic_rs::prelude::*;
use rand::prelude::*;

#[derive(Clone, Debug, PartialEq)]
struct MyEntity {
    val: f32,
}

impl RandomlyMutable for MyEntity {
    fn mutate(&mut self, rate: f32, rng: &mut impl rand::Rng) {
        self.val += rng.gen::<f32>() * rate;
    }
}

impl CrossoverReproduction for MyEntity {
    fn crossover(&self, other: &Self, rng: &mut impl rand::Rng) -> Self {
        let mut child = Self {
            val: (self.val + other.val) / 2.,
        };
        child.mutate(0.25, rng);
        child
    }
}

impl Prunable for MyEntity {}

impl GenerateRandom for MyEntity {
    fn gen_random(rng: &mut impl Rng) -> Self {
        Self {
            val: rng.gen::<f32>() * 1000.,
        }
    }
}

fn main() {
    let mut rng = rand::thread_rng();

    let magic_number = rng.gen::<f32>() * 1000.;
    let fitness = move |e: &MyEntity| -> f32 { -(magic_number - e.val).abs() };

    let mut sim = GeneticSim::new(
        Vec::gen_random(&mut rng, 100),
        fitness,
        crossover_pruning_nextgen,
    );

    for _ in 0..100 {
        sim.next_generation();
    }

    dbg!(sim.entities, magic_number);
}

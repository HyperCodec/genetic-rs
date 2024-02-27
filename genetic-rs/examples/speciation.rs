use genetic_rs::prelude::*;

#[derive(Clone, PartialEq, Debug)]
struct MyGenome {
    val1: f32,
    val2: f32,
}

impl RandomlyMutable for MyGenome {
    fn mutate(&mut self, rate: f32, rng: &mut impl rand::Rng) {
        self.val1 += rng.gen_range(-1.0..1.0) * rate;
        self.val2 += rng.gen_range(-1.0..1.0) * rate;
    }
}

impl DivisionReproduction for MyGenome {
    fn divide(&self, rng: &mut impl rand::Rng) -> Self {
        let mut child = self.clone();
        child.mutate(0.25, rng);
        child
    }
}

impl CrossoverReproduction for MyGenome {
    fn crossover(&self, other: &Self, rng: &mut impl rand::Rng) -> Self {
        let mut child = Self {
            val1: (self.val1 + other.val1) / 2.,
            val2: (self.val2 + other.val2) / 2.,
        };
        child.mutate(0.25, rng);
        child
    }
}

impl Speciated for MyGenome {
    fn is_same_species(&self, other: &Self) -> bool {
        // declared same species by being close to matching values.
        (self.val1 - other.val1).abs() + (self.val2 - other.val2).abs() <= 5.
    }
}

impl Prunable for MyGenome {}

impl GenerateRandom for MyGenome {
    fn gen_random(rng: &mut impl rand::Rng) -> Self {
        Self {
            val1: rng.gen_range(-1.0..1.0),
            val2: rng.gen_range(-1.0..1.0),
        }
    }
}

fn fitness(g: &MyGenome) -> f32 {
    // train to make val1 and val2 as different as possible
    (g.val1 - g.val2).abs()
}

#[cfg(not(feature = "rayon"))]
fn main() {
    let mut rng = rand::thread_rng();

    let mut sim = GeneticSim::new(
        Vec::gen_random(&mut rng, 100),
        fitness,
        speciated_crossover_pruning_nextgen,
    );

    // speciation tends to take more generations (not needed to this extent, but the crate is fast enough to where it isn't much of a compromise)
    for _ in 0..1000 {
        sim.next_generation();
    }

    dbg!(sim.genomes);
}

#[cfg(feature = "rayon")]
fn main() {
    let mut sim = GeneticSim::new(
        Vec::gen_random(100),
        fitness,
        speciated_crossover_pruning_nextgen,
    );

    for _ in 0..1000 {
        sim.next_generation();
    }

    dbg!(sim.genomes);
}

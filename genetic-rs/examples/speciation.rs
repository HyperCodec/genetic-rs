use genetic_rs::prelude::*;

#[derive(Clone, PartialEq, Debug)]
struct MyGenome {
    val1: f32,
    val2: f32,
}

impl RandomlyMutable for MyGenome {
    type Context = ();

    fn mutate(&mut self, _: &(), rate: f32, rng: &mut impl Rng) {
        self.val1 += rng.random_range(-1.0..1.0) * rate;
        self.val2 += rng.random_range(-1.0..1.0) * rate;
    }
}

impl Mitosis for MyGenome {
    type Context = ();

    fn divide(&self, _: &(), rate: f32, rng: &mut impl Rng) -> Self {
        let mut child = self.clone();
        child.mutate(&(), rate, rng);
        child
    }
}

impl Crossover for MyGenome {
    type Context = ();

    fn crossover(&self, other: &Self, _: &(), rate: f32, rng: &mut impl Rng) -> Self {
        let mut child = Self {
            val1: (self.val1 + other.val1) / 2.,
            val2: (self.val2 + other.val2) / 2.,
        };
        child.mutate(&(), rate, rng);
        child
    }
}

#[derive(PartialEq, Eq, Hash)]
struct MySpecies(u32);

impl Speciated for MyGenome {
    type Species = MySpecies;

    fn species(&self) -> Self::Species {
        MySpecies((self.val1 + self.val2).round() as u32)
    }
}

impl GenerateRandom for MyGenome {
    fn gen_random(rng: &mut impl Rng) -> Self {
        Self {
            val1: rng.random_range(-1.0..1.0),
            val2: rng.random_range(-1.0..1.0),
        }
    }
}

fn fitness(g: &MyGenome) -> f32 {
    // train to make val1 and val2 as different as possible
    (g.val1 - g.val2).abs()
}

fn main() {
    let mut rng = rand::rng();

    let mut sim = GeneticSim::new(
        Vec::gen_random(&mut rng, 100),
        FitnessEliminator::new_with_default(fitness),
        // 25% mutation rate, allow cross-species reproduction in emergency scenarios
        SpeciatedCrossoverRepopulator::new(0.25, true, ()),
    );

    sim.perform_generations(100);

    dbg!(sim.genomes);
}

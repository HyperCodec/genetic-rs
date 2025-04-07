//! The example from the README.md and crate root

use genetic_rs::prelude::*;

#[derive(Clone, Debug)] // clone is currently a required derive for pruning nextgens.
struct MyGenome {
    field1: f32,
}

// required in all of the builtin functions as requirements of `DivisionReproduction` and `CrossoverReproduction`.
impl RandomlyMutable for MyGenome {
    fn mutate(&mut self, rate: f32, rng: &mut impl Rng) {
        self.field1 += rng.random::<f32>() * rate;
    }
}

// required for `division_pruning_nextgen`.
impl DivisionReproduction for MyGenome {
    fn divide(&self, rng: &mut impl Rng) -> Self {
        let mut child = self.clone();
        child.mutate(0.25, rng); // use a constant mutation rate when spawning children in pruning algorithms.
        child
    }
}

// required for the builtin pruning algorithms.
impl Prunable for MyGenome {
    fn despawn(self) {
        // unneccessary to implement this function, but it can be useful for debugging and cleaning up genomes.
        println!("{:?} died", self);
    }
}

// helper trait that allows us to use `Vec::gen_random` for the initial population.
impl GenerateRandom for MyGenome {
    fn gen_random(rng: &mut impl Rng) -> Self {
        Self {
            field1: rng.random(),
        }
    }
}

fn my_fitness_fn(ent: &MyGenome) -> f32 {
    // this just means that the algorithm will try to create as big a number as possible due to fitness being directly taken from the field.
    // in a more complex genetic algorithm, you will want to utilize `ent` to test them and generate a reward.
    ent.field1
}

#[cfg(not(feature = "rayon"))]
fn main() {
    let mut rng = rand::rng();
    let mut sim = GeneticSim::new(
        // you must provide a random starting population.
        // size will be preserved in builtin nextgen fns, but it is not required to keep a constant size if you were to build your own nextgen function.
        // in this case, you do not need to specify a type for `Vec::gen_random` because of the input of `my_fitness_fn`.
        Vec::gen_random(&mut rng, 100),
        my_fitness_fn,
        division_pruning_nextgen,
    );

    // perform evolution (100 gens)
    sim.perform_generations(100);

    dbg!(sim.genomes);
}

#[cfg(feature = "rayon")]
fn main() {
    let mut sim = GeneticSim::new(
        // you must provide a random starting population.
        // size will be preserved in builtin nextgen fns, but it is not required to keep a constant size if you were to build your own nextgen function.
        // in this case, you do not need to specify a type for `Vec::gen_random` because of the input of `my_fitness_fn`.
        Vec::gen_random(100),
        my_fitness_fn,
        division_pruning_nextgen,
    );

    // perform evolution (100 gens)
    for _ in 0..100 {
        sim.next_generation(); // in a genetic algorithm with state, such as a physics simulation, you'd want to do things with `sim.randomomes` in between these calls
    }

    dbg!(sim.genomes);
}

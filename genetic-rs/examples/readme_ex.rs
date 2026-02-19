//! The example from the README.md and crate root

use genetic_rs::prelude::*;

#[derive(Clone, Debug)] // clone is currently a required derive for pruning nextgens.
struct MyGenome {
    field1: f32,
}

// required in all of the builtin repopulators as requirements of `Mitosis` and `Crossover`.
impl RandomlyMutable for MyGenome {
    type Context = ();

    fn mutate(&mut self, _: &(), rate: f32, rng: &mut impl Rng) {
        self.field1 += rng.random::<f32>() * rate;
    }
}

// use auto derives for the builtin nextgen functions to work with your genome.
impl Mitosis for MyGenome {
    type Context = ();

    fn divide(&self, _: &(), rate: f32, rng: &mut impl Rng) -> Self {
        let mut child = self.clone();
        child.mutate(&(), rate, rng);
        child
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

fn main() {
    let mut rng = rand::rng();
    let mut sim = GeneticSim::new(
        // you must provide a random starting population.
        // size will be preserved in builtin nextgen fns, but it is not required to keep a constant size if you were to build your own nextgen function.
        // in this case, you do not need to specify a type for `Vec::gen_random` because of the input of `my_fitness_fn`.
        Vec::gen_random(&mut rng, 100),
        FitnessEliminator::new_without_observer(my_fitness_fn),
        MitosisRepopulator::new(0.25, ()), // 25% mutation rate
    );

    // perform evolution (100 gens)
    sim.perform_generations(100);

    dbg!(sim.genomes);
}

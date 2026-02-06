# genetic-rs

[<img alt="github" src="https://img.shields.io/github/last-commit/inflectrix/genetic-rs" height="20">](https://github.com/inflectrix/genetic-rs)
[<img alt="crates.io" src="https://img.shields.io/crates/d/genetic-rs" height="20">](https://crates.io/crates/genetic-rs)
[<img alt="docs.rs" src="https://img.shields.io/docsrs/genetic-rs" height="20">](https://docs.rs/genetic-rs)

A small framework for managing genetic algorithms.

### Features
First off, this crate comes with the `builtin`, `crossover`, and `genrand` features by default. If you want it to be parallelized (which is true in most cases), you can add the `rayon` feature. If you want your crossover to be speciated, you can add the `speciation` feature.

### How to Use
> [!NOTE] 
> If you are interested in implementing NEAT with this, try out the [neat](https://crates.io/crates/neat) crate

Here's a simple genetic algorithm:

```rust
use genetic_rs::prelude::*;

// `Mitosis` can be derived if both `Clone` and `RandomlyMutable` are present.
#[derive(Clone, Debug, Mitosis)]
struct MyGenome {
    field1: f32,
}

// required in all of the builtin Repopulators as requirements of `Mitosis` and `Crossover`
impl RandomlyMutable for MyGenome {
    type Context = (); // empty context for a simple mutation
    
    fn mutate(&mut self, _ctx: &(), rate: f32, rng: &mut impl Rng) {
        self.field1 += rng.random::<f32>() * rate;
    }
}

// allows us to use `Vec::gen_random` for the initial population. note that `Vec::gen_random` has a slightly different function signature depending on whether the `rayon` feature is enabled.
impl GenerateRandom for MyGenome {
    fn gen_random(rng: &mut impl Rng) -> Self {
        Self { field1: rng.random() }
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
        // in this case, the compiler can infer the type of `Vec::gen_random` because of the input of `my_fitness_fn`.
        Vec::gen_random(&mut rng, 100),
        FitnessEliminator::new_with_default(my_fitness_fn),
        MitosisRepopulator::new(0.25, ()), // 25% mutation rate, empty context
    );
 
    // perform evolution (100 gens)
    sim.perform_generations(100);
 
    dbg!(sim.genomes);
}
```

That is the minimal code for a working genetic algorithm on default features (+ rayon). You can [read the docs](https://docs.rs/genetic-rs) or [check the examples](/genetic-rs/examples/) for more complicated systems. I highly recommend looking into crossover reproduction, as it tends to produce better results than mitosis.

### License
This project falls under the `MIT` license.

# genetic-rs

[<img alt="github" src="https://img.shields.io/github/last-commit/inflectrix/genetic-rs" height="20">](https://github.com/inflectrix/genetic-rs)
[<img alt="crates.io" src="https://img.shields.io/crates/d/genetic-rs" height="20">](https://crates.io/crates/genetic-rs)
[<img alt="docs.rs" src="https://img.shields.io/docsrs/genetic-rs" height="20">](https://docs.rs/genetic-rs)

A small crate for quickstarting genetic algorithm projects.

### How to Use
*note: if you are interested in implementing NEAT with this, try out the [neat](https://crates.io/crates/neat) crate*

### Features
First off, this crate comes with the `builtin` and `genrand` features by default. If you want to add the builtin crossover reproduction extension, you can do so by adding the `crossover` feature. If you want it to be parallelized, you can add the `rayon` feature. If you want your crossover to be speciated, you can add the `speciation` feature.

Once you have eveything imported as you wish, you can define your genome and impl the required traits:

```rust
#[derive(Clone, Debug)] // clone is currently a required derive for pruning nextgens.
struct MyGenome {
    field1: f32,
}

// required in all of the builtin functions as requirements of `DivsionReproduction` and `CrossoverReproduction`
impl RandomlyMutable for MyGenome {
    fn mutate(&mut self, rate: f32, rng: &mut impl rand::Rng) {
        self.field1 += rng.gen::<f32>() * rate;
    }
}

// required for `division_pruning_nextgen`.
impl DivsionReproduction for MyGenome {
    fn divide(&self, rng: &mut impl rand::Rng) -> Self {
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
    fn gen_random(rng: &mut impl rand::Rng) -> Self {
        Self { field1: rng.gen() }
    }
}
```

Once you have a struct, you must create your fitness function:
```rust
fn my_fitness_fn(ent: &MyGenome) -> f32 {
    // this just means that the algorithm will try to create as big a number as possible due to fitness being directly taken from the field.
    // in a more complex genetic algorithm, you will want to utilize `ent` to test them and generate a reward.
    ent.field1
}
```


Once you have your reward function, you can create a `GeneticSim` object to manage and control the evolutionary steps:

```rust
fn main() {
    let mut rng = rand::thread_rng();
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
```

That is the minimal code for a working pruning-based genetic algorithm. You can [read the docs](https://docs.rs/genetic-rs) or [check the examples](/genetic-rs/examples/) for more complicated systems.

### License
This project falls under the `MIT` license.

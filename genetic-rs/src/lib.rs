//! A small crate to quickstart genetic algorithm projects
//!
//! ### How to Use
//!
//! ### Features
//! First off, this crate comes with the `builtin` and `genrand` features by default.
//! If you want to add the builtin crossover reproduction extension, you can do so by adding the `crossover` feature.
//! If you want it to be parallelized, you can add the `rayon` feature.
//! If you want your crossover to be speciated, you can add the `speciation` feature.
//!
//! Once you have eveything imported as you wish, you can define your genomes and impl the required traits:
//!
//! ```rust, ignore
//! #[derive(Clone, Debug)] // clone is currently a required derive for pruning nextgens.
//! struct MyGenome {
//!     field1: f32,
//! }
//!
//! // required in all of the builtin functions as requirements of `DivisionReproduction` and `CrossoverReproduction`
//! impl RandomlyMutable for MyGenome {
//!     fn mutate(&mut self, rate: f32, rng: &mut impl rand::Rng) {
//!         self.field1 += rng.gen::<f32>() * rate;
//!     }
//! }
//!
//! // required for `division_pruning_nextgen`.
//! impl DivisionReproduction for MyGenome {
//!     fn divide(&self, rng: &mut impl rand::Rng) -> Self {
//!         let mut child = self.clone();
//!         child.mutate(0.25, rng); // use a constant mutation rate when spawning children in pruning algorithms.
//!         child
//!     }
//! }
//!
//! // required for the builtin pruning algorithms.
//! impl Prunable for MyGenome {
//!     fn despawn(self) {
//!         // unneccessary to implement this function, but it can be useful for debugging and cleaning up genomes.
//!         println!("{:?} died", self);
//!     }
//! }
//!
//! // helper trait that allows us to use `Vec::gen_random` for the initial population.
//! impl GenerateRandom for MyGenome {
//!     fn gen_random(rng: &mut impl rand::Rng) -> Self {
//!         Self { field1: rng.gen() }
//!     }
//! }
//! ```
//!
//! Once you have a struct, you must create your fitness function:
//! ```rust, ignore
//! fn my_fitness_fn(ent: &MyGenome) -> f32 {
//!     // this just means that the algorithm will try to create as big a number as possible due to fitness being directly taken from the field.
//!     // in a more complex genetic algorithm, you will want to utilize `ent` to test them and generate a reward.
//!     ent.field1
//! }
//! ```
//!
//!
//! Once you have your fitness function, you can create a [`GeneticSim`] object to manage and control the evolutionary steps:
//!
//! ```rust, ignore
//! fn main() {
//!     let mut rng = rand::thread_rng();
//!     let mut sim = GeneticSim::new(
//!         // you must provide a random starting population.
//!         // size will be preserved in builtin nextgen fns, but it is not required to keep a constant size if you were to build your own nextgen function.
//!         // in this case, you do not need to specify a type for `Vec::gen_random` because of the input of `my_fitness_fn`.
//!         Vec::gen_random(&mut rng, 100),
//!         my_fitness_fn,
//!         division
//!     );
//!     
//!     // perform evolution (100 gens)
//!     for _ in 0..100 {
//!         sim.next_generation(); // in a genetic algorithm with state, such as a physics simulation, you'd want to do things with `sim.genomes` in between these calls
//!     }
//!     
//!     dbg!(sim.genomes);
//! }
//! ```
//!
//! That is the minimal code for a working pruning-based genetic algorithm. You can [read the docs](https://docs.rs/genetic-rs) or [check the examples](/examples/) for more complicated systems.
//!
//! ### License
//! This project falls under the `MIT` license.

pub mod prelude {
    pub use genetic_rs_common::prelude::*;

    #[cfg(feature = "derive")]
    pub use genetic_rs_macros::*;
}

pub use genetic_rs_common::*;

#[cfg(feature = "derive")]
pub use genetic_rs_macros::*;

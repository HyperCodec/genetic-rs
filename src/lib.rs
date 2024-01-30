#![warn(missing_docs)]
#![allow(clippy::needless_doctest_main)]

//! A small crate to quickstart genetic algorithm projects
//!
//! ### How to Use
//! First off, this crate comes with the `builtin` and `genrand` features by default. If you want to add the builtin crossover reproduction extension, you can do so by adding the `crossover` feature.
//!
//! Once you have eveything imported as you wish, you can define your entity and impl the required traits:
//!
//! ```rust, ignore
//! #[derive(Clone, Debug)] // clone is currently a required derive for pruning nextgens.
//! struct MyEntity {
//!     field1: f32,
//! }
//!
//! // required in all of the builtin functions as requirements of `DivisionReproduction` and `CrossoverReproduction`
//! impl RandomlyMutable for MyEntity {
//!     fn mutate(&mut self, rate: f32, rng: &mut impl rand::Rng) {
//!         self.field1 += rng.gen::<f32>() * rate;
//!     }
//! }
//!
//! // required for `division_pruning_nextgen`.
//! impl DivisionReproduction for MyEntity {
//!     fn spawn_child(&self, rng: &mut impl rand::Rng) -> Self {
//!         let mut child = self.clone();
//!         child.mutate(0.25, rng); // use a constant mutation rate when spawning children in pruning algorithms.
//!         child
//!     }
//! }
//!
//! // required for the builtin pruning algorithms.
//! impl Prunable for MyEntity {
//!     fn despawn(self) {
//!         // unneccessary to implement this function, but it can be useful for debugging and cleaning up entities.
//!         println!("{:?} died", self);
//!     }
//! }
//!
//! // helper trait that allows us to use `Vec::gen_random` for the initial population.
//! impl GenerateRandom for MyEntity {
//!     fn gen_random(rng: &mut impl rand::Rng) -> Self {
//!         Self { field1: rng.gen() }
//!     }
//! }
//! ```
//!
//! Once you have a struct, you must create your fitness function:
//! ```rust, ignore
//! fn my_fitness_fn(ent: &MyEntity) -> f32 {
//!     // this just means that the algorithm will try to create as big a number as possible due to fitness being directly taken from the field.
//!     // in a more complex genetic algorithm, you will want to utilize `ent` to test them and generate a reward.
//!     ent.field1
//! }
//! ```
//!
//!
//! Once you have your fitness function, you can create a `GeneticSim` object to manage and control the evolutionary steps:
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
//!         sim.next_generation(); // in a genetic algorithm with state, such as a physics simulation, you'd want to do things with `sim.entities` in between these calls
//!     }
//!     
//!     dbg!(sim.entities);
//! }
//! ```
//!
//! That is the minimal code for a working pruning-based genetic algorithm. You can [read the docs](https://docs.rs/genetic-rs) or [check the examples](/examples/) for more complicated systems.
//!
//! ### License
//! This project falls under the `MIT` license.

use replace_with::replace_with_or_abort;

/// Built-in nextgen functions and traits to go with them.
#[cfg(feature = "builtin")]
pub mod builtin;

/// Used to quickly import everything this crate has to offer.
/// Simply add `use genetic_rs::prelude::*` to begin using this crate.
pub mod prelude;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Represents a fitness function. Inputs a reference to the entity and outputs an f32.
pub type FitnessFn<E> = dyn Fn(&E) -> f32 + Send + Sync + 'static;

/// Represents a nextgen function. Inputs entities and rewards and produces the next generation
pub type NextgenFn<E> = dyn Fn(Vec<(E, f32)>) -> Vec<E> + Send + Sync + 'static;

/// The simulation controller.
/// ```rust
/// use genetic_rs::prelude::*;
///
/// #[derive(Debug, Clone)]
/// struct MyEntity {
///     a: f32,
///     b: f32,
/// }
///
/// impl RandomlyMutable for MyEntity {
///     fn mutate(&mut self, rate: f32, rng: &mut impl rand::Rng) {
///         self.a += rng.gen::<f32>() * rate;
///         self.b += rng.gen::<f32>() * rate;
///     }
/// }
///
/// impl DivisionReproduction for MyEntity {
///     fn spawn_child(&self, rng: &mut impl rand::Rng) -> Self {
///         let mut child = self.clone();
///         child.mutate(0.25, rng); // you'll generally want to use a constant mutation rate for mutating children.
///         child
///     }
/// }
///
/// impl Prunable for MyEntity {} // if we wanted to, we could implement the `despawn` function to run any cleanup code as needed. in this example, though, we do not need it.
///
/// impl GenerateRandom for MyEntity {
///     fn gen_random(rng: &mut impl rand::Rng) -> Self {
///         Self {
///             a: rng.gen(),
///             b: rng.gen(),
///         }
///     }
/// }
///
/// fn main() {
///     let my_fitness_fn = |e: &MyEntity| {
///         e.a * e.b // should result in entities increasing their value
///     };
///
///     let mut rng = rand::thread_rng();
///
///     let mut sim = GeneticSim::new(
///         Vec::gen_random(&mut rng, 1000),
///         my_fitness_fn,
///         division_pruning_nextgen,
///     );
///
///     for _ in 0..100 {
///         // if this were a more complex simulation, you might test entities in `sim.entities` between `next_generation` calls to provide a more accurate reward.
///         sim.next_generation();
///     }
///
///     dbg!(sim.entities);
/// }
/// ```
#[cfg(not(feature = "rayon"))]
pub struct GeneticSim<E>
where
    E: Sized,
{
    /// The current population of entities
    pub entities: Vec<E>,
    fitness: Box<FitnessFn<E>>,
    next_gen: Box<NextgenFn<E>>,
}

/// Rayon version of the [GeneticSim] struct
#[cfg(feature = "rayon")]
pub struct GeneticSim<E>
where
    E: Sized + Send,
{
    /// The current population of entities
    pub entities: Vec<E>,
    fitness: Box<FitnessFn<E>>,
    next_gen: Box<NextgenFn<E>>,
}

#[cfg(not(feature = "rayon"))]
impl<E> GeneticSim<E>
where
    E: Sized,
{
    /// Creates a GeneticSim with a given population of `starting_entities` (the size of which will be retained),
    /// a given fitness function, and a given nextgen function.
    pub fn new(
        starting_entities: Vec<E>,
        fitness: impl Fn(&E) -> f32 + Send + Sync + 'static,
        next_gen: impl Fn(Vec<(E, f32)>) -> Vec<E> + Send + Sync + 'static,
    ) -> Self {
        Self {
            entities: starting_entities,
            fitness: Box::new(fitness),
            next_gen: Box::new(next_gen),
        }
    }

    /// Uses the `next_gen` provided in [GeneticSim::new] to create the next generation of entities.
    pub fn next_generation(&mut self) {
        // TODO maybe remove unneccessary dependency, can prob use std::mem::replace
        replace_with_or_abort(&mut self.entities, |entities| {
            let rewards = entities
                .into_iter()
                .map(|e| {
                    let fitness: f32 = (self.fitness)(&e);
                    (e, fitness)
                })
                .collect();

            (self.next_gen)(rewards)
        });
    }
}

#[cfg(feature = "rayon")]
impl<E> GeneticSim<E>
where
    E: Sized + Send,
{
    /// Creates a new GeneticSim using a starting population, fitness function, and nextgen function
    pub fn new(
        starting_entities: Vec<E>,
        fitness: impl Fn(&E) -> f32 + Send + Sync + 'static,
        next_gen: impl Fn(Vec<(E, f32)>) -> Vec<E> + Send + Sync + 'static,
    ) -> Self {
        Self {
            entities: starting_entities,
            fitness: Box::new(fitness),
            next_gen: Box::new(next_gen),
        }
    }

    /// Performs selection and produces the next generation within the simulation.
    pub fn next_generation(&mut self) {
        replace_with_or_abort(&mut self.entities, |entities| {
            let rewards = entities
                .into_par_iter()
                .map(|e| {
                    let fitness: f32 = (self.fitness)(&e);
                    (e, fitness)
                })
                .collect();

            (self.next_gen)(rewards)
        });
    }
}

#[cfg(feature = "genrand")]
use rand::prelude::*;

/// Helper trait used in the generation of random starting populations
#[cfg(feature = "genrand")]
pub trait GenerateRandom {
    /// Create a completely random instance of the entity
    fn gen_random(rng: &mut impl Rng) -> Self;
}

/// Blanket trait used on collections that contain objects implementing GenerateRandom
#[cfg(all(feature = "genrand", not(feature = "rayon")))]
pub trait GenerateRandomCollection<T>
where
    T: GenerateRandom,
{
    /// Generate a random collection of the inner objects with a given amount
    fn gen_random(rng: &mut impl Rng, amount: usize) -> Self;
}

/// Rayon version of the [GenerateRandomCollection] trait
#[cfg(all(feature = "genrand", feature = "rayon"))]
pub trait GenerateRandomCollection<T>
where
    T: GenerateRandom + Send,
{
    /// Generate a random collection of the inner objects with the given amount. Does not pass in rng like the sync counterpart.
    fn gen_random(amount: usize) -> Self;
}

#[cfg(not(feature = "rayon"))]
impl<C, T> GenerateRandomCollection<T> for C
where
    C: FromIterator<T>,
    T: GenerateRandom,
{
    fn gen_random(rng: &mut impl Rng, amount: usize) -> Self {
        (0..amount)
            .into_iter()
            .map(|_| T::gen_random(rng))
            .collect()
    }
}

#[cfg(feature = "rayon")]
impl<C, T> GenerateRandomCollection<T> for C
where
    C: FromParallelIterator<T>,
    T: GenerateRandom + Send,
{
    fn gen_random(amount: usize) -> Self {
        (0..amount)
            .into_par_iter()
            .map(|_| T::gen_random(&mut rand::thread_rng()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::prelude::*;

    #[test]
    fn send_sim() {
        let mut sim = GeneticSim::new(vec![()], |_| 0., |_| vec![()]);

        let h = std::thread::spawn(move || {
            sim.next_generation();
        });

        h.join().unwrap();
    }
}

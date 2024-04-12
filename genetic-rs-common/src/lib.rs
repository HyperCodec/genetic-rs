#![warn(missing_docs)]
#![allow(clippy::needless_doctest_main)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! The crate containing the core traits and structs of genetic-rs.

use replace_with::replace_with_or_abort;

/// Built-in nextgen functions and traits to go with them.
#[cfg_attr(docsrs, doc(cfg(feature = "builtin")))]
#[cfg(feature = "builtin")]
pub mod builtin;

/// Used to quickly import everything this crate has to offer.
/// Simply add `use genetic_rs::prelude::*` to begin using this crate.
pub mod prelude;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Represents a fitness function. Inputs a reference to the genome and outputs an f32.
pub trait FitnessFn<G> {
    /// Evaluates a genome's fitness
    fn fitness(&self, genome: &G) -> f32;
}

impl<F: Fn(&G) -> f32, G> FitnessFn<G> for F {
    fn fitness(&self, genome: &G) -> f32 {
        (self)(genome)
    }
}

/// Represents a nextgen function. Inputs genomes and rewards and produces the next generation
pub trait NextgenFn<G> {
    /// Creates the next generation from the current fitness values.
    fn next_gen(&self, fitness: Vec<(G, f32)>) -> Vec<G>;
}

impl<F: Fn(Vec<(G, f32)>) -> Vec<G>, G> NextgenFn<G> for F {
    fn next_gen(&self, fitness: Vec<(G, f32)>) -> Vec<G> {
        (self)(fitness)
    }
}

/// The simulation controller.
/// ```rust
/// use genetic_rs_common::prelude::*;
///
/// #[derive(Debug, Clone)]
/// struct MyGenome {
///     a: f32,
///     b: f32,
/// }
///
/// impl RandomlyMutable for MyGenome {
///     fn mutate(&mut self, rate: f32, rng: &mut impl rand::Rng) {
///         self.a += rng.gen::<f32>() * rate;
///         self.b += rng.gen::<f32>() * rate;
///     }
/// }
///
/// impl DivisionReproduction for MyGenome {
///     fn divide(&self, rng: &mut impl rand::Rng) -> Self {
///         let mut child = self.clone();
///         child.mutate(0.25, rng); // you'll generally want to use a constant mutation rate for mutating children.
///         child
///     }
/// }
///
/// impl Prunable for MyGenome {} // if we wanted to, we could implement the `despawn` function to run any cleanup code as needed. in this example, though, we do not need it.
///
/// impl GenerateRandom for MyGenome {
///     fn gen_random(rng: &mut impl rand::Rng) -> Self {
///         Self {
///             a: rng.gen(),
///             b: rng.gen(),
///         }
///     }
/// }
///
/// fn main() {
///     let my_fitness_fn = |e: &MyGenome| {
///         e.a * e.b // should result in genomes increasing their value
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
///         // if this were a more complex simulation, you might test genomes in `sim.genomes` between `next_generation` calls to provide a more accurate reward.
///         sim.next_generation();
///     }
///
///     dbg!(sim.genomes);
/// }
/// ```
#[cfg(not(feature = "rayon"))]
pub struct GeneticSim<F, NG, G>
where
    F: FitnessFn<G>,
    NG: NextgenFn<G>,
    G: Sized,
{
    /// The current population of genomes
    pub genomes: Vec<G>,
    fitness: F,
    next_gen: NG,
}

/// Rayon version of the [`GeneticSim`] struct
#[cfg(feature = "rayon")]
pub struct GeneticSim<F, NG, G>
where
    F: FitnessFn<G> + Send + Sync,
    NG: NextgenFn<G> + Send + Sync,
    G: Sized + Send,
{
    /// The current population of genomes
    pub genomes: Vec<G>,
    fitness: F,
    next_gen: NG,
}

#[cfg(not(feature = "rayon"))]
impl<F, NG, G> GeneticSim<F, NG, G>
where
    F: FitnessFn<G>,
    NG: NextgenFn<G>,
    G: Sized,
{
    /// Creates a [`GeneticSim`] with a given population of `starting_genomes` (the size of which will be retained),
    /// a given fitness function, and a given nextgen function.
    pub fn new(starting_genomes: Vec<G>, fitness: F, next_gen: NG) -> Self {
        Self {
            genomes: starting_genomes,
            fitness,
            next_gen,
        }
    }

    /// Uses the `next_gen` provided in [`GeneticSim::new`] to create the next generation of genomes.
    pub fn next_generation(&mut self) {
        // TODO maybe remove unneccessary dependency, can prob use std::mem::replace
        replace_with_or_abort(&mut self.genomes, |genomes| {
            let rewards = genomes
                .into_iter()
                .map(|e| {
                    let fitness: f32 = self.fitness.fitness(&e);
                    (e, fitness)
                })
                .collect();

            self.next_gen.next_gen(rewards)
        });
    }
}

#[cfg(feature = "rayon")]
impl<F, NG, G> GeneticSim<F, NG, G>
where
    F: FitnessFn<G> + Send + Sync,
    NG: NextgenFn<G> + Send + Sync,
    G: Sized + Send,
{
    /// Creates a [`GeneticSim`] with a given population of `starting_genomes` (the size of which will be retained),
    /// a given fitness function, and a given nextgen function.
    pub fn new(starting_genomes: Vec<G>, fitness: F, next_gen: NG) -> Self {
        Self {
            genomes: starting_genomes,
            fitness,
            next_gen,
        }
    }

    /// Performs selection and produces the next generation within the simulation.
    pub fn next_generation(&mut self) {
        replace_with_or_abort(&mut self.genomes, |genomes| {
            let rewards = genomes
                .into_par_iter()
                .map(|e| {
                    let fitness: f32 = self.fitness.fitness(&e);
                    (e, fitness)
                })
                .collect();

            self.next_gen.next_gen(rewards)
        });
    }
}

#[cfg(feature = "genrand")]
use rand::prelude::*;

/// Helper trait used in the generation of random starting populations
#[cfg(feature = "genrand")]
#[cfg_attr(docsrs, doc(cfg(feature = "genrand")))]
pub trait GenerateRandom {
    /// Create a completely random instance of the genome
    fn gen_random(rng: &mut impl Rng) -> Self;
}

/// Blanket trait used on collections that contain objects implementing [`GenerateRandom`]
#[cfg(all(feature = "genrand", not(feature = "rayon")))]
#[cfg_attr(docsrs, doc(cfg(feature = "genrand")))]
pub trait GenerateRandomCollection<T>
where
    T: GenerateRandom,
{
    /// Generate a random collection of the inner objects with a given amount
    fn gen_random(rng: &mut impl Rng, amount: usize) -> Self;
}

/// Rayon version of the [`GenerateRandomCollection`] trait
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
        (0..amount).map(|_| T::gen_random(rng)).collect()
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
        let mut sim = GeneticSim::new(vec![()], |_: &()| 0., |_: Vec<((), f32)>| vec![()]);

        let h = std::thread::spawn(move || {
            sim.next_generation();
        });

        h.join().unwrap();
    }
}

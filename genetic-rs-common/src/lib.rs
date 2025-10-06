#![warn(missing_docs)]
#![allow(clippy::needless_doctest_main)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! The crate containing the core traits and structs of genetic-rs.

/// Built-in nextgen functions and traits to go with them.
#[cfg_attr(docsrs, doc(cfg(feature = "builtin")))]
#[cfg(feature = "builtin")]
pub mod builtin;

/// Used to quickly import everything this crate has to offer.
/// Simply add `use genetic_rs::prelude::*` to begin using this crate.
pub mod prelude;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

#[cfg(feature = "tracing")]
use tracing::*;

#[cfg(feature = "tracing")]
#[allow(missing_docs)]
pub trait Rng: rand::Rng + std::fmt::Debug {}

#[cfg(feature = "tracing")]
impl<T: rand::Rng + std::fmt::Debug> Rng for T {}

#[cfg(not(feature = "tracing"))]
#[allow(missing_docs)]
pub trait Rng: rand::Rng {}

#[cfg(not(feature = "tracing"))]
impl<T: rand::Rng> Rng for T {}

/// Tests and eliminates the unfit from the simulation.
pub trait Eliminator<G> {
    /// Tests and eliminates the unfit from the simulation.
    fn eliminate(&self, genomes: Vec<G>) -> Vec<G>;
}

/// Refills the population of the simulation based on survivors.
pub trait Repopulator<G> {
    /// Replaces the genomes in the simulation.
    fn repopulate(&self, genomes: &mut Vec<G>, target_size: usize);
}

/// This struct is the main entry point for the simulation. It handles the state and evolution of the genomes
/// based on what eliminator and repopulator it receives.
#[cfg(not(feature = "rayon"))]
pub struct GeneticSim<G: Sized, E: Eliminator<G>, R: Repopulator<G>> {
    /// The current population of genomes
    pub genomes: Vec<G>,

    /// The eliminator used to eliminate unfit genomes
    pub eliminator: E,

    /// The repopulator used to refill the population
    pub repopulator: R,
}

/// Rayon version of the [`GeneticSim`] struct
#[cfg(feature = "rayon")]
pub struct GeneticSim<
    G: Sized + Sync,
    E: Eliminator<G> + Send + Sync,
    R: Repopulator<G> + Send + Sync,
> {
    /// The current population of genomes
    pub genomes: Vec<G>,

    /// The eliminator used to eliminate unfit genomes
    pub eliminator: E,

    /// The repopulator used to refill the population
    pub repopulator: R,
}

#[cfg(not(feature = "rayon"))]
impl<G, E, R> GeneticSim<G, E, R>
where
    G: Sized,
    E: Eliminator<G>,
    R: Repopulator<G>,
{
    /// Creates a [`GeneticSim`] with a given population of `starting_genomes` (the size of which will be retained),
    /// a given fitness function, and a given nextgen function.
    pub fn new(starting_genomes: Vec<G>, eliminator: E, repopulator: R) -> Self {
        Self {
            genomes: starting_genomes,
            eliminator,
            repopulator,
        }
    }

    /// Uses the [`Eliminator`] and [`Repopulator`] provided in [`GeneticSim::new`] to create the next generation of genomes.
    pub fn next_generation(&mut self) {
        #[cfg(feature = "tracing")]
        let span = span!(Level::TRACE, "next_generation");

        #[cfg(feature = "tracing")]
        let enter = span.enter();

        let genomes = std::mem::take(&mut self.genomes);

        let target_size = genomes.len();
        self.genomes = self.eliminator.eliminate(genomes);
        self.repopulator.repopulate(&mut self.genomes, target_size);

        #[cfg(feature = "tracing")]
        drop(enter);
    }

    /// Calls [`next_generation`][GeneticSim::next_generation] `count` number of times.
    pub fn perform_generations(&mut self, count: usize) {
        for _ in 0..count {
            self.next_generation();
        }
    }
}

#[cfg(feature = "rayon")]
impl<G, E, R> GeneticSim<G, E, R>
where
    G: Sized + Send + Sync,
    E: Eliminator<G> + Send + Sync,
    R: Repopulator<G> + Send + Sync,
{
    /// Creates a [`GeneticSim`] with a given population of `starting_genomes` (the size of which will be retained),
    /// a given fitness function, and a given nextgen function.
    pub fn new(starting_genomes: Vec<G>, eliminator: E, repopulator: R) -> Self {
        Self {
            genomes: starting_genomes,
            eliminator,
            repopulator,
        }
    }

    /// Uses the [`Eliminator`] and [`Repopulator`] provided in [`GeneticSim::new`] to create the next generation of genomes.
    pub fn next_generation(&mut self) {
        #[cfg(feature = "tracing")]
        let span = span!(Level::TRACE, "next_generation");

        #[cfg(feature = "tracing")]
        let enter = span.enter();

        let genomes = std::mem::take(&mut self.genomes);
        let target_size = genomes.len();
        self.genomes = self.eliminator.eliminate(genomes);
        self.repopulator.repopulate(&mut self.genomes, target_size);

        #[cfg(feature = "tracing")]
        drop(enter);
    }

    /// Calls [`next_generation`][GeneticSim::next_generation] `count` number of times.
    pub fn perform_generations(&mut self, count: usize) {
        for _ in 0..count {
            self.next_generation();
        }
    }
}

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
    #[cfg_attr(feature = "tracing", instrument)]
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
    #[cfg_attr(feature = "tracing", instrument)]
    fn gen_random(amount: usize) -> Self {
        (0..amount)
            .into_par_iter()
            .map(|_| T::gen_random(&mut rand::rng()))
            .collect()
    }
}

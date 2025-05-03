use crate::{Repopulator, Rng};

/// Used in all of the builtin [`next_gen`]s to randomly mutate genomes a given amount
#[cfg(not(feature = "tracing"))]
pub trait RandomlyMutable {
    /// Mutate the genome with a given mutation rate (0..1)
    fn mutate(&mut self, rate: f32, rng: &mut impl Rng);
}

/// Used in all of the builtin [`next_gen`]s to randomly mutate genomes a given amount
#[cfg(feature = "tracing")]
pub trait RandomlyMutable: std::fmt::Debug {
    /// Mutate the genome with a given mutation rate (0..1)
    fn mutate(&mut self, rate: f32, rng: &mut impl Rng);
}


/// Used in dividually-reproducing [`next_gen`]s
#[cfg(not(feature = "tracing"))]
pub trait DivisionReproduction: Clone {
    /// Create a new child with mutation. Similar to [RandomlyMutable::mutate], but returns a new instance instead of modifying the original.
    /// If it is simply returning a cloned and mutated version, consider using a constant mutation rate.
    fn divide(&self, rate: f32, rng: &mut impl Rng) -> Self;
}

/// Used in dividually-reproducing [`next_gen`]s
#[cfg(feature = "tracing")]
pub trait DivisionReproduction: std::fmt::Debug {
    /// Create a new child with mutation. Similar to [RandomlyMutable::mutate], but returns a new instance instead of modifying the original.
    /// If it is simply returning a cloned and mutated version, consider using a constant mutation rate.
    fn divide(&self, rate: f32, rng: &mut impl Rng) -> Self;
}

/// Used in crossover-reproducing [`next_gen`]s
#[cfg(all(feature = "crossover", not(feature = "tracing")))]
#[cfg_attr(docsrs, doc(cfg(feature = "crossover")))]
pub trait CrossoverReproduction {
    /// Use crossover reproduction to create a new genome.
    fn crossover(&self, other: &Self, rate: f32, rng: &mut impl Rng) -> Self;
}

/// Used in crossover-reproducing [`next_gen`]s
#[cfg(all(feature = "crossover", feature = "tracing"))]
#[cfg_attr(docsrs, doc(cfg(feature = "crossover")))]
pub trait CrossoverReproduction: std::fmt::Debug {
    /// Use crossover reproduction to create a new genome.
    fn crossover(&self, other: &Self, rate: f32, rng: &mut impl Rng) -> Self;
}

/// Used in speciated crossover nextgens. Allows for genomes to avoid crossover with ones that are too dissimilar.
#[cfg(all(feature = "speciation", not(feature = "tracing")))]
#[cfg_attr(docsrs, doc(cfg(feature = "speciation")))]
pub trait Speciated: Sized {
    /// Calculates whether two genomes are similar enough to be considered part of the same species.
    fn is_same_species(&self, other: &Self) -> bool;

    /// Filters a list of genomes based on whether they are of the same species.
    fn filter_same_species<'a>(&'a self, genomes: &'a [Self]) -> Vec<&Self> {
        genomes.iter().filter(|g| self.is_same_species(g)).collect()
    }
}

/// Used in speciated crossover nextgens. Allows for genomes to avoid crossover with ones that are too dissimilar.
#[cfg(all(feature = "speciation", feature = "tracing"))]
#[cfg_attr(docsrs, doc(cfg(feature = "speciation")))]
pub trait Speciated: Sized + std::fmt::Debug {
    /// Calculates whether two genomes are similar enough to be considered part of the same species.
    fn is_same_species(&self, other: &Self) -> bool;

    /// Filters a list of genomes based on whether they are of the same species.
    fn filter_same_species<'a>(&self, genomes: &'a [Self]) -> Vec<&'a Self> {
        genomes.iter().filter(|g| self.is_same_species(g)).collect()
    }
}

pub struct DivisionRepopulator<G: DivisionReproduction> {
    /// The mutation rate to use when mutating genomes. 0.0 - 1.0
    pub mutation_rate: f32,
    _marker: std::marker::PhantomData<G>,
}

impl<G: DivisionReproduction> DivisionRepopulator<G> {
    /// Creates a new [`DivisionRepopulator`].
    pub fn new(mutation_rate: f32) -> Self {
        Self {
            mutation_rate,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<G> Repopulator<G> for DivisionRepopulator<G>
where
    G: DivisionReproduction
{
    fn repopulate(&self, genomes: &mut Vec<G>, target_size: usize) {
        let mut rng = rand::rng();
        let mut champions = genomes.clone().iter().cycle();

        // TODO maybe rayonify
        while genomes.len() < target_size {
            let parent = champions.next().unwrap();
            let child = parent.divide(self.mutation_rate, &mut rng);
            genomes.push(child);
        }
    }
}

pub struct CrossoverRepopulator<G: CrossoverReproduction> {
    /// The mutation rate to use when mutating genomes. 0.0 - 1.0
    pub mutation_rate: f32,
    _marker: std::marker::PhantomData<G>,
}

impl<G: CrossoverReproduction> CrossoverRepopulator<G> {
    /// Creates a new [`CrossoverRepopulator`].
    pub fn new(mutation_rate: f32) -> Self {
        Self {
            mutation_rate,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<G> Repopulator<G> for CrossoverRepopulator<G>
where
    G: CrossoverReproduction,
{
    fn repopulate(&self, genomes: &mut Vec<G>, target_size: usize) {
        let mut rng = rand::rng();
        let champions = genomes.clone();
        let mut champs_cycle = champions.iter().cycle();

        // TODO maybe rayonify
        while genomes.len() < target_size {
            let parent1 = champions.next().unwrap();
            let parent2 = &champions[rng.random_range(0..champions.len() - 1)];

            if parent1 == parent2 {
                continue;
            }

            let child = parent1.crossover(parent2, &mut rng);
            genomes.push(child);
        }
    }
}
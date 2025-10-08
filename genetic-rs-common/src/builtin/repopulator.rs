use rand::Rng as RandRng;

use crate::Repopulator;

/// Used in other traits to randomly mutate genomes a given amount
pub trait RandomlyMutable {
    /// Mutate the genome with a given mutation rate (0..1)
    fn mutate(&mut self, rate: f32, rng: &mut impl rand::Rng);
}

/// Used in dividually-reproducing [`Repopulator`]s
pub trait Mitosis: Clone + RandomlyMutable {
    /// Create a new child with mutation. Similar to [RandomlyMutable::mutate], but returns a new instance instead of modifying the original.
    fn divide(&self, rate: f32, rng: &mut impl rand::Rng) -> Self {
        let mut child = self.clone();
        child.mutate(rate, rng);
        child
    }
}

/// Repopulator that uses division reproduction to create new genomes.
pub struct MitosisRepopulator<G: Mitosis> {
    /// The mutation rate to use when mutating genomes. 0.0 - 1.0
    pub mutation_rate: f32,
    _marker: std::marker::PhantomData<G>,
}

impl<G: Mitosis> MitosisRepopulator<G> {
    /// Creates a new [`MitosisRepopulator`].
    pub fn new(mutation_rate: f32) -> Self {
        Self {
            mutation_rate,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<G> Repopulator<G> for MitosisRepopulator<G>
where
    G: Mitosis,
{
    fn repopulate(&self, genomes: &mut Vec<G>, target_size: usize) {
        let mut rng = rand::rng();
        let champions = genomes.clone();
        let mut champs_cycle = champions.iter().cycle();

        // TODO maybe rayonify
        while genomes.len() < target_size {
            let parent = champs_cycle.next().unwrap();
            let child = parent.divide(self.mutation_rate, &mut rng);
            genomes.push(child);
        }
    }
}

#[cfg(feature = "crossover")]
mod crossover {
    use super::*;

    /// Used in crossover-reproducing [`Repopulator`]s
    #[cfg_attr(docsrs, doc(cfg(feature = "crossover")))]
    pub trait Crossover: Clone {
        /// Use crossover reproduction to create a new genome.
        fn crossover(&self, other: &Self, rate: f32, rng: &mut impl rand::Rng) -> Self;
    }

    /// Repopulator that uses crossover reproduction to create new genomes.
    #[cfg_attr(docsrs, doc(cfg(feature = "crossover")))]
    pub struct CrossoverRepopulator<G: Crossover> {
        /// The mutation rate to use when mutating genomes. 0.0 - 1.0
        pub mutation_rate: f32,
        _marker: std::marker::PhantomData<G>,
    }

    impl<G: Crossover> CrossoverRepopulator<G> {
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
        G: Crossover,
    {
        fn repopulate(&self, genomes: &mut Vec<G>, target_size: usize) {
            let mut rng = rand::rng();
            let champions = genomes.clone();
            let mut champs_cycle = champions.iter().enumerate().cycle();

            // TODO maybe rayonify
            while genomes.len() < target_size {
                let (i, parent1) = champs_cycle.next().unwrap();
                let mut j = rng.random_range(1..champions.len());
                if i == j {
                    j = 0;
                }
                let parent2 = &genomes[j];

                let child = parent1.crossover(parent2, self.mutation_rate, &mut rng);

                genomes.push(child);
            }
        }
    }
}

#[cfg(feature = "crossover")]
pub use crossover::*;

#[cfg(feature = "speciation")]
mod speciation {
    use super::*;

    /// Used in speciated crossover nextgens. Allows for genomes to avoid crossover with ones that are too different.
    #[cfg_attr(docsrs, doc(cfg(feature = "speciation")))]
    pub trait Speciated: Sized {
        /// Calculates whether two genomes are similar enough to be considered part of the same species.
        fn is_same_species(&self, other: &Self) -> bool;

        /// Filters a list of genomes based on whether they are of the same species.
        fn filter_same_species<'a>(&'a self, genomes: &'a [Self]) -> Vec<&'a Self> {
            genomes.iter().filter(|g| self.is_same_species(g)).collect()
        }
    }

    /// Repopulator that uses crossover reproduction to create new genomes, but only between genomes of the same species.
    #[cfg_attr(docsrs, doc(cfg(feature = "speciation")))]
    pub struct SpeciatedCrossoverRepopulator<G: Crossover + Speciated + PartialEq> {
        /// The mutation rate to use when mutating genomes. 0.0 - 1.0
        pub mutation_rate: f32,
        _marker: std::marker::PhantomData<G>,
    }

    impl<G: Crossover + Speciated + PartialEq> SpeciatedCrossoverRepopulator<G> {
        /// Creates a new [`SpeciatedCrossoverRepopulator`].
        pub fn new(mutation_rate: f32) -> Self {
            Self {
                mutation_rate,
                _marker: std::marker::PhantomData,
            }
        }
    }

    impl<G> Repopulator<G> for SpeciatedCrossoverRepopulator<G>
    where
        G: Crossover + Speciated + PartialEq,
    {
        fn repopulate(&self, genomes: &mut Vec<G>, target_size: usize) {
            let mut rng = rand::rng();
            let champions = genomes.clone();
            let mut champs_cycle = champions.iter().cycle();

            // TODO maybe rayonify
            while genomes.len() < target_size {
                let parent1 = champs_cycle.next().unwrap();
                let mut parent2 = &champions[rng.random_range(0..champions.len() - 1)];

                while parent1 == parent2 || !parent1.is_same_species(parent2) {
                    // TODO panic or eliminate if this parent cannot find another survivor in the same species
                    parent2 = &champions[rng.random_range(0..champions.len() - 1)];
                }

                let child = parent1.crossover(parent2, self.mutation_rate, &mut rng);

                genomes.push(child);
            }
        }
    }
}

#[cfg(feature = "speciation")]
pub use speciation::*;

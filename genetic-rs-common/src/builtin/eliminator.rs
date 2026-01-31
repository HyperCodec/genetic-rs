use std::ops::Not;

use crate::Eliminator;
use crate::FeatureBoundedGenome;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// A trait for fitness functions. This allows for more flexibility in defining fitness functions.
/// Any `Fn(&G) -> f32` can be used as a fitness function.
pub trait FitnessFn<G> {
    /// Evaluates a genome's fitness
    fn fitness(&self, genome: &G) -> f32;
}

impl<G, F> FitnessFn<G> for F
where
    F: Fn(&G) -> f32,
{
    fn fitness(&self, genome: &G) -> f32 {
        (self)(genome)
    }
}

#[doc(hidden)]
#[cfg(not(feature = "rayon"))]
pub trait FeatureBoundedFitnessFn<G: FeatureBoundedGenome>: FitnessFn<G> {}
#[cfg(not(feature = "rayon"))]
impl<G: FeatureBoundedGenome, T: FitnessFn<G>> FeatureBoundedFitnessFn<G> for T {}

#[doc(hidden)]
#[cfg(feature = "rayon")]
pub trait FeatureBoundedFitnessFn<G: FeatureBoundedGenome>: FitnessFn<G> + Send + Sync {}
#[cfg(feature = "rayon")]
impl<G: FeatureBoundedGenome, T: FitnessFn<G> + Send + Sync> FeatureBoundedFitnessFn<G> for T {}

/// A fitness-based eliminator that eliminates genomes based on their fitness scores.
pub struct FitnessEliminator<F: FitnessFn<G>, G: FeatureBoundedGenome> {
    /// The fitness function used to evaluate genomes.
    pub fitness_fn: F,

    /// The percentage of genomes to keep. Must be between 0.0 and 1.0.
    pub threshold: f32,

    _marker: std::marker::PhantomData<G>,
}

impl<F, G> FitnessEliminator<F, G>
where
    F: FeatureBoundedFitnessFn<G>,
    G: FeatureBoundedGenome,
{
    /// Creates a new [`FitnessEliminator`] with a given fitness function and threshold.
    /// Panics if the threshold is not between 0.0 and 1.0.
    pub fn new(fitness_fn: F, threshold: f32) -> Self {
        if !(0.0..=1.0).contains(&threshold) {
            panic!("Threshold must be between 0.0 and 1.0");
        }
        Self {
            fitness_fn,
            threshold,
            _marker: std::marker::PhantomData,
        }
    }

    /// Creates a new [`FitnessEliminator`] with a default threshold of 0.5 (all genomes below median fitness are eliminated).
    pub fn new_with_default(fitness_fn: F) -> Self {
        Self::new(fitness_fn, 0.5)
    }

    /// Calculates the fitness of each genome and sorts them by fitness.
    /// Returns a vector of tuples containing the genome and its fitness score.
    #[cfg(not(feature = "rayon"))]
    pub fn calculate_and_sort(&self, genomes: Vec<G>) -> Vec<(G, f32)> {
        let mut fitnesses: Vec<(G, f32)> = genomes
            .into_iter()
            .map(|g| {
                let fit = self.fitness_fn.fitness(&g);
                (g, fit)
            })
            .collect();
        fitnesses.sort_by(|(_a, afit), (_b, bfit)| bfit.partial_cmp(afit).unwrap());
        fitnesses
    }

    /// Calculates the fitness of each genome and sorts them by fitness.
    /// Returns a vector of tuples containing the genome and its fitness score.
    #[cfg(feature = "rayon")]
    pub fn calculate_and_sort(&self, genomes: Vec<G>) -> Vec<(G, f32)> {
        let mut fitnesses: Vec<(G, f32)> = genomes
            .into_par_iter()
            .map(|g| {
                let fit = self.fitness_fn.fitness(&g);
                (g, fit)
            })
            .collect();
        fitnesses.sort_by(|(_a, afit), (_b, bfit)| bfit.partial_cmp(afit).unwrap());
        fitnesses
    }
}

impl<F, G> Eliminator<G> for FitnessEliminator<F, G>
where
    F: FeatureBoundedFitnessFn<G>,
    G: FeatureBoundedGenome,
{
    #[cfg(not(feature = "rayon"))]
    fn eliminate(&self, genomes: Vec<G>) -> Vec<G> {
        let mut fitnesses = self.calculate_and_sort(genomes);
        let median_index = (fitnesses.len() as f32) * self.threshold;
        fitnesses.truncate(median_index as usize + 1);
        fitnesses.into_iter().map(|(g, _)| g).collect()
    }

    #[cfg(feature = "rayon")]
    fn eliminate(&self, genomes: Vec<G>) -> Vec<G> {
        let mut fitnesses = self.calculate_and_sort(genomes);
        let median_index = (fitnesses.len() as f32) * self.threshold;
        fitnesses.truncate(median_index as usize + 1);
        fitnesses.into_par_iter().map(|(g, _)| g).collect()
    }
}

#[cfg(feature = "knockout")]
mod knockout {
    use std::cmp::Ordering;

    use super::*;

    /// A distinct type to help clarify the result of a knockout function.
    #[cfg_attr(docsrs, doc(cfg(feature = "knockout")))]
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub enum KnockoutWinner {
        /// The first genome parameter won.
        First,

        /// The second genome parameter won.
        Second,
    }

    impl Into<usize> for KnockoutWinner {
        fn into(self) -> usize {
            match self {
                Self::First => 0,
                Self::Second => 1,
            }
        }
    }

    impl Not for KnockoutWinner {
        type Output = Self;

        fn not(self) -> Self::Output {
            match self {
                Self::First => Self::Second,
                Self::Second => Self::First,
            }
        }
    }

    impl From<Ordering> for KnockoutWinner {
        fn from(ordering: Ordering) -> Self {
            match ordering {
                Ordering::Less | Ordering::Equal => Self::First,
                Ordering::Greater => Self::Second,
            }
        }
    }

    /// A function that pits two genomes against each other and determines a winner.
    #[cfg_attr(docsrs, doc(cfg(feature = "knockout")))]
    pub trait KnockoutFn<G> {
        /// Tests the genomes to figure out who wins.
        fn knockout(&self, a: &G, b: &G) -> KnockoutWinner;
    }

    impl<G, F> KnockoutFn<G> for F
    where
        F: Fn(&G, &G) -> KnockoutWinner,
    {
        fn knockout(&self, a: &G, b: &G) -> KnockoutWinner {
            (self)(a, b)
        }
    }

    #[doc(hidden)]
    #[cfg(not(feature = "rayon"))]
    pub trait FeatureBoundedKnockoutFn<G>: KnockoutFn<G> {}
    #[cfg(not(feature = "rayon"))]
    impl<G, T: KnockoutFn<G>> FeatureBoundedKnockoutFn<G> for T {}

    #[doc(hidden)]
    #[cfg(feature = "rayon")]
    pub trait FeatureBoundedKnockoutFn<G>: KnockoutFn<G> + Send + Sync {}

    #[cfg(feature = "rayon")]
    impl<G, T: KnockoutFn<G> + Send + Sync> FeatureBoundedKnockoutFn<G> for T {}

    /// The action a knockout eliminator should take if the number of genomes is odd.
    #[cfg_attr(docsrs, doc(cfg(feature = "knockout")))]
    pub enum ActionIfOdd {
        /// Always expect an even number, crash if odd.
        Panic,

        /// Eliminate one of the genomes without checking it to
        /// bring it back to even, then proceed with normal knockout.
        DeleteSingle,

        /// Preserve one of the genomes without checking it to bring
        /// it back to even, then proceed with normal knockout.
        KeepSingle,
    }

    impl ActionIfOdd {
        pub(crate) fn exec<G>(
            &self,
            rng: &mut impl rand::Rng,
            genomes: &mut Vec<G>,
            output: &mut Vec<G>,
        ) {
            match self {
                Self::Panic => panic!("Knockout eliminator received an odd number of genomes"),
                Self::DeleteSingle => {
                    genomes.remove(rng.random_range(0..genomes.len()));
                }
                Self::KeepSingle => output.push(genomes.remove(rng.random_range(0..genomes.len()))),
            };
        }
    }

    /// Eliminator that pits genomes against each other and eliminates the weaker ones.
    #[cfg_attr(docsrs, doc(cfg(feature = "knockout")))]
    pub struct KnockoutEliminator<G: FeatureBoundedGenome, K: KnockoutFn<G>> {
        /// The function that determines the winner of a pair of genomes.
        pub knockout_fn: K,

        /// The action the eliminator should take if there is an odd number of genomes.
        pub action_if_odd: ActionIfOdd,

        _marker: std::marker::PhantomData<G>,
    }

    impl<G, K> KnockoutEliminator<G, K>
    where
        G: FeatureBoundedGenome,
        K: FeatureBoundedKnockoutFn<G>,
    {
        /// Creates a new [`KnockoutEliminator`]
        pub fn new(knockout_fn: K, action_if_odd: ActionIfOdd) -> Self {
            Self {
                knockout_fn,
                action_if_odd,
                _marker: std::marker::PhantomData,
            }
        }
    }

    impl<G, K> Eliminator<G> for KnockoutEliminator<G, K>
    where
        G: FeatureBoundedGenome,
        K: FeatureBoundedKnockoutFn<G>,
    {
        fn eliminate(&self, mut genomes: Vec<G>) -> Vec<G> {
            let len = genomes.len();

            if len < 2 {
                return genomes;
            }

            let mut rng = rand::rng();
            let mut output = Vec::with_capacity(genomes.len() / 2);

            if len % 2 != 0 {
                self.action_if_odd.exec(&mut rng, &mut genomes, &mut output);
            }

            debug_assert!(genomes.len() % 2 == 0);

            #[cfg(not(feature = "rayon"))]
            {
                use itertools::Itertools;

                output.extend(
                    genomes
                        .drain(..)
                        .tuples()
                        .map(|(a, b)| match self.knockout_fn.knockout(&a, &b) {
                            KnockoutWinner::First => a,
                            KnockoutWinner::Second => b,
                        })
                        .collect::<Vec<_>>(),
                );
            }

            #[cfg(feature = "rayon")]
            {
                output.extend(
                    genomes
                        .par_drain(..)
                        .chunks(2)
                        .map(|mut c| {
                            c.remove(<KnockoutWinner as Into<usize>>::into(
                                self.knockout_fn.knockout(&c[0], &c[1]),
                            ))
                        })
                        .collect::<Vec<_>>(),
                );
            }

            output
        }
    }
}

#[cfg(feature = "knockout")]
pub use knockout::*;

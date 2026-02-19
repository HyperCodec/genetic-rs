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

/// A trait for observing fitness scores. This can be used to implement things like logging or statistics collection.
pub trait FitnessObserver<G> {
    /// Observes the fitness scores of a generation of genomes.
    fn observe(&self, fitnesses: &[(G, f32)]);
}

impl<G> FitnessObserver<G> for () {
    fn observe(&self, _fitnesses: &[(G, f32)]) {}
}

#[cfg(not(feature = "rayon"))]
#[doc(hidden)]
pub trait FeatureBoundedFitnessObserver<G: FeatureBoundedGenome>: FitnessObserver<G> {}

#[cfg(not(feature = "rayon"))]
impl<G: FeatureBoundedGenome, T: FitnessObserver<G>> FeatureBoundedFitnessObserver<G> for T {}

#[cfg(feature = "rayon")]
#[doc(hidden)]
pub trait FeatureBoundedFitnessObserver<G: FeatureBoundedGenome>:
    FitnessObserver<G> + Send + Sync
{
}
#[cfg(feature = "rayon")]
impl<G: FeatureBoundedGenome, T: FitnessObserver<G> + Send + Sync> FeatureBoundedFitnessObserver<G>
    for T
{
}

/// A fitness-based eliminator that eliminates genomes based on their fitness scores.
pub struct FitnessEliminator<
    F: FeatureBoundedFitnessFn<G>,
    G: FeatureBoundedGenome,
    O: FeatureBoundedFitnessObserver<G> = (),
> {
    /// The fitness function used to evaluate genomes.
    pub fitness_fn: F,

    /// The percentage of genomes to keep. Must be between 0.0 and 1.0.
    pub threshold: f32,

    /// The fitness observer used to observe fitness scores.
    pub observer: O,

    _marker: std::marker::PhantomData<G>,
}

impl<F, G, O> FitnessEliminator<F, G, O>
where
    F: FeatureBoundedFitnessFn<G>,
    G: FeatureBoundedGenome,
    O: FeatureBoundedFitnessObserver<G>,
{
    /// The default threshold for the [`FitnessEliminator`]. This is the percentage of genomes to keep. All genomes below the median fitness will be eliminated.
    pub const DEFAULT_THRESHOLD: f32 = 0.5;

    /// Creates a new [`FitnessEliminator`] with a given fitness function and threshold.
    /// Panics if the threshold is not between 0.0 and 1.0.
    pub fn new(fitness_fn: F, threshold: f32, observer: O) -> Self {
        if !(0.0..=1.0).contains(&threshold) {
            panic!("Threshold must be between 0.0 and 1.0");
        }
        Self {
            fitness_fn,
            threshold,
            observer,
            _marker: std::marker::PhantomData,
        }
    }

    /// Creates a new [`FitnessEliminator`] with a default threshold of 0.5 (all genomes below median fitness are eliminated).
    pub fn new_with_default_threshold(fitness_fn: F, observer: O) -> Self {
        Self::new(fitness_fn, Self::DEFAULT_THRESHOLD, observer)
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

    /// Creates a new builder for [`FitnessEliminator`] to make it easier to construct with default parameters.
    pub fn builder() -> FitnessEliminatorBuilder<F, G, O> {
        FitnessEliminatorBuilder::default()
    }
}

impl<F, G, O> FitnessEliminator<F, G, O>
where
    F: FeatureBoundedFitnessFn<G>,
    G: FeatureBoundedGenome,
    O: FeatureBoundedFitnessObserver<G> + Default,
{
    /// Creates a new [`FitnessEliminator`] with a default observer that does nothing.
    pub fn new_with_default_observer(fitness_fn: F, threshold: f32) -> Self {
        Self::new(fitness_fn, threshold, O::default())
    }

    /// Creates a new [`FitnessEliminator`] with a default threshold of 0.5 and a default observer.
    /// You must specify the observer type explicitly, e.g., `FitnessEliminator::new_with_default::<()>(fitness_fn)`.
    pub fn new_with_default(fitness_fn: F) -> Self {
        Self::new_with_default_observer(fitness_fn, Self::DEFAULT_THRESHOLD)
    }
}

/// Implementation specifically for the unit type `()` observer (the default).
impl<F, G> FitnessEliminator<F, G, ()>
where
    F: FeatureBoundedFitnessFn<G>,
    G: FeatureBoundedGenome,
{
    /// Creates a new [`FitnessEliminator`] with a default threshold of 0.5 and unit observer `()`.
    /// This is a convenience function that doesn't require explicit type annotations.
    pub fn new_without_observer(fitness_fn: F) -> Self {
        Self::new(fitness_fn, Self::DEFAULT_THRESHOLD, ())
    }
}

impl<F, G, O> Eliminator<G> for FitnessEliminator<F, G, O>
where
    F: FeatureBoundedFitnessFn<G>,
    G: FeatureBoundedGenome,
    O: FeatureBoundedFitnessObserver<G>,
{
    #[cfg(not(feature = "rayon"))]
    fn eliminate(&self, genomes: Vec<G>) -> Vec<G> {
        let mut fitnesses = self.calculate_and_sort(genomes);
        let median_index = (fitnesses.len() as f32) * self.threshold;
        fitnesses.truncate(median_index as usize + 1);
        self.observer.observe(&fitnesses);
        fitnesses.into_iter().map(|(g, _)| g).collect()
    }

    #[cfg(feature = "rayon")]
    fn eliminate(&self, genomes: Vec<G>) -> Vec<G> {
        let mut fitnesses = self.calculate_and_sort(genomes);
        let median_index = (fitnesses.len() as f32) * self.threshold;
        fitnesses.truncate(median_index as usize + 1);
        self.observer.observe(&fitnesses);
        fitnesses.into_par_iter().map(|(g, _)| g).collect()
    }
}

/// A builder for [`FitnessEliminator`] to make it easier to construct with default parameters.
pub struct FitnessEliminatorBuilder<F: FitnessFn<G>, G, O: FitnessObserver<G> = ()> {
    fitness_fn: Option<F>,
    threshold: f32,
    observer: Option<O>,
    _marker: std::marker::PhantomData<G>,
}

impl<F, G, O> FitnessEliminatorBuilder<F, G, O>
where
    F: FeatureBoundedFitnessFn<G>,
    G: FeatureBoundedGenome,
    O: FeatureBoundedFitnessObserver<G>,
{
    /// Sets the fitness function for the [`FitnessEliminator`].
    pub fn fitness_fn(mut self, fitness_fn: F) -> Self {
        self.fitness_fn = Some(fitness_fn);
        self
    }

    /// Sets the threshold for the [`FitnessEliminator`].
    pub fn threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Sets the observer for the [`FitnessEliminator`].
    pub fn observer(mut self, observer: O) -> Self {
        self.observer = Some(observer);
        self
    }

    /// Builds the [`FitnessEliminator`].
    /// Panics if the fitness function or observer was not set.
    pub fn build_or_panic(self) -> FitnessEliminator<F, G, O> {
        let fitness_fn = self.fitness_fn.expect("Fitness function must be set");
        let observer = self.observer.expect(
            "Observer must be set. Use build_or_default() if the observer implements Default.",
        );
        FitnessEliminator::new(fitness_fn, self.threshold, observer)
    }
}

impl<F, G, O> FitnessEliminatorBuilder<F, G, O>
where
    F: FeatureBoundedFitnessFn<G>,
    G: FeatureBoundedGenome,
    O: FeatureBoundedFitnessObserver<G> + Default,
{
    /// Builds the [`FitnessEliminator`].
    /// If no observer was set, uses the default observer implementation.
    /// This method is only available when the observer type implements [`Default`].
    pub fn build(self) -> FitnessEliminator<F, G, O> {
        let fitness_fn = self.fitness_fn.expect("Fitness function must be set");
        let observer = self.observer.unwrap_or_default();
        FitnessEliminator::new(fitness_fn, self.threshold, observer)
    }
}

impl<F, G, O> Default for FitnessEliminatorBuilder<F, G, O>
where
    F: FeatureBoundedFitnessFn<G>,
    G: FeatureBoundedGenome,
    O: FeatureBoundedFitnessObserver<G>,
{
    fn default() -> Self {
        Self {
            fitness_fn: None,
            threshold: 0.5,
            observer: None,
            _marker: std::marker::PhantomData,
        }
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

    impl From<KnockoutWinner> for usize {
        fn from(winner: KnockoutWinner) -> Self {
            match winner {
                KnockoutWinner::First => 0,
                KnockoutWinner::Second => 1,
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

    /// A knockout function that uses a fitness function to determine the winner.
    pub struct FitnessKnockoutFn<F: FitnessFn<G>, G: FeatureBoundedGenome> {
        /// The fitness function used to evaluate the genomes.
        pub fitness_fn: F,
        _marker: std::marker::PhantomData<G>,
    }

    impl<F: FitnessFn<G>, G: FeatureBoundedGenome> FitnessKnockoutFn<F, G> {
        /// Creates a new [`FitnessKnockoutFn`] with a given fitness function.
        pub fn new(fitness_fn: F) -> Self {
            Self {
                fitness_fn,
                _marker: std::marker::PhantomData,
            }
        }
    }

    impl<F, G> KnockoutFn<G> for FitnessKnockoutFn<F, G>
    where
        F: FeatureBoundedFitnessFn<G>,
        G: FeatureBoundedGenome,
    {
        fn knockout(&self, a: &G, b: &G) -> KnockoutWinner {
            let afit = self.fitness_fn.fitness(a);
            let bfit = self.fitness_fn.fitness(b);
            afit.total_cmp(&bfit).into()
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

            if !len.is_multiple_of(2) {
                self.action_if_odd.exec(&mut rng, &mut genomes, &mut output);
            }

            debug_assert!(genomes.len().is_multiple_of(2));

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

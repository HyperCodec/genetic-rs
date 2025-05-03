use crate::Eliminator;

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

/// A fitness-based eliminator that eliminates genomes based on their fitness scores.
pub struct FitnessEliminator<F: FitnessFn<G>, G> {
    /// The fitness function used to evaluate genomes.
    pub fitness_fn: F,

    /// The percentage of genomes to keep. Must be between 0.0 and 1.0.
    pub threshold: f32,

    _marker: std::marker::PhantomData<G>,
}

impl<F: FitnessFn<G>, G> FitnessEliminator<F, G> {
    /// Creates a new [`FitnessEliminator`] with a given fitness function and threshold.
    /// Panics if the threshold is not between 0.0 and 1.0.
    pub fn new(fitness_fn: F, threshold: f32) -> Self {
        if !(0.0..=1.0).contains(&threshold) {
            panic!("Threshold must be between 0.0 and 1.0");
        }
        Self {
            fitness_fn,
            threshold,
            _marker: std::marker::PhantomData
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

impl<F: FitnessFn<G>, G> Eliminator<G> for FitnessEliminator<F, G> {
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
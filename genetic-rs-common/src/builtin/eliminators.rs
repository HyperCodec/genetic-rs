use crate::Eliminator;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

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

pub struct FitnessEliminator<F: FitnessFn<G>, G> {
    pub fitness_fn: F,

    /// The percentage of genomes to keep. Must be between 0.0 and 1.0.
    pub threshold: f32,

    _marker: std::marker::PhantomData<G>,
}

impl<F: FitnessFn<G>, G> FitnessEliminator<F, G> {
    /// Creates a new [`FitnessEliminator`] with a given fitness function and threshold.
    /// Panics if the threshold is not between 0.0 and 1.0.
    pub fn new(fitness_fn: F, threshold: f32) -> Self {
        if threshold < 0.0 || threshold > 1.0 {
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
}

impl<F: FitnessFn<G>, G> Eliminator<G> for FitnessEliminator<F, G> {
    #[cfg(not(feature = "rayon"))]
    fn eliminate(&self, genomes: Vec<G>) -> Vec<G> {
        let mut fitnesses: Vec<(G, f32)> = genomes.iter().map(|g| (g, self.fitness_fn.fitness(&g))).collect();
        fitnesses.sort_by(|(_a, afit), (_b, bfit)| afit.partial_cmp(bfit).unwrap());
        let median_index = (fitnesses.len() as f32) * self.threshold;
        fitnesses.truncate(median_index as usize + 1);
        fitnesses.into_iter().map(|(g, _)| g).collect()
    }

    #[cfg(feature = "rayon")]
    fn eliminate(&self, genomes: Vec<G>) -> Vec<G> {
        let mut fitnesses: Vec<(G, f32)> = genomes.into_par_iter().map(|g| (g, self.fitness_fn.fitness(&g))).collect();
        fitnesses.sort_by(|(_a, afit), (_b, bfit)| afit.partial_cmp(bfit).unwrap());
        let median_index = (fitnesses.len() as f32) * self.threshold;
        fitnesses.truncate(median_index as usize + 1);
        fitnesses.into_par_iter().map(|(g, _)| g).collect()
    }
}

// TODO  `ObservedFitnessEliminator` that sends the `fitnesses` to observer(s) before truncating
use crate::Repopulator;

/// Used in other traits to randomly mutate genomes a given amount
pub trait RandomlyMutable {
    /// Simulation-wide context required for this mutation implementation.
    type Context;

    /// Mutate the genome with a given mutation rate (0..1)
    fn mutate(&mut self, ctx: &Self::Context, rate: f32, rng: &mut impl rand::Rng);
}

// TODO rayon version
impl<'a, T: RandomlyMutable + 'a, I: Iterator<Item = &'a mut T>> RandomlyMutable for I {
    type Context = T::Context;

    fn mutate(&mut self, ctx: &Self::Context, rate: f32, rng: &mut impl rand::Rng) {
        self.for_each(|x| x.mutate(ctx, rate, rng));
    }
}

/// Used in dividually-reproducing [`Repopulator`]s
pub trait Mitosis: Clone {
    /// Simulation-wide context required for this mitosis implementation.
    type Context;

    /// Create a new child with mutation. Similar to [`RandomlyMutable::mutate`], but returns a new instance instead of modifying the original.
    fn divide(&self, ctx: &<Self as Mitosis>::Context, rate: f32, rng: &mut impl rand::Rng)
        -> Self;
}

impl<T: Mitosis> Mitosis for Vec<T> {
    type Context = T::Context;

    fn divide(
        &self,
        ctx: &<Self as Mitosis>::Context,
        rate: f32,
        rng: &mut impl rand::Rng,
    ) -> Self {
        let mut child = Vec::with_capacity(self.len());
        for gene in self {
            child.push(gene.divide(ctx, rate, rng));
        }
        child
    }
}

/// Repopulator that uses division reproduction to create new genomes.
pub struct MitosisRepopulator<G: Mitosis> {
    /// The mutation rate to use when mutating genomes. 0.0 - 1.0
    pub mutation_rate: f32,

    /// The context to use when mutating genomes.
    pub ctx: G::Context,
    _marker: std::marker::PhantomData<G>,
}

impl<G: Mitosis> MitosisRepopulator<G> {
    /// Creates a new [`MitosisRepopulator`].
    pub fn new(mutation_rate: f32, ctx: G::Context) -> Self {
        Self {
            mutation_rate,
            ctx,
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
            let child = parent.divide(&self.ctx, self.mutation_rate, &mut rng);
            genomes.push(child);
        }
    }
}

#[cfg(feature = "crossover")]
mod crossover {
    use rand::RngExt;

    use super::*;

    /// Used in crossover-reproducing [`Repopulator`]s
    #[cfg_attr(docsrs, doc(cfg(feature = "crossover")))]
    pub trait Crossover: Clone {
        /// Simulation-wide context required for this crossover implementation.
        type Context;

        /// Use crossover reproduction to create a new genome.
        fn crossover(
            &self,
            other: &Self,
            ctx: &Self::Context,
            rate: f32,
            rng: &mut impl rand::Rng,
        ) -> Self;
    }

    /// Repopulator that uses crossover reproduction to create new genomes.
    #[cfg_attr(docsrs, doc(cfg(feature = "crossover")))]
    pub struct CrossoverRepopulator<G: Crossover> {
        /// The mutation rate to use when mutating genomes. 0.0 - 1.0
        pub mutation_rate: f32,

        /// Additional context for crossover/mutation.
        pub ctx: G::Context,
        _marker: std::marker::PhantomData<G>,
    }

    impl<G: Crossover> CrossoverRepopulator<G> {
        /// Creates a new [`CrossoverRepopulator`].
        pub fn new(mutation_rate: f32, ctx: G::Context) -> Self {
            Self {
                mutation_rate,
                ctx,
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

                let child = parent1.crossover(parent2, &self.ctx, self.mutation_rate, &mut rng);

                genomes.push(child);
            }
        }
    }
}

#[cfg(feature = "crossover")]
pub use crossover::*;

#[cfg(feature = "speciation")]
mod speciation {
    use std::collections::HashMap;

    use rand::RngExt;

    use super::*;

    /// Used in speciated crossover nextgens. Allows for genomes to avoid crossover with ones that are too different.
    #[cfg_attr(docsrs, doc(cfg(feature = "speciation")))]
    pub trait Speciated {
        /// The type used to distinguish
        /// one genome's species from another.
        type Species: Eq + std::hash::Hash; // I really don't like that we need `Eq` when `PartialEq` better fits the definiton.

        /// Get/calculate this genome's species.
        fn species(&self) -> Self::Species;
    }

    /// Repopulator that uses crossover reproduction to create new genomes, but only between genomes of the same species.
    #[cfg_attr(docsrs, doc(cfg(feature = "speciation")))]
    pub struct SpeciatedCrossoverRepopulator<G: Crossover + Speciated> {
        /// The inner crossover repopulator. This holds the settings for crossover operations,
        /// but may also be called if [`allow_emergency_repr`][Self::allow_emergency_repr] is `true`.
        pub crossover: CrossoverRepopulator<G>,

        /// Whether to allow genomes to reproduce across species boundaries
        /// (effectively vanilla crossover)
        /// in emergency situations where no genomes have compatible partners.
        /// If disabled, the simulation will panic in such a situation.
        pub allow_emergency_repr: bool,

        _marker: std::marker::PhantomData<G>,
    }

    impl<G: Crossover + Speciated> SpeciatedCrossoverRepopulator<G> {
        /// Creates a new [`SpeciatedCrossoverRepopulator`].
        pub fn new(mutation_rate: f32, allow_emergency_repr: bool, ctx: G::Context) -> Self {
            Self {
                crossover: CrossoverRepopulator::new(mutation_rate, ctx),
                allow_emergency_repr,
                _marker: std::marker::PhantomData,
            }
        }
    }

    impl<G> Repopulator<G> for SpeciatedCrossoverRepopulator<G>
    where
        G: Crossover + Speciated,
    {
        // i'm still not really satisfied with this implementation,
        // but it's better than the old one.
        fn repopulate(&self, genomes: &mut Vec<G>, target_size: usize) {
            let initial_size = genomes.len();
            let mut rng = rand::rng();
            let mut species: HashMap<<G as Speciated>::Species, Vec<&G>> = HashMap::new();

            for genome in genomes.iter() {
                let spec = genome.species();
                species.entry(spec).or_insert_with(Vec::new).push(genome);
            }

            let mut species_iter = species.values();
            let to_create = target_size - initial_size;
            let mut new_genomes = Vec::with_capacity(to_create);

            while new_genomes.len() < to_create {
                if let Some(spec) = species_iter.next() {
                    if spec.len() < 2 {
                        continue;
                    }

                    for (i, &parent1) in spec.iter().enumerate() {
                        let mut j = rng.random_range(1..spec.len());
                        if j == i {
                            j = 0;
                        }
                        let parent2 = spec[j];

                        new_genomes.push(parent1.crossover(
                            parent2,
                            &self.crossover.ctx,
                            self.crossover.mutation_rate,
                            &mut rng,
                        ));
                    }
                } else {
                    // reached the end, reset the iterator

                    if new_genomes.is_empty() {
                        // no genomes have compatible partners
                        if self.allow_emergency_repr {
                            self.crossover.repopulate(genomes, target_size);
                            return;
                        } else {
                            panic!("no genomes with common species");
                        }
                    }

                    species_iter = species.values();
                }
            }

            genomes.extend(new_genomes);
        }
    }
}

#[cfg(feature = "speciation")]
pub use speciation::*;

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
    fn repopulate(&mut self, genomes: &mut Vec<G>, target_size: usize) {
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
        fn repopulate(&mut self, genomes: &mut Vec<G>, target_size: usize) {
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

    impl<G> Default for CrossoverRepopulator<G>
    where
        G: Crossover,
        G::Context: Default,
    {
        fn default() -> Self {
            Self::new(0.05, G::Context::default())
        }
    }
}

#[cfg(feature = "crossover")]
pub use crossover::*;

#[cfg(feature = "speciation")]
mod speciation {
    use rand::RngExt;

    use crate::speciation::{Speciated, SpeciatedPopulation};

    use super::*;

    /// The action to take when a genome is found to be in a species by itself.
    /// This can be used to prevent species from going extinct due to bad luck in crossover.
    pub enum ActionIfIsolated {
        /// Do nothing, allowing the species to go extinct if the genome is not fit enough.
        /// Note that if all species have only one member (not likely, but possible),
        /// this can result in the repopulator hanging.
        DoNothing,

        /// Perform crossover between the genome and itself to create a new member of the species.
        /// This can help prevent species from going extinct, but can also lead to less diversity in the population.
        CrossoverSelf,

        /// Perform crossover between the genome and a random member of the most similar species to create a new member of the species.
        /// This can help prevent species from going extinct while maintaining more diversity than self-crossover, but can also lead to more computational overhead.
        CrossoverSimilarSpecies,

        /// Perform crossover between the genome and a random member of the entire population, ignoring species boundaries, to create a new member of the species.
        /// This can help prevent species from going extinct, but can also contribute to more broken or dysfunctional species.
        CrossoverRandom,
    }

    /// Repopulator that uses crossover reproduction to create new genomes, but only between genomes of the same species.
    pub struct SpeciatedCrossoverRepopulator<G: Crossover + Speciated> {
        /// The inner crossover repopulator. This holds the settings for crossover operations.
        pub inner: CrossoverRepopulator<G>,

        /// The threshold used to determine if a genome belongs in a species.
        /// See [`SpeciatedPopulation::threshold`] for more info.
        pub speciation_threshold: f32,

        /// The action to take when a genome is found to be in a species by itself.
        pub action_if_isolated: ActionIfIsolated,

        /// Additional context for speciation.
        pub ctx: <G as Speciated>::Context,

        _marker: std::marker::PhantomData<G>,
    }

    impl<G: Crossover + Speciated> SpeciatedCrossoverRepopulator<G> {
        /// Creates a new [`SpeciatedCrossoverRepopulator`].
        pub fn new(
            mutation_rate: f32,
            threshold: f32,
            action_if_isolated: ActionIfIsolated,
            crossover_ctx: <G as Crossover>::Context,
            spec_ctx: <G as Speciated>::Context,
        ) -> Self {
            Self {
                inner: CrossoverRepopulator::new(mutation_rate, crossover_ctx),
                ctx: spec_ctx,
                speciation_threshold: threshold,
                action_if_isolated,
                _marker: std::marker::PhantomData,
            }
        }

        /// Creates a new [`SpeciatedCrossoverRepopulator`] from an existing [`CrossoverRepopulator`], using the same mutation settings.
        pub fn from_crossover(
            inner: CrossoverRepopulator<G>,
            threshold: f32,
            action_if_isolated: ActionIfIsolated,
            ctx: <G as Speciated>::Context,
        ) -> Self {
            Self {
                inner,
                ctx,
                speciation_threshold: threshold,
                action_if_isolated,
                _marker: std::marker::PhantomData,
            }
        }
    }

    impl<G> Repopulator<G> for SpeciatedCrossoverRepopulator<G>
    where
        G: Crossover + Speciated,
    {
        fn repopulate(&mut self, genomes: &mut Vec<G>, target_size: usize) {
            let initial_size = genomes.len();
            if initial_size >= target_size {
                return;
            }

            let mut rng = rand::rng();
            let population =
                SpeciatedPopulation::from_genomes(genomes, self.speciation_threshold, &self.ctx);

            let amount_to_make = target_size - initial_size;
            let mut species_cycle = population.round_robin_enumerate();

            let mut i = 0;
            while i < amount_to_make {
                let (species_i, genome_i) = species_cycle.next().unwrap();
                let species = &population.species[species_i];
                let parent1 = &genomes[genome_i];
                if species.len() < 2 {
                    match self.action_if_isolated {
                        ActionIfIsolated::DoNothing => continue,
                        ActionIfIsolated::CrossoverSelf => {
                            let child = parent1.crossover(
                                parent1,
                                &self.inner.ctx,
                                self.inner.mutation_rate,
                                &mut rng,
                            );
                            genomes.push(child);
                            i += 1;
                            continue;
                        }
                        ActionIfIsolated::CrossoverSimilarSpecies => {
                            let mut best_species_i = 0;
                            let mut best_divergence = f32::MAX;
                            for (j, species) in population.species.iter().enumerate() {
                                if j == species_i || species.is_empty() {
                                    continue;
                                }
                                let representative = &genomes[species[0]];
                                let divergence = parent1.divergence(representative, &self.ctx);
                                if divergence < best_divergence {
                                    best_divergence = divergence;
                                    best_species_i = j;
                                }
                            }

                            let best_species = &population.species[best_species_i];
                            let j = rng.random_range(0..best_species.len());
                            let parent2 = &genomes[best_species[j]];
                            let child = parent1.crossover(
                                parent2,
                                &self.inner.ctx,
                                self.inner.mutation_rate,
                                &mut rng,
                            );
                            genomes.push(child);
                            i += 1;
                            continue;
                        }
                        ActionIfIsolated::CrossoverRandom => {
                            let mut j = rng.random_range(1..genomes.len());
                            if j == genome_i {
                                j = 0;
                            }

                            let parent2 = &genomes[j];
                            let child = parent1.crossover(
                                parent2,
                                &self.inner.ctx,
                                self.inner.mutation_rate,
                                &mut rng,
                            );
                            genomes.push(child);
                            i += 1;
                            continue;
                        }
                    }
                }

                let mut j = rng.random_range(1..species.len());
                if genome_i == species[j] {
                    j = 0;
                }
                let parent2 = &genomes[species[j]];

                let child =
                    parent1.crossover(parent2, &self.inner.ctx, self.inner.mutation_rate, &mut rng);
                genomes.push(child);

                i += 1;
            }
        }
    }

    impl<G> Default for SpeciatedCrossoverRepopulator<G>
    where
        G: Crossover + Speciated,
        <G as Crossover>::Context: Default,
        <G as Speciated>::Context: Default,
    {
        fn default() -> Self {
            Self::from_crossover(
                CrossoverRepopulator::default(),
                0.1,
                ActionIfIsolated::CrossoverSimilarSpecies,
                <G as Speciated>::Context::default(),
            )
        }
    }
}

#[cfg(feature = "speciation")]
pub use speciation::*;

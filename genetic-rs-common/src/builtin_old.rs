use crate::Rng;

// TODO clean up all this spaghetti and replace with eliminator and repopulator traits 

/// Provides some basic nextgens for [`GeneticSim`][crate::GeneticSim].
pub mod next_gen {
    use super::*;

    #[cfg(feature = "rayon")]
    use rayon::prelude::*;

    #[cfg(feature = "tracing")]
    use tracing::*;
    
    /// Prunes half of the genomes and randomly crosses over the remaining ones.
    #[cfg(all(feature = "crossover", not(feature = "rayon")))]
    #[cfg_attr(docsrs, doc(cfg(feature = "crossover")))]
    #[cfg_attr(feature = "tracing", instrument)]
    pub fn crossover_pruning_nextgen<E: CrossoverReproduction + Prunable + Clone + PartialEq>(
        rewards: Vec<(E, f32)>,
    ) -> Vec<E> {
        let population_size = rewards.len();
        let mut next_gen = pruning_helper(rewards);

        let mut rng = rand::rng();

        // TODO remove clone smh
        let og_champions = next_gen.clone();

        let mut og_champs_cycle = og_champions.iter().cycle();

        while next_gen.len() < population_size {
            let g1 = og_champs_cycle.next().unwrap();
            let g2 = &og_champions[rng.random_range(0..og_champions.len() - 1)];

            if g1 == g2 {
                continue;
            }

            #[cfg(feature = "tracing")]
            let span = span!(
                Level::DEBUG,
                "crossover",
                a = tracing::field::debug(g1),
                b = tracing::field::debug(g2)
            );
            #[cfg(feature = "tracing")]
            let enter = span.enter();

            next_gen.push(g1.crossover(g2, &mut rng));

            #[cfg(feature = "tracing")]
            drop(enter);
        }

        next_gen
    }

    /// Rayon version of the [`crossover_pruning_nextgen`] function.
    #[cfg(all(feature = "crossover", feature = "rayon",))]
    #[cfg_attr(feature = "tracing", instrument)]
    pub fn crossover_pruning_nextgen<
        G: CrossoverReproduction + Prunable + Clone + Send + PartialEq,
    >(
        rewards: Vec<(G, f32)>,
    ) -> Vec<G> {
        let population_size = rewards.len();
        let mut next_gen = pruning_helper(rewards);

        let mut rng = rand::rng();

        // TODO remove clone smh
        let og_champions = next_gen.clone();

        let mut og_champs_cycle = og_champions.iter().cycle();

        while next_gen.len() < population_size {
            let g1 = og_champs_cycle.next().unwrap();
            let g2 = &og_champions[rng.random_range(0..og_champions.len() - 1)];

            if g1 == g2 {
                continue;
            }

            #[cfg(feature = "tracing")]
            let span = span!(
                Level::DEBUG,
                "crossover",
                a = tracing::field::debug(g1),
                b = tracing::field::debug(g2)
            );
            #[cfg(feature = "tracing")]
            let enter = span.enter();

            next_gen.push(g1.crossover(g2, &mut rng));

            #[cfg(feature = "tracing")]
            drop(enter);

            next_gen.push(g1.crossover(g2, &mut rng));
        }

        next_gen
    }

    /// Similar to [`crossover_pruning_nextgen`], this nextgen will prune and then perform crossover reproduction.
    /// With this function, crossover reproduction will only occur if both genomes are of the same species, otherwise one will perform divison to reproduce.
    #[cfg(all(feature = "speciation", not(feature = "rayon")))]
    #[cfg_attr(docsrs, doc(cfg(feature = "specation")))]
    #[cfg_attr(feature = "tracing", instrument)]
    pub fn speciated_crossover_pruning_nextgen<
        'a,
        G: CrossoverReproduction + DivisionReproduction + Speciated + Prunable + Clone + PartialEq,
    >(
        rewards: Vec<(G, f32)>,
    ) -> Vec<G> {
        let population_size = rewards.len();
        let mut next_gen = pruning_helper(rewards);

        let mut rng = rand::rng();

        // TODO remove clone smh
        let og_champions = next_gen.clone();

        let mut og_champs_cycle = og_champions.iter().cycle();

        while next_gen.len() < population_size {
            let g1 = og_champs_cycle.next().unwrap();
            next_gen.push(species_helper(g1, &og_champions, &mut rng));
        }

        next_gen
    }

    /// Rayon version of [`speciated_crossover_pruning_nextgen`]
    #[cfg(all(feature = "speciation", feature = "rayon"))]
    #[cfg_attr(feature = "tracing", instrument)]
    pub fn speciated_crossover_pruning_nextgen<
        G: CrossoverReproduction
            + DivisionReproduction
            + Speciated
            + Prunable
            + Clone
            + Send
            + PartialEq,
    >(
        rewards: Vec<(G, f32)>,
    ) -> Vec<G> {
        let population_size = rewards.len();
        let mut next_gen = pruning_helper(rewards);

        let mut rng = rand::rng();

        // TODO remove clone smh
        let og_champions = next_gen.clone();

        let mut og_champs_cycle = og_champions.iter().cycle();

        while next_gen.len() < population_size {
            let g1 = og_champs_cycle.next().unwrap();
            next_gen.push(species_helper(g1, &og_champions, &mut rng));
        }

        next_gen
    }

    #[cfg(feature = "speciation")]
    #[cfg_attr(feature = "tracing", instrument)]
    fn species_helper<E: CrossoverReproduction + Speciated + DivisionReproduction>(
        genome: &E,
        genomes: &[E],
        rng: &mut impl Rng,
    ) -> E {
        let same_species = genome.filter_same_species(genomes);

        if same_species.is_empty() {
            // division if can't find any of the same species
            return genome.divide(rng);
        }

        // perform crossover reproduction with genomes of the same species
        let other = same_species[rng.random_range(0..same_species.len())];

        #[cfg(feature = "tracing")]
        let span = span!(
            Level::DEBUG,
            "crossover",
            a = tracing::field::debug(genome),
            b = tracing::field::debug(other)
        );
        #[cfg(feature = "tracing")]
        let enter = span.enter();

        let child = genome.crossover(other, rng);

        #[cfg(feature = "tracing")]
        drop(enter);

        child
    }

    /// helps with builtin pruning nextgens
    #[cfg(not(feature = "rayon"))]
    #[cfg_attr(feature = "tracing", instrument)]
    fn pruning_helper<E: Prunable + Clone>(mut rewards: Vec<(E, f32)>) -> Vec<E> {
        rewards.sort_by(|(_, r1), (_, r2)| r1.partial_cmp(r2).unwrap());

        let median = rewards[rewards.len() / 2].1;

        #[cfg(feature = "tracing")]
        debug!("median: {median}");

        rewards
            .into_iter()
            .filter_map(|(e, r)| {
                if r < median {
                    e.despawn();
                    return None;
                }

                Some(e)
            })
            .collect()
    }

    /// Rayon version of [`pruning_helper`].
    #[cfg(feature = "rayon")]
    #[cfg_attr(feature = "tracing", instrument)]
    fn pruning_helper<E: Prunable + Send>(mut rewards: Vec<(E, f32)>) -> Vec<E> {
        rewards.sort_by(|(_, r1), (_, r2)| r1.partial_cmp(r2).unwrap());

        let median = rewards[rewards.len() / 2].1;

        #[cfg(feature = "tracing")]
        debug!("median: {median}");

        rewards
            .into_par_iter()
            .filter_map(|(e, r)| {
                if r < median {
                    e.despawn();
                    return None;
                }

                Some(e)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    // TODO rewrite bc this is spaghetti asf
    use crate::prelude::*;

    #[derive(Default, Clone, Debug, PartialEq)]
    struct MyGenome(f32);

    impl RandomlyMutable for MyGenome {
        fn mutate(&mut self, rate: f32, rng: &mut impl rand::Rng) {
            self.0 += rng.random::<f32>() * rate;
        }
    }

    impl DivisionReproduction for MyGenome {
        fn divide(&self, rng: &mut impl Rng) -> Self {
            let mut child = self.clone();
            child.mutate(0.25, rng);
            child
        }
    }

    impl Prunable for MyGenome {
        fn despawn(self) {
            println!("RIP {:?}", self);
        }
    }

    impl GenerateRandom for MyGenome {
        fn gen_random(rng: &mut impl Rng) -> Self {
            Self(rng.random())
        }
    }

    #[cfg(feature = "crossover")]
    #[derive(Debug, Clone, PartialEq)]
    struct MyCrossoverGenome(MyGenome);

    #[cfg(feature = "crossover")]
    impl RandomlyMutable for MyCrossoverGenome {
        fn mutate(&mut self, rate: f32, rng: &mut impl Rng) {
            self.0.mutate(rate, rng);
        }
    }

    #[cfg(feature = "crossover")]
    impl CrossoverReproduction for MyCrossoverGenome {
        fn crossover(&self, other: &Self, rng: &mut impl Rng) -> Self {
            let mut child = Self(MyGenome((self.0 .0 + other.0 .0) / 2.));
            child.mutate(0.25, rng);
            child
        }
    }

    #[cfg(feature = "crossover")]
    impl Prunable for MyCrossoverGenome {}

    #[cfg(feature = "crossover")]
    impl GenerateRandom for MyCrossoverGenome {
        fn gen_random(rng: &mut impl Rng) -> Self {
            Self(MyGenome::gen_random(rng))
        }
    }

    #[cfg(feature = "speciation")]
    impl DivisionReproduction for MyCrossoverGenome {
        fn divide(&self, rng: &mut impl Rng) -> Self {
            Self(self.0.divide(rng))
        }
    }

    #[cfg(feature = "speciation")]
    impl Speciated for MyCrossoverGenome {
        fn is_same_species(&self, other: &Self) -> bool {
            (self.0 .0 - other.0 .0).abs() <= 2.
        }
    }

    const MAGIC_NUMBER: f32 = std::f32::consts::E;

    // #[cfg(not(feature = "crossover"))]
    fn my_fitness_fn(ent: &MyGenome) -> f32 {
        (MAGIC_NUMBER - ent.0).abs() * -1.
    }

    #[cfg(feature = "crossover")]
    fn my_crossover_fitness_fn(ent: &MyCrossoverGenome) -> f32 {
        (MAGIC_NUMBER - ent.0 .0).abs() * -1.
    }

    #[cfg(not(feature = "rayon"))]
    #[test]
    fn scramble() {
        let mut rng = rand::rng();
        let mut sim = GeneticSim::new(
            Vec::gen_random(&mut rng, 1000),
            #[cfg(feature = "crossover")]
            my_crossover_fitness_fn,
            #[cfg(not(feature = "crossover"))]
            my_fitness_fn,
            scrambling_nextgen,
        );

        sim.perform_generations(100);

        dbg!(sim.genomes);
    }

    #[cfg(feature = "rayon")]
    fn r_scramble() {
        let mut sim = GeneticSim::new(
            Vec::gen_random(1000),
            #[cfg(not(feature = "crossover"))]
            my_fitness_fn,
            #[cfg(feature = "crossover")]
            my_crossover_fitness_fn,
            scrambling_nextgen,
        );

        sim.perform_generations(100);

        dbg!(sim.genomes);
    }

    #[cfg(not(feature = "rayon"))]
    #[test]
    fn d_prune() {
        let mut rng = rand::rng();
        let mut sim = GeneticSim::new(
            Vec::gen_random(&mut rng, 1000),
            my_fitness_fn,
            division_pruning_nextgen,
        );

        sim.perform_generations(100);

        dbg!(sim.genomes);
    }

    #[cfg(all(feature = "crossover", not(feature = "rayon")))]
    #[test]
    fn c_prune() {
        let mut rng = rand::rng();

        let mut sim = GeneticSim::new(
            Vec::gen_random(&mut rng, 100),
            my_crossover_fitness_fn,
            crossover_pruning_nextgen,
        );

        sim.perform_generations(100);

        dbg!(sim.genomes);
    }

    #[cfg(all(feature = "crossover", feature = "rayon"))]
    #[test]
    fn cr_prune() {
        let mut sim = GeneticSim::new(
            Vec::gen_random(100),
            my_crossover_fitness_fn,
            crossover_pruning_nextgen,
        );

        sim.perform_generations(100);

        dbg!(sim.genomes);
    }

    #[cfg(all(feature = "speciation", not(feature = "rayon")))]
    #[test]
    fn sc_prune() {
        let mut rng = rand::rng();

        let mut sim = GeneticSim::new(
            Vec::gen_random(&mut rng, 100),
            my_crossover_fitness_fn,
            speciated_crossover_pruning_nextgen,
        );

        sim.perform_generations(100);

        dbg!(sim.genomes);
    }

    #[cfg(all(feature = "speciation", feature = "rayon"))]
    #[test]
    fn scr_prune() {
        let mut sim = GeneticSim::new(
            Vec::gen_random(100),
            my_crossover_fitness_fn,
            speciated_crossover_pruning_nextgen,
        );

        sim.perform_generations(100);

        dbg!(sim.genomes);
    }
}

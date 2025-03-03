/// Used in all of the builtin [`next_gen`]s to randomly mutate genomes a given amount
pub trait RandomlyMutable {
    /// Mutate the genome with a given mutation rate (0..1)
    fn mutate(&mut self, rate: f32, rng: &mut impl rand::Rng);
}

/// Used in dividually-reproducing [`next_gen`]s
pub trait DivisionReproduction {
    /// Create a new child with mutation. Similar to [RandomlyMutable::mutate], but returns a new instance instead of modifying the original.
    /// If it is simply returning a cloned and mutated version, consider using a constant mutation rate.
    fn divide(&self, rng: &mut impl rand::Rng) -> Self;
}

/// Used in crossover-reproducing [`next_gen`]s
#[cfg(feature = "crossover")]
#[cfg_attr(docsrs, doc(cfg(feature = "crossover")))]
pub trait CrossoverReproduction {
    /// Use crossover reproduction to create a new genome.
    fn crossover(&self, other: &Self, rng: &mut impl rand::Rng) -> Self;
}

/// Used in pruning [`next_gen`]s
pub trait Prunable: Sized {
    /// This does any unfinished work in the despawning process.
    /// It doesn't need to be implemented unless in specific usecases where your algorithm needs to explicitly despawn a genome.
    fn despawn(self) {}
}

/// Used in speciated crossover nextgens. Allows for genomes to avoid crossover with ones that are too dissimilar.
#[cfg(feature = "speciation")]
#[cfg_attr(docsrs, doc(cfg(feature = "speciation")))]
pub trait Speciated: Sized {
    /// Calculates whether two genomes are similar enough to be considered part of the same species.
    fn is_same_species(&self, other: &Self) -> bool;

    /// Filters a list of genomes based on whether they are of the same species.
    fn filter_same_species<'a>(&'a self, genomes: &'a [Self]) -> Vec<&Self> {
        genomes.iter().filter(|g| self.is_same_species(g)).collect()
    }
}

/// Contains functions used in [`GeneticSim`][crate::GeneticSim].
pub mod next_gen {
    use super::*;

    #[cfg(feature = "rayon")]
    use rayon::prelude::*;

    use rand::prelude::*;

    /// When making a new generation, it mutates each genome a certain amount depending on their reward.
    /// This nextgen is very situational and should not be your first choice.
    pub fn scrambling_nextgen<G: RandomlyMutable>(mut rewards: Vec<(G, f32)>) -> Vec<G> {
        rewards.sort_by(|(_, r1), (_, r2)| r1.partial_cmp(r2).unwrap());

        let len = rewards.len() as f32;
        let mut rng = StdRng::from_rng(rand::thread_rng()).unwrap();

        rewards
            .into_iter()
            .enumerate()
            .map(|(i, (mut g, _))| {
                g.mutate(i as f32 / len, &mut rng);
                g
            })
            .collect()
    }

    /// When making a new generation, it despawns half of the genomes and then spawns children from the remaining to reproduce.
    #[cfg(not(feature = "rayon"))]
    pub fn division_pruning_nextgen<G: DivisionReproduction + Prunable + Clone>(
        rewards: Vec<(G, f32)>,
    ) -> Vec<G> {
        let population_size = rewards.len();
        let mut next_gen = pruning_helper(rewards);

        let mut rng = StdRng::from_rng(rand::thread_rng()).unwrap();

        let mut og_champions = next_gen
            .clone() // TODO remove if possible. currently doing so because `next_gen` is borrowed as mutable later
            .into_iter()
            .cycle();

        while next_gen.len() < population_size {
            let g = og_champions.next().unwrap();

            next_gen.push(g.divide(&mut rng));
        }

        next_gen
    }

    /// Rayon version of the [`division_pruning_nextgen`] function
    #[cfg(feature = "rayon")]
    pub fn division_pruning_nextgen<G: DivisionReproduction + Prunable + Clone + Send>(
        rewards: Vec<(G, f32)>,
    ) -> Vec<G> {
        let population_size = rewards.len();
        let mut next_gen = pruning_helper(rewards);

        let mut rng = StdRng::from_rng(rand::thread_rng()).unwrap();

        let mut og_champions = next_gen
            .clone() // TODO remove if possible. currently doing so because `next_gen` is borrowed as mutable later
            .into_iter()
            .cycle();

        while next_gen.len() < population_size {
            let g = og_champions.next().unwrap();

            next_gen.push(g.divide(&mut rng));
        }

        next_gen
    }

    /// Prunes half of the genomes and randomly crosses over the remaining ones.
    #[cfg(all(feature = "crossover", not(feature = "rayon")))]
    #[cfg_attr(docsrs, doc(cfg(feature = "crossover")))]
    pub fn crossover_pruning_nextgen<E: CrossoverReproduction + Prunable + Clone + PartialEq>(
        rewards: Vec<(E, f32)>,
    ) -> Vec<E> {
        let population_size = rewards.len();
        let mut next_gen = pruning_helper(rewards);

        let mut rng = rand::thread_rng();

        // TODO remove clone smh
        let og_champions = next_gen.clone();

        let mut og_champs_cycle = og_champions.iter().cycle();

        while next_gen.len() < population_size {
            let e1 = og_champs_cycle.next().unwrap();
            let e2 = &og_champions[rng.gen_range(0..og_champions.len() - 1)];

            if e1 == e2 {
                continue;
            }

            next_gen.push(e1.crossover(e2, &mut rng));
        }

        next_gen
    }

    /// Rayon version of the [`crossover_pruning_nextgen`] function.
    #[cfg(all(feature = "crossover", feature = "rayon",))]
    pub fn crossover_pruning_nextgen<
        G: CrossoverReproduction + Prunable + Clone + Send + PartialEq,
    >(
        rewards: Vec<(G, f32)>,
    ) -> Vec<G> {
        let population_size = rewards.len();
        let mut next_gen = pruning_helper(rewards);

        let mut rng = StdRng::from_rng(rand::thread_rng()).unwrap();

        // TODO remove clone smh
        let og_champions = next_gen.clone();

        let mut og_champs_cycle = og_champions.iter().cycle();

        while next_gen.len() < population_size {
            let g1 = og_champs_cycle.next().unwrap();
            let g2 = &og_champions[rng.gen_range(0..og_champions.len() - 1)];

            if g1 == g2 {
                continue;
            }

            next_gen.push(g1.crossover(g2, &mut rng));
        }

        next_gen
    }

    /// Similar to [`crossover_pruning_nextgen`], this nextgen will prune and then perform crossover reproduction.
    /// With this function, crossover reproduction will only occur if both genomes are of the same species, otherwise one will perform divison to reproduce.
    #[cfg(all(feature = "speciation", not(feature = "rayon")))]
    #[cfg_attr(docsrs, doc(cfg(feature = "specation")))]
    pub fn speciated_crossover_pruning_nextgen<
        'a,
        G: CrossoverReproduction + DivisionReproduction + Speciated + Prunable + Clone + PartialEq,
    >(
        rewards: Vec<(G, f32)>,
    ) -> Vec<G> {
        let population_size = rewards.len();
        let mut next_gen = pruning_helper(rewards);

        let mut rng = StdRng::from_rng(rand::thread_rng()).unwrap();

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

        let mut rng = StdRng::from_rng(rand::thread_rng()).unwrap();

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
        let other = same_species[rng.gen_range(0..same_species.len())];

        genome.crossover(other, rng)
    }

    /// helps with builtin pruning nextgens
    #[cfg(not(feature = "rayon"))]
    fn pruning_helper<E: Prunable + Clone>(mut rewards: Vec<(E, f32)>) -> Vec<E> {
        rewards.sort_by(|(_, r1), (_, r2)| r1.partial_cmp(r2).unwrap());

        let median = rewards[rewards.len() / 2].1;

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
    fn pruning_helper<E: Prunable + Send>(mut rewards: Vec<(E, f32)>) -> Vec<E> {
        rewards.sort_by(|(_, r1), (_, r2)| r1.partial_cmp(r2).unwrap());

        let median = rewards[rewards.len() / 2].1;

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
    use crate::prelude::*;

    #[derive(Default, Clone, Debug, PartialEq)]
    struct MyGenome(f32);

    impl RandomlyMutable for MyGenome {
        fn mutate(&mut self, rate: f32, rng: &mut impl rand::Rng) {
            self.0 += rng.gen::<f32>() * rate;
        }
    }

    impl DivisionReproduction for MyGenome {
        fn divide(&self, rng: &mut impl rand::Rng) -> Self {
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
            Self(rng.gen())
        }
    }

    #[cfg(feature = "crossover")]
    #[derive(Debug, Clone, PartialEq)]
    struct MyCrossoverGenome(MyGenome);

    #[cfg(feature = "crossover")]
    impl RandomlyMutable for MyCrossoverGenome {
        fn mutate(&mut self, rate: f32, rng: &mut impl rand::Rng) {
            self.0.mutate(rate, rng);
        }
    }

    #[cfg(feature = "crossover")]
    impl CrossoverReproduction for MyCrossoverGenome {
        fn crossover(&self, other: &Self, rng: &mut impl rand::Rng) -> Self {
            let mut child = Self(MyGenome((self.0 .0 + other.0 .0) / 2.));
            child.mutate(0.25, rng);
            child
        }
    }

    #[cfg(feature = "crossover")]
    impl Prunable for MyCrossoverGenome {}

    #[cfg(feature = "crossover")]
    impl GenerateRandom for MyCrossoverGenome {
        fn gen_random(rng: &mut impl rand::Rng) -> Self {
            Self(MyGenome::gen_random(rng))
        }
    }

    #[cfg(feature = "speciation")]
    impl DivisionReproduction for MyCrossoverGenome {
        fn divide(&self, rng: &mut impl rand::Rng) -> Self {
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
        let mut rng = rand::thread_rng();
        let mut sim = GeneticSim::new(
            Vec::gen_random(&mut rng, 1000),
            my_fitness_fn,
            scrambling_nextgen,
        );

        sim.perform_generations(100);

        dbg!(sim.genomes);
    }

    #[cfg(feature = "rayon")]
    fn r_scramble() {
        let mut sim = GeneticSim::new(Vec::gen_random(1000), my_fitness_fn, scrambling_nextgen);

        sim.perform_generations(100);

        dbg!(sim.genomes);
    }

    #[cfg(not(feature = "rayon"))]
    #[test]
    fn d_prune() {
        let mut rng = rand::thread_rng();
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
        let mut rng = rand::thread_rng();

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
        let mut rng = rand::thread_rng();

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

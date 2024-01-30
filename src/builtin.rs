/// Used in all of the builtin [next_gen]s to randomly mutate entities a given amount
pub trait RandomlyMutable {
    /// Mutate the entity with a given mutation rate (0..1)
    fn mutate(&mut self, rate: f32, rng: &mut impl rand::Rng);
}

/// Used in dividually-reproducing [next_gen]s
pub trait DivisionReproduction: RandomlyMutable {
    /// Create a new child with mutation. Similar to [RandomlyMutable::mutate], but returns a new instance instead of modifying the original.
    /// If it is simply returning a cloned and mutated version, consider using a constant mutation rate.
    fn spawn_child(&self, rng: &mut impl rand::Rng) -> Self;
}

/// Used in crossover-reproducing [next_gen]s
#[cfg(feature = "crossover")]
pub trait CrossoverReproduction: RandomlyMutable {
    /// Use crossover reproduction to create a new entity.
    fn spawn_child(&self, other: &Self, rng: &mut impl rand::Rng) -> Self;
}

/// Used in pruning [next_gen]s
pub trait Prunable: Sized {
    /// This does any unfinished work in the despawning process.
    /// It doesn't need to be implemented unless in specific usecases where your algorithm needs to explicitly despawn an entity.
    fn despawn(self) {}
}

/// Contains functions used in [GeneticSim][crate::GeneticSim].
pub mod next_gen {
    use super::*;
    use rand::prelude::*;

    use rand::{rngs::StdRng, SeedableRng};
    #[cfg(feature = "rayon")]
    use rayon::prelude::*;

    /// When making a new generation, it mutates each entity a certain amount depending on their reward.
    /// This nextgen is very situational and should not be your first choice.
    pub fn scrambling_nextgen<E: RandomlyMutable>(mut rewards: Vec<(E, f32)>) -> Vec<E> {
        rewards.sort_by(|(_, r1), (_, r2)| r1.partial_cmp(r2).unwrap());

        let len = rewards.len() as f32;
        let mut rng = StdRng::from_rng(rand::thread_rng()).unwrap();

        rewards
            .into_iter()
            .enumerate()
            .map(|(i, (mut e, _))| {
                e.mutate(i as f32 / len, &mut rng);
                e
            })
            .collect()
    }

    /// When making a new generation, it despawns half of the entities and then spawns children from the remaining to reproduce.
    /// WIP: const generic for mutation rate, will allow for [DivisionReproduction::spawn_child] to accept a custom mutation rate. Delayed due to current Rust limitations
    #[cfg(not(feature = "rayon"))]
    pub fn division_pruning_nextgen<E: DivisionReproduction + Prunable + Clone>(
        rewards: Vec<(E, f32)>,
    ) -> Vec<E> {
        let population_size = rewards.len();
        let mut next_gen = pruning_helper(rewards);

        let mut rng = StdRng::from_rng(rand::thread_rng()).unwrap();

        let mut og_champions = next_gen
            .clone() // TODO remove if possible. currently doing so because `next_gen` is borrowed as mutable later
            .into_iter()
            .cycle();

        while next_gen.len() < population_size {
            let e = og_champions.next().unwrap();

            next_gen.push(e.spawn_child(&mut rng));
        }

        next_gen
    }

    /// Rayon version of the [division_pruning_nextgen] function
    #[cfg(feature = "rayon")]
    pub fn division_pruning_nextgen<E: DivisionReproduction + Prunable + Clone + Send>(
        rewards: Vec<(E, f32)>,
    ) -> Vec<E> {
        let population_size = rewards.len();
        let mut next_gen = pruning_helper(rewards);

        let mut rng = StdRng::from_rng(rand::thread_rng()).unwrap();

        let mut og_champions = next_gen
            .clone() // TODO remove if possible. currently doing so because `next_gen` is borrowed as mutable later
            .into_iter()
            .cycle();

        while next_gen.len() < population_size {
            let e = og_champions.next().unwrap();

            next_gen.push(e.spawn_child(&mut rng));
        }

        next_gen
    }

    /// Prunes half of the entities and randomly breeds the remaining ones.
    #[cfg(all(feature = "crossover", not(feature = "rayon")))]
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

            next_gen.push(e1.spawn_child(e2, &mut rng));
        }

        next_gen
    }

    /// Rayon version of the [crossover_pruning_nextgen] function.
    #[cfg(all(feature = "crossover", feature = "rayon",))]
    pub fn crossover_pruning_nextgen<
        E: CrossoverReproduction + Prunable + Clone + Send + PartialEq,
    >(
        rewards: Vec<(E, f32)>,
    ) -> Vec<E> {
        let population_size = rewards.len();
        let mut next_gen = pruning_helper(rewards);

        let mut rng = StdRng::from_rng(rand::thread_rng()).unwrap();

        // TODO remove clone smh
        let og_champions = next_gen.clone();

        let mut og_champs_cycle = og_champions.iter().cycle();

        while next_gen.len() < population_size {
            let e1 = og_champs_cycle.next().unwrap();
            let e2 = &og_champions[rng.gen_range(0..og_champions.len() - 1)];

            if e1 == e2 {
                continue;
            }

            next_gen.push(e1.spawn_child(e2, &mut rng));
        }

        next_gen
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

    /// Rayon version of [pruning_helper].
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
    struct MyEntity(f32);

    impl RandomlyMutable for MyEntity {
        fn mutate(&mut self, rate: f32, rng: &mut impl rand::Rng) {
            self.0 += rng.gen::<f32>() * rate;
        }
    }

    impl DivisionReproduction for MyEntity {
        fn spawn_child(&self, rng: &mut impl rand::Rng) -> Self {
            let mut child = self.clone();
            child.mutate(0.25, rng);
            child
        }
    }

    impl Prunable for MyEntity {
        fn despawn(self) {
            println!("RIP {:?}", self);
        }
    }

    impl GenerateRandom for MyEntity {
        fn gen_random(rng: &mut impl Rng) -> Self {
            Self(rng.gen())
        }
    }

    #[cfg(feature = "crossover")]
    #[derive(Debug, Clone, PartialEq)]
    struct MyCrossoverEntity(MyEntity);

    #[cfg(feature = "crossover")]
    impl RandomlyMutable for MyCrossoverEntity {
        fn mutate(&mut self, rate: f32, rng: &mut impl rand::Rng) {
            self.0.mutate(rate, rng);
        }
    }

    #[cfg(feature = "crossover")]
    impl CrossoverReproduction for MyCrossoverEntity {
        fn spawn_child(&self, other: &Self, rng: &mut impl rand::Rng) -> Self {
            let mut child = Self(MyEntity((self.0 .0 + other.0 .0) / 2.));
            child.mutate(0.25, rng);
            child
        }
    }

    #[cfg(feature = "crossover")]
    impl Prunable for MyCrossoverEntity {}

    #[cfg(feature = "crossover")]
    impl GenerateRandom for MyCrossoverEntity {
        fn gen_random(rng: &mut impl rand::Rng) -> Self {
            Self(MyEntity::gen_random(rng))
        }
    }

    const MAGIC_NUMBER: f32 = std::f32::consts::E;

    fn my_fitness_fn(ent: &MyEntity) -> f32 {
        (MAGIC_NUMBER - ent.0).abs() * -1.
    }

    #[cfg(feature = "crossover")]
    fn my_crossover_fitness_fn(ent: &MyCrossoverEntity) -> f32 {
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

        for _ in 0..100 {
            sim.next_generation();
        }

        dbg!(sim.entities);
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

        for _ in 0..100 {
            sim.next_generation();
        }

        dbg!(sim.entities);
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

        for _ in 0..100 {
            sim.next_generation();
        }

        dbg!(sim.entities);
    }
}

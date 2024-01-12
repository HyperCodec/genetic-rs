/// Used in all of the builtin [next_gen]s to randomly mutate entities a given amount
pub trait RandomlyMutable {
    /// Mutate the entity with a given mutation rate (0..1)
    fn mutate(&mut self, rate: f32);
}

/// Used in asexually reproducing [next_gen]s
pub trait ASexualEntity: RandomlyMutable {
    /// Create a new child with mutation. Similar to [RandomlyMutable::mutate], but returns a new instance instead of modifying the original.
    /// If it is simply returning a cloned and mutated version, consider using a constant mutation rate.
    fn spawn_child(&self) -> Self;
}

/// Used in sexually reproducing [next_gen]s
#[cfg(feature = "sexual")]
pub trait SexualEntity: RandomlyMutable {
    /// Use sexual reproduction to create a new entity.
    fn spawn_child(&self, other: &Self) -> Self;
}

/// Used in pruning [next_gen]s
pub trait Prunable {
    /// This does any unfinished work in the despawning process.
    /// It doesn't need to be implemented unless in specific usecases where your algorithm needs to explicitly despawn an entity.
    fn despawn(&mut self) {}
}

/// Contains functions used in [GeneticSim][crate::GeneticSim].
pub mod next_gen {
    use super::*;

    #[cfg(feature = "sexual")] use rand::prelude::*;

    /// When making a new generation, it mutates each entity a certain amount depending on their reward.
    /// This nextgen is very situational and should not be your first choice of them.
    pub fn asexual_scrambling_nextgen<E: ASexualEntity>(mut rewards: Vec<(E, f32)>) -> Vec<E> {
        rewards.sort_by(|(_, r1), (_, r2)| r1.partial_cmp(r2).unwrap());

        let len = rewards.len() as f32;

        rewards
            .into_iter()
            .enumerate()
            .map(|(i, (mut e, _))| {
                e.mutate(i as f32 / len);
                e
            })
            .collect()
    }

    /// When making a new generation, it despawns half of the entities and then spawns children from the remaining to reproduce.
    /// WIP: const generic for mutation rate, will allow for [ASexualEntity::spawn_child] to accept a custom mutation rate. Delayed due to current Rust limitations
    pub fn asexual_pruning_nextgen<E: ASexualEntity + Prunable + Clone>(rewards: Vec<(E, f32)>) -> Vec<E> {
        let population_size = rewards.len();
        let mut next_gen = pruning_helper(rewards);

        let mut og_champions = next_gen
            .clone() // TODO remove if possible. currently doing so because `next_gen` is borrowed as mutable later
            .into_iter()
            .cycle();
        
        while next_gen.len() < population_size {
            let e = og_champions.next().unwrap();

            next_gen.push(e.spawn_child());
        }

        next_gen
    }

    /// Prunes half of the entities and randomly breeds the remaining ones.
    /// S: allow selfbreeding - false by default.
    #[cfg(feature = "sexual")]
    pub fn sexual_pruning_nextgen<E: SexualEntity + Prunable + Clone, const S: bool = false>(rewards: Vec<(E, f32)>) -> Vec<E> {
        let population_size = rewards.len();
        let mut next_gen = pruning_helper(rewards);

        // TODO better/more customizable rng
        let mut rng = rand::thread_rng();

        // TODO remove clone smh
        let og_champions = next_gen.clone();

        let mut og_champs_cycle = og_champions
            .iter()
            .cycle();

        while next_gen.len() < population_size {
            let e1 = og_champs_cycle.next().unwrap();
            let e2 = og_champions[rand::gen::<usize>(0..og_champions.len()-1)];

            if !S && e1 == e2 {
                continue;
            }

            next_gen.push(e1.spawn_child(&e2));
        }

        next_gen
    }

    fn pruning_helper<E: Prunable + Clone>(mut rewards: Vec<(E, f32)>) -> Vec<E> {
        rewards.sort_by(|(_, r1), (_, r2)| r1.partial_cmp(r2).unwrap());

        let median = rewards[rewards.len() / 2].1;

        rewards
            .into_iter()
            .filter_map(|(mut e, r)| {
                if r < median {
                    e.despawn();
                    return None;
                }

                Some(e)
            })
            .collect()
    }
}
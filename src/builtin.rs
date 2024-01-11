/// Used in asexually reproducing [next_gen]s
pub trait ASexualEntity {
    /// Mutate the entity with a given mutation rate (0..1)
    fn mutate(&mut self, rate: f32);

    /// Create a new child with mutation. Similar to [mutate][ASexualEntity::mutate], but returns a new instance instead of modifying the original.
    /// If it is simply returning a cloned and mutated version, consider using a constant default mutation rate.
    fn spawn_child(&self) -> Self;
}

/// Used sexually reproducing [next_gen]s
pub trait SexualEntity {
    /// 
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

    /// When making a new generation, it mutates each entity a certain amount depending on their reward.
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
        let sum: f32 = rewards
            .iter()
            .map(|(_, r)| r)
            .sum();

        let population_size = rewards.len();
        let avg = sum / population_size as f32;

        let mut next_gen: Vec<E> = rewards
            .into_iter()
            .filter_map(|(mut e, r)| {
                if r < avg {
                    e.despawn();
                    return None;
                }

                Some(e)
            })
            .collect();

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
}
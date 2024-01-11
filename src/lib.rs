#![warn(missing_docs)]

//! A small crate to quickstart genetic algorithm projects

use replace_with::replace_with_or_abort;

/// Built-in nextgen functions and traits to go with them.
/// TODO example
#[cfg(feature = "builtin")]
pub mod builtin;

/// Used to quickly import everything this crate has to offer.
/// Simply add `use genetic_rs::prelude::*` to begin using this crate.
pub mod prelude;

/// The simulation controller.
/// TODO example
pub struct GeneticSim<E>
where
    E: Sized,
{
    /// The current population of entities
    pub entities: Vec<E>,
    reward: Box<dyn Fn(&E) -> f32>,
    next_gen: Box<dyn Fn(Vec<(E, f32)>) -> Vec<E>>,
}

impl<E> GeneticSim<E>
where
    E: Sized,
{
    /// Creates a GeneticSim with a given population of `starting_entities` (the size of which will be retained),
    /// a given reward function, and a given nextgen function.
    pub fn new(
        starting_entities: Vec<E>,
        reward: impl Fn(&E) -> f32 + 'static, 
        next_gen: impl Fn(Vec<(E, f32) >) -> Vec<E> + 'static
    ) -> Self {
        Self {
            entities: starting_entities,
            reward: Box::new(reward),
            next_gen: Box::new(next_gen),
        }
    }

    /// Uses the `next_gen` provided in [GeneticSim::new] to create the next generation of entities.
    pub fn next_generation(&mut self) {
        // TODO maybe remove unneccessary dependency, can prob use std::mem::replace
        replace_with_or_abort(&mut self.entities, |entities| {
            let rewards = entities
                .into_iter()
                .map(|e| {
                    let reward = (self.reward)(&e);
                    (e, reward)
                })
                .collect();

            (self.next_gen)(rewards)
        });
    }
}

#[cfg(test)]
mod tests {

    use super::prelude::*;
    use rand::prelude::*;

    #[derive(Default, Clone, Debug)]
    struct MyEntity(f32);

    impl ASexualEntity for MyEntity {
        fn mutate(&mut self, rate: f32) {
            let mut rng = rand::thread_rng();
            self.0 += rng.gen::<f32>() * rate;
        }

        fn spawn_child(&self) -> Self {
            let mut child = self.clone();
            child.mutate(0.25);
            child
        }
    }

    impl Prunable for MyEntity {
        fn despawn(&mut self) {
            println!("RIP {:?}", self);
        }
    }

    const MAGIC_NUMBER: f32 = std::f32::consts::E;

    fn my_reward_fn(ent: &MyEntity) -> f32 {
        (MAGIC_NUMBER - ent.0).abs() * -1.
    }

    #[test]
    fn as_scramble() {
        let mut sim = GeneticSim::new(
            vec![MyEntity::default(); 100], 
            my_reward_fn, 
            asexual_scrambling_nextgen,
        );

        for _ in 0..100 {
            sim.next_generation();
        }

        dbg!(sim.entities);
    }

    #[test]
    fn as_prune() {
        let mut sim = GeneticSim::new(
            vec![MyEntity::default(); 100],
            my_reward_fn,
            asexual_pruning_nextgen,
        );

        for _ in 0..100 {
            sim.next_generation();
        }

        dbg!(sim.entities);
    }
}
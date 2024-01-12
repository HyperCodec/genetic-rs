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
/// ```rust
/// use genetic_rs::prelude::*;
/// 
/// #[derive(Debug, Clone)]
/// struct MyEntity {
///     a: f32,
///     b: f32,
/// }
/// 
/// impl RandomlyMutable for MyEntity {
///     fn mutate(&mut self, rate: f32) {
///         self.a += fastrand::f32() * rate;
///         self.b += fastrand::f32() * rate;
///     }
/// }
/// 
/// impl ASexualEntity for MyEntity {
///     fn spawn_child(&self) -> Self {
///         let mut child = self.clone();
///         child.mutate(0.25); // you'll generally want to use a constant mutation rate for mutating children.
///         child
///     }
/// }
/// 
/// impl Prunable for MyEntity {} // if we wanted to, we could implement the `despawn` function to run any cleanup code as needed. in this example, though, we do not need it.
/// 
/// fn main() {
///     let population: Vec<_> = (0..100)
///         .into_iter()
///         .map(|_| MyEntity { a: fastrand::f32(), b: fastrand::f32() })
///         .collect();
///     
///     let my_reward_fn = |e: &MyEntity| {
///         e.a * e.b // should result in entities increasing their value
///     };
/// 
///     let mut sim = GeneticSim::new(
///         population,
///         my_reward_fn,
///         asexual_pruning_nextgen,
///     );
/// 
///     for _ in 0..1000 {
///         // if this were a more complex simulation, you might test entities in `sim.entities` between `next_generation` calls to provide a more accurate reward.
///         sim.next_generation();
///     }
/// 
///     dbg!(sim.entities);
/// }
/// ```
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

    #[derive(Default, Clone, Debug)]
    struct MyEntity(f32);

    impl RandomlyMutable for MyEntity {
        fn mutate(&mut self, rate: f32) {
            self.0 += fastrand::f32() * rate;
        }
    }

    impl ASexualEntity for MyEntity {
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
    fn scramble() {
        let pop = (0..1000)
            .map(|_| MyEntity(fastrand::f32()))
            .collect();

        let mut sim = GeneticSim::new(
            pop, 
            my_reward_fn, 
            scrambling_nextgen,
        );

        for _ in 0..100 {
            sim.next_generation();
        }

        dbg!(sim.entities);
    }

    #[test]
    fn a_prune() {
        let pop = (0..1000)
            .map(|_| MyEntity(fastrand::f32()))
            .collect();

        let mut sim = GeneticSim::new(
            pop,
            my_reward_fn,
            asexual_pruning_nextgen,
        );

        for _ in 0..100 {
            sim.next_generation();
        }

        dbg!(sim.entities);
    }
}
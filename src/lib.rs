#![warn(missing_docs)]

//! A small crate to quickstart genetic algorithm projects

use replace_with::replace_with_or_abort;

/// Built-in nextgen functions and traits to go with them.
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
/// impl GenerateRandom for MyEntity {
///     fn gen_random(rng: &mut impl Rng) -> Self {
///         Self {
///             a: rng.gen(),
///             b: rng.gen(),
///         }
///     }
/// }
/// 
/// fn main() {
///     let my_fitness_fn = |e: &MyEntity| {
///         e.a * e.b // should result in entities increasing their value
///     };
/// 
///     let mut rng = rand::thread_rng();
/// 
///     let mut sim = GeneticSim::new(
///         Vec::gen_random(&mut rng, 1000),
///         my_fitness_fn,
///         asexual_pruning_nextgen,
///     );
/// 
///     for _ in 0..100 {
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
    fitness: Box<dyn Fn(&E) -> f32>,
    next_gen: Box<dyn Fn(Vec<(E, f32)>) -> Vec<E>>,
}

impl<E> GeneticSim<E>
where
    E: Sized,
{
    /// Creates a GeneticSim with a given population of `starting_entities` (the size of which will be retained),
    /// a given fitness function, and a given nextgen function.
    pub fn new(
        starting_entities: Vec<E>,
        fitness: impl Fn(&E) -> f32 + 'static, 
        next_gen: impl Fn(Vec<(E, f32) >) -> Vec<E> + 'static
    ) -> Self {
        Self {
            entities: starting_entities,
            fitness: Box::new(fitness),
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
                    let fitness: f32 = (self.fitness)(&e);
                    (e, fitness)
                })
                .collect();

            (self.next_gen)(rewards)
        });
    }
}

#[cfg(feature = "genrand")] use rand::prelude::*;

/// Helper trait used in the generation of random starting populations
#[cfg(feature = "genrand")]
pub trait GenerateRandom {
    /// Create a completely random instance of the entity
    fn gen_random(rng: &mut impl Rng) -> Self;
}

/// Blanket trait used on collections that contain objects implementing GenerateRandom
#[cfg(feature = "genrand")]
pub trait GenerateRandomCollection<T>
where
    T: GenerateRandom,
{
    /// Generate a random collection of the inner objects with a given amount
    fn gen_random(rng: &mut impl Rng, amount: usize) -> Self;
}

impl<C, T> GenerateRandomCollection<T> for C
where
    C: FromIterator<T>,
    T: GenerateRandom,
{
    fn gen_random(rng: &mut impl Rng, amount: usize) -> Self {
        (0..amount)
            .into_iter()
            .map(|_| T::gen_random(rng))
            .collect()
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
        fn despawn(self) {
            println!("RIP {:?}", self);
        }
    }

    impl GenerateRandom for MyEntity {
        fn gen_random(rng: &mut impl Rng) -> Self {
            Self(rng.gen())
        }
    }

    const MAGIC_NUMBER: f32 = std::f32::consts::E;

    fn my_fitness_fn(ent: &MyEntity) -> f32 {
        (MAGIC_NUMBER - ent.0).abs() * -1.
    }

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

    #[test]
    fn a_prune() {
        let mut rng = rand::thread_rng();
        let mut sim = GeneticSim::new(
            Vec::gen_random(&mut rng, 1000),
            my_fitness_fn,
            asexual_pruning_nextgen,
        );

        for _ in 0..100 {
            sim.next_generation();
        }

        dbg!(sim.entities);
    }
}
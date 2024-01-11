use replace_with::replace_with_or_abort;

#[cfg(feature = "builtin")]
pub mod builtin;

pub struct GeneticSim<E>
where
    E: Sized,
{
    entities: Vec<E>, // should work without actually owning the entities
    reward: Box<dyn Fn(&E) -> f32>,
    next_gen: Box<dyn Fn(Vec<(E, f32)>) -> Vec<E>>,
}

impl<E> GeneticSim<E>
where
    E: Sized,
{
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

    /// Uses the `next_gen` provided in [GeneticSim::new] to create the next generation of 
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
    use super::*;

    #[test]
    fn test_api() {
        
    }
}
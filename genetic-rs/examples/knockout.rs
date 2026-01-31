use std::cmp::Ordering;

use genetic_rs::prelude::*;

#[derive(Clone, Debug)]
struct Genome(f32);

impl GenerateRandom for Genome {
    fn gen_random(rng: &mut impl rand::Rng) -> Self {
        Self(rng.random())
    }
}

impl RandomlyMutable for Genome {
    type Context = ();

    fn mutate(&mut self, _: &(), rate: f32, rng: &mut impl Rng) {
        self.0 += rng.random::<f32>() * rate;
    }
}

impl Mitosis for Genome {
    type Context = ();

    fn divide(&self, _: &(), rate: f32, rng: &mut impl Rng) -> Self {
        let mut child = self.clone();
        child.mutate(&(), rate, rng);
        child
    }
}

impl Crossover for Genome {
    type Context = ();

    fn crossover(&self, other: &Self, _: &(), rate: f32, rng: &mut impl Rng) -> Self {
        let mut child = Self((self.0 + other.0) / 2.);
        child.mutate(&(), rate, rng);
        child
    }
}

fn knockout(a: &Genome, b: &Genome) -> KnockoutWinner {
    match a.0.total_cmp(&b.0) {
        Ordering::Equal | Ordering::Greater => KnockoutWinner::First,
        Ordering::Less => KnockoutWinner::Second,
    }
}

fn main() {
    #[cfg(not(feature = "rayon"))]
    let mut rng = rand::rng();

    let mut sim = GeneticSim::new(
        #[cfg(not(feature = "rayon"))]
        Vec::gen_random(&mut rng, 100),
        #[cfg(feature = "rayon")]
        Vec::gen_random(100),
        // we are using crossover, so we can always expect it
        // to maintain 100 genomes, which is an even number.
        KnockoutEliminator::new(knockout, ActionIfOdd::Panic),
        CrossoverRepopulator::new(0.25, ()),
    );

    sim.perform_generations(100);

    dbg!(&sim.genomes);
}

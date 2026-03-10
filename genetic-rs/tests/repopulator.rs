//! Integration tests for [`MitosisRepopulator`], [`CrossoverRepopulator`],
//! and the [`FromParent`] helper.

use genetic_rs::prelude::*;

// ─────────────────────────────────────────────────────────────────────────────
// Shared test genome
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq)]
struct Genome(f32);

impl GenerateRandom for Genome {
    fn gen_random(rng: &mut impl rand::Rng) -> Self {
        Self(rng.random())
    }
}

impl RandomlyMutable for Genome {
    type Context = ();

    fn mutate(&mut self, _: &(), rate: f32, rng: &mut impl rand::Rng) {
        self.0 += rng.random::<f32>() * rate;
    }
}

impl Mitosis for Genome {
    type Context = ();

    fn divide(&self, ctx: &(), rate: f32, rng: &mut impl rand::Rng) -> Self {
        let mut child = self.clone();
        child.mutate(ctx, rate, rng);
        child
    }
}

/// Deterministic crossover: the child is exactly the average of its parents.
/// A zero mutation rate means no random component is added.
impl Crossover for Genome {
    type Context = ();

    fn crossover(&self, other: &Self, _: &(), _rate: f32, _rng: &mut impl rand::Rng) -> Self {
        Self((self.0 + other.0) / 2.0)
    }
}

fn fitness(g: &Genome) -> f32 {
    g.0
}

// ─────────────────────────────────────────────────────────────────────────────
// MitosisRepopulator
// ─────────────────────────────────────────────────────────────────────────────

/// The repopulator must grow the population back to the target size.
#[test]
fn mitosis_repopulator_fills_to_target() {
    let mut rng = rand::rng();
    let mut genomes: Vec<Genome> = Vec::gen_random(&mut rng, 5);
    MitosisRepopulator::new(0.1, ()).repopulate(&mut genomes, 20);
    assert_eq!(genomes.len(), 20);
}

/// If the population is already at the target, it must not grow further.
#[test]
fn mitosis_repopulator_no_op_when_at_target() {
    let mut rng = rand::rng();
    let mut genomes: Vec<Genome> = Vec::gen_random(&mut rng, 10);
    MitosisRepopulator::new(0.1, ()).repopulate(&mut genomes, 10);
    assert_eq!(genomes.len(), 10);
}

/// At mutation rate 0.0 each child must be an exact clone of its parent.
#[test]
fn mitosis_zero_mutation_rate_produces_clone() {
    let parent = Genome(42.0);
    let mut rng = rand::rng();
    let child = parent.divide(&(), 0.0, &mut rng);
    assert_eq!(
        child, parent,
        "child at rate 0.0 must be identical to parent"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// CrossoverRepopulator
// ─────────────────────────────────────────────────────────────────────────────

/// The crossover repopulator must grow the population back to the target size.
#[test]
fn crossover_repopulator_fills_to_target() {
    let mut rng = rand::rng();
    let mut genomes: Vec<Genome> = Vec::gen_random(&mut rng, 5);
    CrossoverRepopulator::new(0.1, ()).repopulate(&mut genomes, 20);
    assert_eq!(genomes.len(), 20);
}

/// If the population is already at the target, it must not grow further.
#[test]
fn crossover_repopulator_no_op_when_at_target() {
    let mut rng = rand::rng();
    let mut genomes: Vec<Genome> = Vec::gen_random(&mut rng, 10);
    CrossoverRepopulator::new(0.1, ()).repopulate(&mut genomes, 10);
    assert_eq!(genomes.len(), 10);
}

/// Crossover child must equal the average of its parents when mutation rate is 0.
#[test]
fn crossover_child_is_average_of_parents() {
    let p1 = Genome(2.0);
    let p2 = Genome(4.0);
    let mut rng = rand::rng();
    let child = p1.crossover(&p2, &(), 0.0, &mut rng);
    assert!(
        (child.0 - 3.0).abs() < 1e-6,
        "expected child value 3.0, got {}",
        child.0
    );
}

/// Population size stays constant when `GeneticSim` uses crossover over many generations.
#[test]
fn population_size_constant_with_crossover_sim() {
    let mut rng = rand::rng();
    let initial_size = 30;
    let mut sim = GeneticSim::new(
        Vec::<Genome>::gen_random(&mut rng, initial_size),
        FitnessEliminator::new_without_observer(fitness),
        CrossoverRepopulator::new(0.05, ()),
    );
    sim.perform_generations(50);
    assert_eq!(sim.genomes.len(), initial_size);
}

// ─────────────────────────────────────────────────────────────────────────────
// FromParent
// ─────────────────────────────────────────────────────────────────────────────

/// [`Vec::from_parent`] must produce exactly the requested number of genomes.
#[test]
fn from_parent_creates_correct_count() {
    let parent = Genome(1.0);
    let population = Vec::<Genome>::from_parent(parent, 15, (), 0.0);
    assert_eq!(population.len(), 15);
}

/// The original parent must be the first element of the returned population.
#[test]
fn from_parent_includes_original_parent() {
    let parent = Genome(7.0);
    let population = Vec::<Genome>::from_parent(parent.clone(), 5, (), 0.0);
    assert_eq!(
        population[0], parent,
        "the first element must be the original parent"
    );
}

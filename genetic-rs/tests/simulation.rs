//! Integration tests for [`GeneticSim`] core invariants, [`FitnessEliminator`] behaviour,
//! and helper types such as [`FitnessObserver`].

use genetic_rs::prelude::*;

// ─────────────────────────────────────────────────────────────────────────────
// Shared test genome
// ─────────────────────────────────────────────────────────────────────────────

/// A simple genome whose fitness is just its value.
#[derive(Clone, Debug)]
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

fn fitness(g: &Genome) -> f32 {
    g.0
}

// ─────────────────────────────────────────────────────────────────────────────
// GeneticSim population-size invariants
// ─────────────────────────────────────────────────────────────────────────────

/// The population size must be identical before and after a single generation.
#[test]
fn population_size_preserved_after_next_generation() {
    let mut rng = rand::rng();
    let initial_size = 20;
    let mut sim = GeneticSim::new(
        Vec::<Genome>::gen_random(&mut rng, initial_size),
        FitnessEliminator::new_without_observer(fitness),
        MitosisRepopulator::new(0.1, ()),
    );
    sim.next_generation();
    assert_eq!(sim.genomes.len(), initial_size);
}

/// The population size must remain constant over many generations.
#[test]
fn population_size_preserved_over_many_generations() {
    let mut rng = rand::rng();
    let initial_size = 50;
    let mut sim = GeneticSim::new(
        Vec::<Genome>::gen_random(&mut rng, initial_size),
        FitnessEliminator::new_without_observer(fitness),
        MitosisRepopulator::new(0.1, ()),
    );
    sim.perform_generations(100);
    assert_eq!(sim.genomes.len(), initial_size);
}

// ─────────────────────────────────────────────────────────────────────────────
// GenerateRandom / GenerateRandomCollection
// ─────────────────────────────────────────────────────────────────────────────

/// [`Vec::gen_random`] must produce exactly the requested number of genomes.
#[test]
fn gen_random_collection_produces_correct_size() {
    let mut rng = rand::rng();
    for size in [1usize, 10, 100] {
        let population: Vec<Genome> = Vec::gen_random(&mut rng, size);
        assert_eq!(
            population.len(),
            size,
            "expected {size} genomes, got {}",
            population.len()
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FitnessEliminator elimination behaviour
// ─────────────────────────────────────────────────────────────────────────────

/// With the default threshold (0.5) and 10 genomes, exactly 6 must survive and
/// all survivors must be in the top half by fitness.
#[test]
fn fitness_eliminator_keeps_top_by_fitness() {
    // Genomes with distinct, known fitness values 0.0 … 9.0.
    let genomes: Vec<Genome> = (0..10).map(|i| Genome(i as f32)).collect();
    let mut eliminator = FitnessEliminator::new_without_observer(fitness);
    let survivors = eliminator.eliminate(genomes);

    // threshold=0.5, n=10 → floor(10*0.5) + 1 = 6 survivors.
    assert_eq!(
        survivors.len(),
        6,
        "expected 6 survivors with default threshold"
    );

    // Every survivor must be in the top half (fitness ≥ 4.0).
    for g in &survivors {
        assert!(
            g.0 >= 4.0,
            "genome with fitness {} survived but is below the top-half threshold",
            g.0,
        );
    }
}

/// The genome with the highest fitness must always survive elimination.
#[test]
fn fitness_eliminator_highest_fitness_always_survives() {
    let genomes: Vec<Genome> = (0..20).map(|i| Genome(i as f32)).collect();
    let mut eliminator = FitnessEliminator::new_without_observer(fitness);
    let survivors = eliminator.eliminate(genomes);
    assert!(
        survivors.iter().any(|g| g.0 == 19.0),
        "the highest-fitness genome (19.0) must always survive"
    );
}

/// The genome with the lowest fitness must always be eliminated.
#[test]
fn fitness_eliminator_lowest_fitness_never_survives() {
    let genomes: Vec<Genome> = (0..20).map(|i| Genome(i as f32)).collect();
    let mut eliminator = FitnessEliminator::new_without_observer(fitness);
    let survivors = eliminator.eliminate(genomes);
    assert!(
        !survivors.iter().any(|g| g.0 == 0.0),
        "the lowest-fitness genome (0.0) must always be eliminated"
    );
}

/// [`FitnessEliminator::calculate_and_sort`] must return genomes sorted in
/// descending order of fitness.
#[test]
fn fitness_eliminator_sorts_descending() {
    let genomes = vec![Genome(3.0), Genome(1.0), Genome(4.0), Genome(2.0)];
    let eliminator = FitnessEliminator::new_without_observer(fitness);
    let sorted = eliminator.calculate_and_sort(genomes);

    for window in sorted.windows(2) {
        assert!(
            window[0].1 >= window[1].1,
            "fitness values are not in descending order: {} before {}",
            window[0].1,
            window[1].1,
        );
    }
}

/// A custom threshold controls the exact fraction of the population kept.
#[test]
fn fitness_eliminator_custom_threshold() {
    // 10 genomes, threshold=0.3 → floor(10*0.3) + 1 = 4 survivors.
    let genomes: Vec<Genome> = (0..10).map(|i| Genome(i as f32)).collect();
    let mut eliminator = FitnessEliminator::new(fitness, 0.3, ());
    let survivors = eliminator.eliminate(genomes);
    assert_eq!(
        survivors.len(),
        4,
        "expected 4 survivors with threshold=0.3"
    );
}

/// [`FitnessEliminator::new`] must panic when the threshold is outside [0, 1].
#[test]
#[should_panic]
fn fitness_eliminator_invalid_threshold_panics() {
    FitnessEliminator::new(fitness, 1.5_f32, ());
}

// ─────────────────────────────────────────────────────────────────────────────
// FitnessEliminator builder
// ─────────────────────────────────────────────────────────────────────────────

/// The builder pattern must produce an eliminator with the specified settings.
#[test]
fn fitness_eliminator_builder() {
    let mut rng = rand::rng();
    let mut eliminator: FitnessEliminator<_, Genome, ()> = FitnessEliminator::builder()
        .fitness_fn(fitness)
        .threshold(0.4)
        .build();

    let genomes: Vec<Genome> = Vec::gen_random(&mut rng, 10);
    let _ = eliminator.eliminate(genomes);
    assert!((eliminator.threshold - 0.4).abs() < 1e-6);
}

// ─────────────────────────────────────────────────────────────────────────────
// FitnessObserver
// ─────────────────────────────────────────────────────────────────────────────

/// A counting observer used below.
struct Counter(usize);

impl FitnessObserver<Genome> for Counter {
    fn observe(&mut self, _: &[(Genome, f32)]) {
        self.0 += 1;
    }
}

/// The observer must be called exactly once per generation.
#[test]
fn observer_called_once_per_generation() {
    let mut rng = rand::rng();
    let eliminator = FitnessEliminator::new(fitness, 0.5, Counter(0));
    let mut sim = GeneticSim::new(
        Vec::<Genome>::gen_random(&mut rng, 10),
        eliminator,
        MitosisRepopulator::new(0.0, ()),
    );

    sim.perform_generations(7);
    assert_eq!(
        sim.eliminator.observer.0, 7,
        "observer must be called once per generation"
    );
}

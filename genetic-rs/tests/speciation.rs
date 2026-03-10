//! Integration tests for [`SpeciatedPopulation`], [`SpeciatedFitnessEliminator`],
//! and [`SpeciatedCrossoverRepopulator`].

use genetic_rs::prelude::*;
use genetic_rs::speciation::SpeciatedPopulation;

// ─────────────────────────────────────────────────────────────────────────────
// Shared test genome
// ─────────────────────────────────────────────────────────────────────────────

/// A genome whose species membership is fully determined by its integer `class`.
/// Two genomes with the same class have divergence 0.0; different classes give 1.0.
#[derive(Clone, Debug, PartialEq)]
struct Genome {
    class: i32,
    val: f32,
}

impl Speciated for Genome {
    type Context = ();

    fn divergence(&self, other: &Self, _: &()) -> f32 {
        if self.class == other.class {
            0.0
        } else {
            1.0
        }
    }
}

impl GenerateRandom for Genome {
    fn gen_random(rng: &mut impl rand::Rng) -> Self {
        Self {
            class: rng.random_range(0..3),
            val: rng.random(),
        }
    }
}

impl RandomlyMutable for Genome {
    type Context = ();

    fn mutate(&mut self, _: &(), rate: f32, rng: &mut impl rand::Rng) {
        self.val += rng.random::<f32>() * rate;
    }
}

impl Crossover for Genome {
    type Context = ();

    fn crossover(&self, other: &Self, _: &(), _rate: f32, _rng: &mut impl rand::Rng) -> Self {
        Self {
            class: self.class,
            val: (self.val + other.val) / 2.0,
        }
    }
}

fn fitness(g: &Genome) -> f32 {
    g.val
}

// ─────────────────────────────────────────────────────────────────────────────
// SpeciatedPopulation — species grouping
// ─────────────────────────────────────────────────────────────────────────────

/// All identical genomes must end up in a single species.
#[test]
fn identical_genomes_in_same_species() {
    let genomes: Vec<Genome> = (0..5).map(|_| Genome { class: 0, val: 0.0 }).collect();
    let pop = SpeciatedPopulation::from_genomes(&genomes, 0.5, &());
    assert_eq!(pop.species().len(), 1, "expected a single species");
    assert_eq!(
        pop.species()[0].len(),
        5,
        "all genomes must belong to the single species"
    );
}

/// Genomes of four distinct classes must form four separate species.
#[test]
fn different_class_genomes_in_different_species() {
    let genomes: Vec<Genome> = (0..4).map(|i| Genome { class: i, val: 0.0 }).collect();
    let pop = SpeciatedPopulation::from_genomes(&genomes, 0.5, &());
    assert_eq!(pop.species().len(), 4);
}

/// With a threshold > 1.0 (larger than the max divergence) all genomes,
/// regardless of class, must end up in the same species.
#[test]
fn high_threshold_groups_all_genomes() {
    let genomes: Vec<Genome> = (0..6)
        .map(|i| Genome {
            class: i % 3,
            val: 0.0,
        })
        .collect();
    // Divergence is at most 1.0; with threshold 1.5 everything is "close enough".
    let pop = SpeciatedPopulation::from_genomes(&genomes, 1.5, &());
    assert_eq!(pop.species().len(), 1, "all genomes must be in one species");
}

/// Every genome index must appear in exactly one species.
#[test]
fn every_genome_index_appears_exactly_once() {
    let n = 9usize;
    let genomes: Vec<Genome> = (0..n)
        .map(|i| Genome {
            class: (i % 3) as i32,
            val: i as f32,
        })
        .collect();
    let pop = SpeciatedPopulation::from_genomes(&genomes, 0.5, &());

    let mut seen = vec![false; n];
    for species in pop.species() {
        for &idx in species {
            assert!(
                !seen[idx],
                "genome index {idx} appears in more than one species"
            );
            seen[idx] = true;
        }
    }
    assert!(
        seen.iter().all(|&v| v),
        "not every genome index appeared in a species"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// SpeciatedPopulation — insert_genome
// ─────────────────────────────────────────────────────────────────────────────

/// Inserting a genome from a new class must create a new species.
#[test]
fn insert_genome_creates_new_species_for_novel_genome() {
    let genomes = vec![
        Genome { class: 0, val: 0.0 },
        Genome { class: 1, val: 0.0 }, // divergence 1.0 > threshold 0.5 → new species
    ];
    let mut pop = SpeciatedPopulation::new(0.5);
    pop.insert_genome(0, &genomes, &());
    let created_new = pop.insert_genome(1, &genomes, &());
    assert!(created_new, "expected a new species to be created");
    assert_eq!(pop.species().len(), 2);
}

/// Inserting a genome from an existing class must join that species.
#[test]
fn insert_genome_joins_existing_species_for_similar_genome() {
    let genomes = vec![
        Genome { class: 0, val: 0.0 },
        Genome { class: 0, val: 1.0 }, // divergence 0.0 < threshold 0.5 → joins species
    ];
    let mut pop = SpeciatedPopulation::new(0.5);
    pop.insert_genome(0, &genomes, &());
    let created_new = pop.insert_genome(1, &genomes, &());
    assert!(
        !created_new,
        "must not create a new species for a similar genome"
    );
    assert_eq!(pop.species().len(), 1);
    assert_eq!(pop.species()[0].len(), 2);
}

// ─────────────────────────────────────────────────────────────────────────────
// SpeciatedPopulation — round-robin iterator
// ─────────────────────────────────────────────────────────────────────────────

/// The round-robin iterator must visit every genome index within one full cycle.
/// With equal-sized species the cycle length equals the total number of genomes.
#[test]
fn round_robin_covers_all_genomes() {
    // 3 classes × 2 genomes each = 6 genomes, 3 species of equal size.
    let genomes: Vec<Genome> = (0..6)
        .map(|i| Genome {
            class: (i % 3) as i32,
            val: i as f32,
        })
        .collect();
    let pop = SpeciatedPopulation::from_genomes(&genomes, 0.5, &());

    let mut seen = vec![false; 6];
    for idx in pop.round_robin().take(6) {
        seen[idx] = true;
    }
    assert!(
        seen.iter().all(|&v| v),
        "round_robin must cover all genome indices in one cycle"
    );
}

/// The enumerate variant must yield matching (species_index, genome_index) pairs.
#[test]
fn round_robin_enumerate_species_index_is_valid() {
    let genomes: Vec<Genome> = (0..6)
        .map(|i| Genome {
            class: (i % 3) as i32,
            val: i as f32,
        })
        .collect();
    let pop = SpeciatedPopulation::from_genomes(&genomes, 0.5, &());

    for (species_i, genome_i) in pop.round_robin_enumerate().take(12) {
        assert!(
            pop.species()[species_i].contains(&genome_i),
            "genome index {genome_i} is not in species {species_i}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SpeciatedFitnessEliminator — fitness scaling and population-size invariants
// ─────────────────────────────────────────────────────────────────────────────

/// The population size must remain constant when using [`SpeciatedFitnessEliminator`]
/// together with [`SpeciatedCrossoverRepopulator`].
#[test]
fn speciated_sim_population_size_preserved() {
    let mut rng = rand::rng();
    let initial_size = 30;

    let eliminator = SpeciatedFitnessEliminator::new(fitness, 0.5, 0.5, (), ());
    let repopulator = SpeciatedCrossoverRepopulator::new(
        0.1,
        0.5,
        ActionIfIsolated::CrossoverSimilarSpecies,
        (),
        (),
    );

    let mut sim = GeneticSim::new(
        Vec::<Genome>::gen_random(&mut rng, initial_size),
        eliminator,
        repopulator,
    );
    sim.perform_generations(20);
    assert_eq!(sim.genomes.len(), initial_size);
}

/// Speciation must give fitness bonuses to rare/isolated genomes:
/// an isolated genome with *lower* raw fitness can out-compete a large species
/// with higher raw fitness because the large species has its fitness divided by
/// its size.
///
/// Setup:
/// - 4 genomes of class 0 with val = 1.0  → adjusted fitness = 1.0 / 4 = 0.25
/// - 1 genome  of class 1 with val = 0.5  → adjusted fitness = 0.5 / 1 = 0.5
///
/// With threshold = 0.5 and 5 genomes, 3 survive.
/// Sorted adjusted fitness: [0.5, 0.25, 0.25, 0.25, 0.25]
/// → the class-1 genome (raw 0.5) must survive despite the lower raw fitness.
#[test]
fn speciation_protects_rare_species() {
    let mut class0_genomes: Vec<Genome> = (0..4).map(|_| Genome { class: 0, val: 1.0 }).collect();
    let rare = Genome { class: 1, val: 0.5 };
    class0_genomes.push(rare.clone());

    let mut eliminator = SpeciatedFitnessEliminator::new(fitness, 0.5, 0.5, (), ());
    let survivors = eliminator.eliminate(class0_genomes);

    assert!(
        survivors.iter().any(|g| g == &rare),
        "the rare species genome must survive despite lower raw fitness"
    );
}

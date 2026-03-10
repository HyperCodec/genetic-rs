//! Integration tests for [`KnockoutEliminator`], [`KnockoutWinner`], and
//! related types.

use std::cmp::Ordering;

use genetic_rs::prelude::*;

// ─────────────────────────────────────────────────────────────────────────────
// Shared test genome
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
struct Genome(f32);

impl GenerateRandom for Genome {
    fn gen_random(rng: &mut impl rand::Rng) -> Self {
        Self(rng.random())
    }
}

/// Knockout function that keeps the genome with the smaller value.
/// `a.total_cmp(&b)` returns `Less` when a < b, mapping to `KnockoutWinner::First` (a survives).
fn smaller_wins(a: &Genome, b: &Genome) -> KnockoutWinner {
    a.0.total_cmp(&b.0).into()
}

// ─────────────────────────────────────────────────────────────────────────────
// KnockoutWinner helpers
// ─────────────────────────────────────────────────────────────────────────────

/// `!First == Second` and `!Second == First`.
#[test]
fn knockout_winner_not_impl() {
    assert_eq!(!KnockoutWinner::First, KnockoutWinner::Second);
    assert_eq!(!KnockoutWinner::Second, KnockoutWinner::First);
}

/// `From<Ordering>` must map correctly.
#[test]
fn knockout_winner_from_ordering() {
    assert_eq!(KnockoutWinner::from(Ordering::Less), KnockoutWinner::First);
    assert_eq!(KnockoutWinner::from(Ordering::Equal), KnockoutWinner::First);
    assert_eq!(
        KnockoutWinner::from(Ordering::Greater),
        KnockoutWinner::Second
    );
}

/// `Into<usize>`: `First` → 0, `Second` → 1.
#[test]
fn knockout_winner_into_usize() {
    assert_eq!(usize::from(KnockoutWinner::First), 0usize);
    assert_eq!(usize::from(KnockoutWinner::Second), 1usize);
}

// ─────────────────────────────────────────────────────────────────────────────
// KnockoutEliminator output-size invariants
// ─────────────────────────────────────────────────────────────────────────────

/// With an even-sized population the output must be exactly half.
#[test]
fn knockout_output_half_size_even_input() {
    let genomes: Vec<Genome> = (0..10).map(|i| Genome(i as f32)).collect();
    let mut elim = KnockoutEliminator::new(smaller_wins, ActionIfOdd::Panic);
    let survivors = elim.eliminate(genomes);
    assert_eq!(survivors.len(), 5);
}

/// `ActionIfOdd::KeepSingle` with 11 genomes: 1 bypasses contest + 5 winners = 6.
#[test]
fn knockout_action_if_odd_keep_single() {
    let genomes: Vec<Genome> = (0..11).map(|i| Genome(i as f32)).collect();
    let mut elim = KnockoutEliminator::new(smaller_wins, ActionIfOdd::KeepSingle);
    let survivors = elim.eliminate(genomes);
    assert_eq!(survivors.len(), 6);
}

/// `ActionIfOdd::DeleteSingle` with 11 genomes: 1 is discarded + 5 winners = 5.
#[test]
fn knockout_action_if_odd_delete_single() {
    let genomes: Vec<Genome> = (0..11).map(|i| Genome(i as f32)).collect();
    let mut elim = KnockoutEliminator::new(smaller_wins, ActionIfOdd::DeleteSingle);
    let survivors = elim.eliminate(genomes);
    assert_eq!(survivors.len(), 5);
}

/// `ActionIfOdd::Panic` must panic when given an odd number of genomes.
#[test]
#[should_panic]
fn knockout_action_if_odd_panic_panics() {
    let genomes: Vec<Genome> = (0..11).map(|i| Genome(i as f32)).collect();
    let mut elim = KnockoutEliminator::new(smaller_wins, ActionIfOdd::Panic);
    let _ = elim.eliminate(genomes);
}

/// With a population of 1, [`KnockoutEliminator`] must return it unchanged.
#[test]
fn knockout_single_genome_returns_unchanged() {
    let genomes = vec![Genome(42.0)];
    let mut elim = KnockoutEliminator::new(smaller_wins, ActionIfOdd::Panic);
    let survivors = elim.eliminate(genomes);
    assert_eq!(survivors.len(), 1);
    assert!((survivors[0].0 - 42.0).abs() < 1e-6);
}

// ─────────────────────────────────────────────────────────────────────────────
// KnockoutEliminator correctness
// ─────────────────────────────────────────────────────────────────────────────

/// In each pair `(a, b)` the correct genome must survive based on the knockout fn.
///
/// With genomes [0, 1, 2, …, 9] and `smaller_wins`:
/// - Pairs: (0,1), (2,3), (4,5), (6,7), (8,9)
/// - `a < b` → `Less` → `First` → a survives → even-indexed values survive.
#[test]
fn knockout_correct_genome_survives() {
    let genomes: Vec<Genome> = (0..10).map(|i| Genome(i as f32)).collect();
    let mut elim = KnockoutEliminator::new(smaller_wins, ActionIfOdd::Panic);
    let survivors = elim.eliminate(genomes);

    for g in &survivors {
        assert!(
            (g.0 as i32) % 2 == 0,
            "expected even (smaller) value to survive each pair, got {}",
            g.0
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FitnessKnockoutFn
// ─────────────────────────────────────────────────────────────────────────────

/// [`FitnessKnockoutFn`] must delegate to the inner fitness function and
/// produce a consistent result based on the `afit.total_cmp(&bfit)` comparison.
///
/// The `From<Ordering>` mapping is:
/// - `Less`    → `First`  (a survives — a had lower fitness)
/// - `Greater` → `Second` (b survives — b had lower fitness)
///
/// This tests the actual implementation behaviour rather than any particular
/// notion of "higher fitness wins"; it guards against regressions in the
/// delegation and ordering logic.
#[test]
fn fitness_knockout_fn_delegates_to_fitness_fn() {
    let ko_fn = FitnessKnockoutFn::new(|g: &Genome| g.0);
    let a = Genome(1.0);
    let b = Genome(9.0);

    // 1.0.total_cmp(&9.0) = Less → First (a survives)
    assert_eq!(ko_fn.knockout(&a, &b), KnockoutWinner::First);

    // 9.0.total_cmp(&1.0) = Greater → Second (b survives)
    assert_eq!(ko_fn.knockout(&b, &a), KnockoutWinner::Second);
}

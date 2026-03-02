//! Integration tests for FitnessEliminator and related types.

use std::cell::Cell;

use genetic_rs::prelude::*;

struct CountingObserver<'a>(&'a Cell<usize>);

impl<'a, G> FitnessObserver<G> for CountingObserver<'a> {
    fn observe(&self, _fitnesses: &[(G, f32)]) {
        self.0.set(self.0.get() + 1);
    }
}

#[test]
fn layered_observer_calls_both() {
    let count_a = Cell::new(0usize);
    let count_b = Cell::new(0usize);

    let layered = CountingObserver(&count_a).layer(CountingObserver(&count_b));

    let fitnesses: Vec<((), f32)> = vec![((), 1.0)];
    layered.observe(&fitnesses);

    assert_eq!(count_a.get(), 1);
    assert_eq!(count_b.get(), 1);
}

#[test]
fn layered_observer_triple_layer() {
    let count_a = Cell::new(0usize);
    let count_b = Cell::new(0usize);
    let count_c = Cell::new(0usize);

    let layered = CountingObserver(&count_a)
        .layer(CountingObserver(&count_b))
        .layer(CountingObserver(&count_c));

    let fitnesses: Vec<((), f32)> = vec![((), 1.0)];
    layered.observe(&fitnesses);

    assert_eq!(count_a.get(), 1);
    assert_eq!(count_b.get(), 1);
    assert_eq!(count_c.get(), 1);
}

//! Integration tests for FitnessEliminator and related types.

use std::cell::Cell;

use genetic_rs::prelude::*;

struct CountingObserver<'a>(&'a Cell<usize>);

impl<'a, G> FitnessObserver<G> for CountingObserver<'a> {
    fn observe(&mut self, _fitnesses: &[(G, f32)]) {
        self.0.set(self.0.get() + 1);
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct TrackingObserver(usize);

impl<G> FitnessObserver<G> for TrackingObserver {
    fn observe(&mut self, _fitnesses: &[(G, f32)]) {
        self.0 += 1;
    }
}

#[test]
fn layered_observer_calls_both() {
    let count_a = Cell::new(0usize);
    let count_b = Cell::new(0usize);

    let mut layered = CountingObserver(&count_a).layer(CountingObserver(&count_b));

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

    let mut layered = CountingObserver(&count_a)
        .layer(CountingObserver(&count_b))
        .layer(CountingObserver(&count_c));

    let fitnesses: Vec<((), f32)> = vec![((), 1.0)];
    layered.observe(&fitnesses);

    assert_eq!(count_a.get(), 1);
    assert_eq!(count_b.get(), 1);
    assert_eq!(count_c.get(), 1);
}

#[test]
fn layered_observer_default() {
    let layered: LayeredObserver<(), TrackingObserver, TrackingObserver> =
        LayeredObserver::default();
    assert_eq!(layered.0, TrackingObserver(0));
    assert_eq!(layered.1, TrackingObserver(0));
}

#[test]
fn layered_observer_clone() {
    let original = TrackingObserver::default().layer(TrackingObserver::default());
    let mut cloned: LayeredObserver<(), _, _> = original.clone();

    let fitnesses: Vec<((), f32)> = vec![((), 1.0)];
    cloned.observe(&fitnesses);

    assert_eq!(cloned.0, TrackingObserver(1));
    assert_eq!(cloned.1, TrackingObserver(1));
    assert_eq!(original.0, TrackingObserver(0));
    assert_eq!(original.1, TrackingObserver(0));
}

#[test]
fn layered_observer_debug() {
    let layered: LayeredObserver<(), TrackingObserver, TrackingObserver> =
        TrackingObserver::default().layer(TrackingObserver::default());
    let debug_str = format!("{:?}", layered);
    assert!(debug_str.contains("LayeredObserver"));
    assert!(debug_str.contains("TrackingObserver"));
}

#[test]
fn layered_observer_partial_eq() {
    let a: LayeredObserver<(), _, _> = TrackingObserver::default().layer(TrackingObserver::default());
    let b: LayeredObserver<(), _, _> = TrackingObserver::default().layer(TrackingObserver::default());
    assert_eq!(a, b);
}

#[test]
fn layered_observer_partial_eq_not_equal() {
    let a: LayeredObserver<(), _, _> = TrackingObserver(1).layer(TrackingObserver(0));
    let b: LayeredObserver<(), _, _> = TrackingObserver(0).layer(TrackingObserver(0));
    assert_ne!(a, b);
}

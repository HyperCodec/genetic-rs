//! Integration tests for the derive macros.
//!
//! These tests guard against macro regressions (see issues #124 and #127).

#![allow(dead_code)]

use genetic_rs::prelude::*;

// ──────────────────────────────────────────────────────────────────────────────
// Helpers shared across multiple tests
// ──────────────────────────────────────────────────────────────────────────────

/// Newtype around f32 so we can implement traits without orphan-rule violations.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Val(f32);

impl RandomlyMutable for Val {
    type Context = ();

    fn mutate(&mut self, _ctx: &Self::Context, rate: f32, rng: &mut impl Rng) {
        self.0 += rng.random_range(-rate..=rate);
    }
}

impl Mitosis for Val {
    type Context = ();

    fn divide(&self, _ctx: &Self::Context, _rate: f32, _rng: &mut impl Rng) -> Self {
        *self
    }
}

impl GenerateRandom for Val {
    fn gen_random(rng: &mut impl rand::Rng) -> Self {
        Val(rng.random())
    }
}

impl Crossover for Val {
    type Context = ();

    fn crossover(&self, other: &Self, _ctx: &(), _rate: f32, _rng: &mut impl Rng) -> Self {
        Val((self.0 + other.0) / 2.0)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// RandomlyMutable
// ──────────────────────────────────────────────────────────────────────────────

/// Plain named struct – context inferred from the first field.
#[derive(Clone, RandomlyMutable)]
struct RandMutNamed {
    x: Val,
    y: Val,
}

#[test]
fn randmut_named_struct() {
    let mut rng = rand::rng();
    let mut genome = RandMutNamed {
        x: Val(0.0),
        y: Val(0.0),
    };
    genome.mutate(&(), 0.5, &mut rng);
}

/// Tuple struct.
#[derive(Clone, RandomlyMutable)]
struct RandMutTuple(Val, Val);

#[test]
fn randmut_tuple_struct() {
    let mut rng = rand::rng();
    let mut genome = RandMutTuple(Val(0.0), Val(0.0));
    genome.mutate(&(), 0.5, &mut rng);
}

/// Empty struct – context should be `()`.
#[derive(Clone, RandomlyMutable)]
struct RandMutEmpty {}

#[test]
fn randmut_empty_struct() {
    let mut rng = rand::rng();
    let mut genome = RandMutEmpty {};
    genome.mutate(&(), 0.0, &mut rng);
}

/// Per-field contexts – `create_context` generates a context struct.
#[derive(Clone, Debug, Default)]
struct CtxA;

#[derive(Clone, Debug, Default)]
struct CtxB;

#[derive(Clone)]
struct FieldA;
impl RandomlyMutable for FieldA {
    type Context = CtxA;
    fn mutate(&mut self, _ctx: &CtxA, _rate: f32, _rng: &mut impl Rng) {}
}

#[derive(Clone)]
struct FieldB;
impl RandomlyMutable for FieldB {
    type Context = CtxB;
    fn mutate(&mut self, _ctx: &CtxB, _rate: f32, _rng: &mut impl Rng) {}
}

#[derive(Clone, RandomlyMutable)]
#[randmut(create_context(name = CreatedCtx, derive(Clone, Debug, Default)))]
struct RandMutWithCreatedCtx {
    a: FieldA,
    b: FieldB,
}

#[test]
fn randmut_create_context() {
    let mut rng = rand::rng();
    let ctx = CreatedCtx::default();
    // Verify the generated context has the correct field types.
    let _: &CtxA = &ctx.a;
    let _: &CtxB = &ctx.b;
    let mut genome = RandMutWithCreatedCtx {
        a: FieldA,
        b: FieldB,
    };
    genome.mutate(&ctx, 0.5, &mut rng);
}

/// `with_context` attribute reuses an existing context type.
#[derive(Clone, RandomlyMutable)]
#[randmut(with_context = CreatedCtx)]
struct RandMutWithProvidedCtx {
    a: FieldA,
    b: FieldB,
}

#[test]
fn randmut_with_context() {
    let mut rng = rand::rng();
    let ctx = CreatedCtx::default();
    let mut genome = RandMutWithProvidedCtx {
        a: FieldA,
        b: FieldB,
    };
    genome.mutate(&ctx, 0.5, &mut rng);
}

// ──────────────────────────────────────────────────────────────────────────────
// Mitosis
// ──────────────────────────────────────────────────────────────────────────────

#[derive(Clone)]
struct MitFieldA;
impl Mitosis for MitFieldA {
    type Context = CtxA;
    fn divide(&self, _ctx: &CtxA, _rate: f32, _rng: &mut impl Rng) -> Self {
        MitFieldA
    }
}

#[derive(Clone)]
struct MitFieldB;
impl Mitosis for MitFieldB {
    type Context = CtxB;
    fn divide(&self, _ctx: &CtxB, _rate: f32, _rng: &mut impl Rng) -> Self {
        MitFieldB
    }
}

/// Plain named struct – context inferred from the first field.
#[derive(Clone, Mitosis)]
struct MitosisNamed {
    a: MitFieldA,
}

#[test]
fn mitosis_named_struct_shared_ctx() {
    let mut rng = rand::rng();
    let genome = MitosisNamed { a: MitFieldA };
    let _child = genome.divide(&CtxA, 0.5, &mut rng);
}

/// Empty struct – context should be `()`.
#[derive(Clone, Mitosis)]
struct MitosisEmpty {}

#[test]
fn mitosis_empty_struct() {
    let mut rng = rand::rng();
    let genome = MitosisEmpty {};
    let _child = genome.divide(&(), 0.0, &mut rng);
}

/// `use_randmut = true` – delegates to `RandomlyMutable`.
#[derive(Clone, RandomlyMutable, Mitosis)]
#[mitosis(use_randmut = true)]
struct MitosisUseRandmut(Val);

#[test]
fn mitosis_use_randmut_true() {
    let mut rng = rand::rng();
    let genome = MitosisUseRandmut(Val(1.0));
    let _child = genome.divide(&(), 0.5, &mut rng);
}

/// `use_randmut = false` previously caused a proc-macro panic (issue #127).
/// Verify it now compiles and behaves like a normal Mitosis derive.
#[derive(Clone, Mitosis)]
#[mitosis(use_randmut = false)]
struct MitosisUseRandmutFalse {
    a: MitFieldA,
}

#[test]
fn mitosis_use_randmut_false_no_panic() {
    let mut rng = rand::rng();
    let genome = MitosisUseRandmutFalse { a: MitFieldA };
    // Context is inferred from the first field.
    let _child = genome.divide(&CtxA, 0.5, &mut rng);
}

/// `create_context` with `Mitosis` must generate fields typed as
/// `<FieldTy as Mitosis>::Context`, NOT `<FieldTy as RandomlyMutable>::Context`.
/// The wrong trait was used in 1.2.0 (issue #127).
#[derive(Clone, Mitosis)]
#[mitosis(create_context(name = MitosisCreatedCtx, derive(Clone, Debug, Default)))]
struct MitosisWithCreatedCtx {
    a: MitFieldA,
    b: MitFieldB,
}

#[test]
fn mitosis_create_context_uses_mitosis_context() {
    let mut rng = rand::rng();
    let ctx = MitosisCreatedCtx::default();
    // MitosisCreatedCtx.a must be <MitFieldA as Mitosis>::Context = CtxA
    // MitosisCreatedCtx.b must be <MitFieldB as Mitosis>::Context = CtxB
    let _: &CtxA = &ctx.a;
    let _: &CtxB = &ctx.b;

    let genome = MitosisWithCreatedCtx {
        a: MitFieldA,
        b: MitFieldB,
    };
    let _child = genome.divide(&ctx, 0.5, &mut rng);
}

/// `with_context` for Mitosis reuses an existing context struct.
#[derive(Clone, Mitosis)]
#[mitosis(with_context = MitosisCreatedCtx)]
struct MitosisWithProvidedCtx {
    a: MitFieldA,
    b: MitFieldB,
}

#[test]
fn mitosis_with_context() {
    let mut rng = rand::rng();
    let ctx = MitosisCreatedCtx::default();
    let genome = MitosisWithProvidedCtx {
        a: MitFieldA,
        b: MitFieldB,
    };
    let _child = genome.divide(&ctx, 0.5, &mut rng);
}

/// Tuple struct with Mitosis.
#[derive(Clone, Mitosis)]
struct MitosisTuple(Val);

#[test]
fn mitosis_tuple_struct() {
    let mut rng = rand::rng();
    let genome = MitosisTuple(Val(0.0));
    let _child = genome.divide(&(), 0.5, &mut rng);
}

// ──────────────────────────────────────────────────────────────────────────────
// GenerateRandom
// ──────────────────────────────────────────────────────────────────────────────

/// Named struct.
#[derive(Clone, GenerateRandom)]
struct GenRandNamed {
    x: Val,
    y: Val,
}

#[test]
fn genrand_named_struct() {
    let mut rng = rand::rng();
    let _genome = GenRandNamed::gen_random(&mut rng);
}

/// Tuple struct.
#[derive(Clone, GenerateRandom)]
struct GenRandTuple(Val, Val);

#[test]
fn genrand_tuple_struct() {
    let mut rng = rand::rng();
    let _genome = GenRandTuple::gen_random(&mut rng);
}

/// Empty struct.
#[derive(Clone, GenerateRandom)]
struct GenRandEmpty {}

#[test]
fn genrand_empty_struct() {
    let mut rng = rand::rng();
    let _genome = GenRandEmpty::gen_random(&mut rng);
}

// ──────────────────────────────────────────────────────────────────────────────
// Crossover
// ──────────────────────────────────────────────────────────────────────────────

#[derive(Clone)]
struct CrossFieldA;
impl Crossover for CrossFieldA {
    type Context = CtxA;
    fn crossover(&self, _other: &Self, _ctx: &CtxA, _rate: f32, _rng: &mut impl Rng) -> Self {
        CrossFieldA
    }
}

#[derive(Clone)]
struct CrossFieldB;
impl Crossover for CrossFieldB {
    type Context = CtxB;
    fn crossover(&self, _other: &Self, _ctx: &CtxB, _rate: f32, _rng: &mut impl Rng) -> Self {
        CrossFieldB
    }
}

/// Named struct – shared context inferred from the first field.
#[derive(Clone, Crossover)]
struct CrossoverNamed {
    a: CrossFieldA,
}

#[test]
fn crossover_named_struct_shared_ctx() {
    let mut rng = rand::rng();
    let g1 = CrossoverNamed { a: CrossFieldA };
    let g2 = CrossoverNamed { a: CrossFieldA };
    let _child = g1.crossover(&g2, &CtxA, 0.5, &mut rng);
}

/// `create_context` with Crossover.
#[derive(Clone, Crossover)]
#[crossover(create_context(name = CrossoverCreatedCtx, derive(Clone, Debug, Default)))]
struct CrossoverWithCreatedCtx {
    a: CrossFieldA,
    b: CrossFieldB,
}

#[test]
fn crossover_create_context() {
    let mut rng = rand::rng();
    let ctx = CrossoverCreatedCtx::default();
    let g1 = CrossoverWithCreatedCtx {
        a: CrossFieldA,
        b: CrossFieldB,
    };
    let g2 = CrossoverWithCreatedCtx {
        a: CrossFieldA,
        b: CrossFieldB,
    };
    let _child = g1.crossover(&g2, &ctx, 0.5, &mut rng);
}

/// Tuple struct with Crossover.
#[derive(Clone, Crossover)]
struct CrossoverTuple(Val);

#[test]
fn crossover_tuple_struct() {
    let mut rng = rand::rng();
    let g1 = CrossoverTuple(Val(1.0));
    let g2 = CrossoverTuple(Val(2.0));
    let child = g1.crossover(&g2, &(), 0.5, &mut rng);
    assert_eq!(child.0, Val(1.5));
}

/// Empty struct with Crossover.
#[derive(Clone, Crossover)]
struct CrossoverEmpty {}

#[test]
fn crossover_empty_struct() {
    let mut rng = rand::rng();
    let g1 = CrossoverEmpty {};
    let g2 = CrossoverEmpty {};
    let _child = g1.crossover(&g2, &(), 0.0, &mut rng);
}

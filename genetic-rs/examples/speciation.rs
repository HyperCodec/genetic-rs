use genetic_rs::prelude::*;

/// Max value of divergence() that will count two genomes as the same species.
const SPECIATION_THRESHOLD: f32 = 0.1;

#[derive(Clone, PartialEq, Debug)]
struct MyGenome {
    vals: Vec<f32>,
}

impl GenerateRandom for MyGenome {
    fn gen_random(rng: &mut impl rand::Rng) -> Self {
        Self {
            vals: (0..10).map(|_| rng.random_range(-1.0..1.0)).collect(),
        }
    }
}

impl RandomlyMutable for MyGenome {
    type Context = ();

    fn mutate(&mut self, _ctx: &Self::Context, rate: f32, rng: &mut impl rand::Rng) {
        // internal mutation
        for val in &mut self.vals {
            if rng.random_bool(rate as f64) {
                *val += rng.random_range(-0.1..0.1);
            }
        }

        // structural mutations
        if rng.random_bool(rate as f64) {
            // add a new random gene
            self.vals.push(rng.random_range(-1.0..1.0));
        }
        if self.vals.len() > 1 && rng.random_bool(rate as f64) {
            // remove a random gene
            let index = rng.random_range(0..self.vals.len());
            self.vals.remove(index);
        }
    }
}

impl Mitosis for MyGenome {
    type Context = ();

    fn divide(&self, ctx: &<Self as Mitosis>::Context, rate: f32, rng: &mut impl rand::Rng)
            -> Self {
        let mut child = self.clone();
        child.mutate(ctx, rate, rng);
        child
    }
}

impl Crossover for MyGenome {
    type Context = ();

    fn crossover(&self, other: &Self, ctx: &Self::Context, rate: f32, rng: &mut impl rand::Rng) -> Self {
        // even though they ideally shouldn't crossover if they aren't the same length
        // because of speciation, we can still allow it in emergency scenarios to prevent extinction
        // and encourage more diversity, at the cost of more broken or dysfunctional species
        let mut child;
        let smaller;
        if self.vals.len() < other.vals.len() {
            child = other.clone();
            smaller = self;
        } else {
            child = self.clone();
            smaller = other;
        }

        for i in 0..smaller.vals.len() {
            if rng.random_bool(0.5) {
                child.vals[i] = smaller.vals[i];
            }
        }

        child.mutate(ctx, rate, rng);

        child
    }
}

impl Speciated for MyGenome {
    type Context = ();

    fn divergence(&self, other: &Self, _ctx: &Self::Context) -> f32 {
        // distance in lengths divided by the larger length
        let larger = self.vals.len().max(other.vals.len());
        // can't be 0 because of how we set up mutate()
        assert!(larger != 0);
        let length_diff = (self.vals.len() as isize - other.vals.len() as isize).abs() as f32;
        length_diff / larger as f32
    }
}

fn fitness(genome: &MyGenome) -> f32 {
    // the fitness function is just the sum of the genes, which encourages longer genomes.
    // however, since the structure mutation can add negative values initially, speciation
    // will help protect those genomes until they can mutate the internal state into positive values.
    genome.vals.iter().sum()
}

fn print_fitnesses(fitnesses: &[(MyGenome, f32)]) {
    // note that with SpeciatedFitnessEliminator,
    // these values are divided by the number of genomes in the species.
    let hi = fitnesses[0].1;
    let med = fitnesses[fitnesses.len() / 2].1;
    let lo = fitnesses[fitnesses.len()-1].1;

    println!("hi: {hi} med: {med} lo: {lo}");
}

fn main() {
    let mut rng = rand::rng();

    let fitness_eliminator = FitnessEliminator::builder()
        .fitness_fn(fitness)
        .observer(print_fitnesses)
        .build_or_panic();

    let crossover_rep = CrossoverRepopulator::default();

    let mut sim = GeneticSim::new(
        Vec::gen_random(&mut rng, 100),
        SpeciatedFitnessEliminator::from_fitness_eliminator(fitness_eliminator, SPECIATION_THRESHOLD, ()),
        SpeciatedCrossoverRepopulator::from_crossover(crossover_rep, SPECIATION_THRESHOLD, ActionIfIsolated::CrossoverSimilarSpecies, ())
    );

    sim.perform_generations(100);

    // rerunning multiple times should show a lot of diversity in genome length
    dbg!(&sim.genomes[0]);
}

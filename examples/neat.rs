use std::sync::{Arc, RwLock};

use genetic_rs::prelude::*;
use rand::prelude::*;

fn simulate_agent(dna: &AgentDNA, max_steps: usize, rng: &mut impl Rng) -> f32 {
    let mut agent = Agent::from(dna);

    let mut food_pos: (usize, usize);
    let mut agent_pos: (usize, usize);

    let mut reward = 0.;
    let mut steps = 0;

    loop {
        // game cycle (resets map after agent gets food)
        agent_pos = (rng.gen_range(0..10), rng.gen_range(0..10));
        food_pos = (rng.gen_range(0..10), rng.gen_range(0..10));

        while agent_pos == food_pos {
            food_pos = (rng.gen_range(0..10), rng.gen_range(0..10));
        }

        loop {
            // movement cycle

            // input (relative location of the food)
            let ai_input = vec![food_pos.0 as f32 - agent_pos.0 as f32, food_pos.1 as f32 - agent_pos.1 as f32];

            let output = agent.network.predict(ai_input);
            agent.network.flush_state();

            if output[0] >= output[1] {
                agent_pos.0 += 1;
            } else {
                agent_pos.1 += 1;
            }

            if agent_pos == food_pos {
                reward += 10.;
                break; // new board
            } else {
                reward -= 0.01; // encourage agent to get there as fast as possible, but don't discourage exploration.
            }

            steps += 1;

            if steps == max_steps {
                return reward;
            }
        }
    }
}

struct Agent {
    network: NeuralNetwork,
}

impl From<&AgentDNA> for Agent {
    fn from(value: &AgentDNA) -> Self {
        Self {
            network: NeuralNetwork::from(&value.network),
        }
    }
}

#[derive(Clone)]
struct AgentDNA {
    network: StatelessNeuralNetwork,
}

impl RandomlyMutable for AgentDNA {
    fn mutate(&mut self, rate: f32, rng: &mut impl rand::Rng) {
        self.network.mutate(rate, rng);
    }
}

impl DivisionReproduction for AgentDNA {
    fn spawn_child(&self, rng: &mut impl rand::Rng) -> Self {
        Self {
            network: self.network.spawn_child(rng),
        }
    }
}

impl Prunable for AgentDNA {}

impl GenerateRandom for AgentDNA {
    fn gen_random(_rng: &mut impl Rng) -> Self {
        Self {
            network: StatelessNeuralNetwork::new(2, 3, 2),
        }
    }
}

fn main() {
    let mut rng = rand::thread_rng();

    let max_steps = Arc::new(RwLock::new(10));
    let ms = max_steps.clone();

    let seed = Arc::new(RwLock::new(rng.gen::<[u8; 32]>()));
    let s = seed.clone();

    let fitness = move |dna: &AgentDNA| {
        let msr = ms.read().unwrap();
        let sr = s.read().unwrap();
        simulate_agent(dna, *msr, &mut StdRng::from_seed(*sr))
    };

    let mut sim = GeneticSim::new(
        Vec::gen_random(&mut rng, 100),
        fitness.clone(), // just cloned for the sum stuff later
        division_pruning_nextgen,
    );

    for _ in 0..100 {
        // shouldn't need to do anything with `sim.entities` since this is still simple enough to run within the scope of the fitness function.
        sim.next_generation();

        // increase step count with each generation
        let mut msw = max_steps.write().unwrap();
        *msw += 1;

        // change seed
        let mut sw = seed.write().unwrap();
        *sw = rng.gen();
    }

    // test networks
    let fit: Vec<_> = sim.entities
        .iter()
        .map(fitness)
        .collect();
    
    // collect stats
    let maxfit = *fit
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    let sum: f32 = fit
        .iter()
        .sum();

    let avg = sum / sim.entities.len() as f32;

    dbg!(maxfit, avg);
}
use crate::prelude::*;
use std::{borrow::Borrow, rc::Rc};

#[derive(Clone)]
pub struct StatelessNeuralNetwork {
    input_layer: Vec<Rc<StatelessNeuron>>,
    hidden_layers: Vec<Rc<StatelessNeuron>>,
    output_layer: Vec<Rc<StatelessNeuron>>,
}

impl StatelessNeuralNetwork {
    pub fn new(inputs: usize, hidden: usize, outputs: usize) -> Self {
        let mut rng = rand::thread_rng();

        let input_layer: Vec<_> = (0..inputs)
            .map(|i| Rc::new(StatelessNeuron::new(vec![], NeuronLocator::Input(i), &mut rng)))
            .collect();

        let hidden_layers: Vec<_> = (0..hidden)
            .map(|i| Rc::new(StatelessNeuron::new(input_layer.clone(), NeuronLocator::Hidden(i), &mut rng)))
            .collect();

        let output_layer: Vec<_> = (0..outputs)
            .map(|i| Rc::new(StatelessNeuron::new(hidden_layers.clone(), NeuronLocator::Output(i), &mut rng)))
            .collect();

        Self {
            input_layer,
            hidden_layers,
            output_layer,
        }
    }

    fn rand_neuron_mut(&mut self, rng: &mut impl rand::Rng) -> (&mut Rc<StatelessNeuron>, NeuronLocator) {
        if rng.gen::<f32>() <= 0.5 {
            let i = rng.gen_range(0..self.output_layer.len());
            return (&mut self.output_layer[i], NeuronLocator::Output(i));
        }

        let i = rng.gen_range(0..self.hidden_layers.len());
        (&mut self.hidden_layers[i], NeuronLocator::Hidden(i))
    }


    fn rand_neuron(&self, rng: &mut impl rand::Rng) -> (&Rc<StatelessNeuron>, NeuronLocator) {
        if rng.gen::<f32>() <= 0.5 {
            let i = rng.gen_range(0..self.output_layer.len());
            return (&self.output_layer[i], NeuronLocator::Output(i));
        }

        let i = rng.gen_range(0..self.hidden_layers.len());
        (&self.hidden_layers[i], NeuronLocator::Hidden(i))
    }

    fn is_connection_safe(&self, n1: &Rc<StatelessNeuron>, n2: &Rc<StatelessNeuron>) -> bool {
        // check if connection is safe (going n2 -> n1 if represented by forward propagation).
        for (n, _w) in &n2.inputs {
            if n == n1 || !self.is_connection_safe(n1, n) {
                return false; // if returned, instantly escape entire recursion.
            }
        }

        true // only returned once it reaches input layer (or some other neuron with no inputs)
    }
}

impl RandomlyMutable for StatelessNeuralNetwork {
    fn mutate(&mut self, rate: f32, rng: &mut impl rand::Rng) {
        // network-wide mutation
        let mutation = NetworkWideMutation::gen_random(rng);

        match mutation {
            NetworkWideMutation::AddConnection => {
                // add connection between two neurons, but take caution to make sure it isn't looping into itself.
                let (mut n1, mut loc1) = self.rand_neuron(rng);
                let (mut n2, mut loc2) = self.rand_neuron(rng);

                // search for valid neuron pair
                while !self.is_connection_safe(n1, n2) {
                    (n1, loc1) = self.rand_neuron(rng);
                    (n2, loc2) = self.rand_neuron(rng);
                }

                let n2 = match loc2 {
                    NeuronLocator::Input(i) => self.input_layer[i].clone(),
                    NeuronLocator::Hidden(i) => self.hidden_layers[i].clone(),
                    NeuronLocator::Output(i) => self.output_layer[i].clone(),
                };

                let n1 = match loc1 {
                    NeuronLocator::Input(i) => Rc::get_mut(&mut self.input_layer[i]).unwrap(),
                    NeuronLocator::Hidden(i) => Rc::get_mut(&mut self.hidden_layers[i]).unwrap(),
                    NeuronLocator::Output(i) => Rc::get_mut(&mut self.output_layer[i]).unwrap(),
                };

                n1.inputs.push((n2, rng.gen::<f32>()));
            },
            NetworkWideMutation::RemoveConnection => {
                let n = Rc::get_mut(self.rand_neuron_mut(rng).0).unwrap();
                n.inputs.remove(rng.gen_range(0..n.inputs.len()));
            },
            NetworkWideMutation::AddNeuron => {
                // split preexisting connection to put new neuron in.
                let n = Rc::get_mut(self.rand_neuron_mut(rng).0).unwrap();

                let i = rng.gen_range(0..n.inputs.len());
                let (n2, w) = n.inputs.remove(i);
                let n3 = Rc::new(StatelessNeuron::new(vec![n2], NeuronLocator::Input(i), rng));

                n.inputs.push((n3, w));
            },
            NetworkWideMutation::RemoveNeuron => {
                self.hidden_layers.remove(rng.gen_range(0..self.hidden_layers.len()));
            },
        }

        // change weights anyway
        for n in self.hidden_layers.iter_mut() {
            for (_n, w) in Rc::get_mut(n).unwrap().inputs.iter_mut() {
                if rng.gen::<f32>() < rate {
                    *w += rng.gen::<f32>() * rate;
                }
            }
        }

        for n in self.output_layer.iter_mut() {
            for (_n, w) in Rc::get_mut(n).unwrap().inputs.iter_mut() {
                if rng.gen::<f32>() < rate {
                    *w += rng.gen::<f32>() * rate;
                }
            }
        }
    }
}

impl DivisionReproduction for StatelessNeuralNetwork {
    fn spawn_child(&self, rng: &mut impl rand::Rng) -> Self {
        let mut child = self.clone();
        child.mutate(0.01, rng); // TODO customizable rate
        child
    }
}

/// An enum to organize network mutation types.
pub enum NetworkWideMutation {
    /// Adds a connection between two neurons.
    AddConnection,

    /// Removes a connection between two neurons.
    RemoveConnection,

    /// Splits an existing connection by placing a neuron in the middle.
    AddNeuron,

    /// Removes a neuron and the connections surrounding it.
    RemoveNeuron,
}

#[derive(Clone, PartialEq)]
pub struct StatelessNeuron {
    inputs: Vec<(Rc<StatelessNeuron>, f32)>,
    bias: f32,
    location: NeuronLocator,
}

impl StatelessNeuron {
    pub fn new(inputs: Vec<Rc<StatelessNeuron>>, location: NeuronLocator, rng: &mut impl rand::Rng) -> Self {
        let inputs = inputs
            .into_iter()
            .map(|r| (r, rng.gen::<f32>()))
            .collect();

        let bias = rng.gen::<f32>();

        Self {
            inputs,
            bias,
            location,
        }
    }
}

#[derive(Clone, PartialEq)]
enum NeuronLocator {
    Input(usize),
    Hidden(usize),
    Output(usize),
}

/// A builtin struct that uses the NEAT (Neuro-Evolution Augmented Topology) algorithm.
/// TODO example
#[derive(Clone)]
pub struct NeuralNetwork {
    input_layer: Vec<Rc<Neuron>>,
    hidden_layers: Vec<Rc<Neuron>>,
    output_layer: Vec<Rc<Neuron>>,
}

impl NeuralNetwork {
    /// Creates a simple neural network with 1 hidden layer. This is so that it is still able to be functional, while also mutating without being restrained by any layer boundaries.
    pub fn new(inputs: usize, hidden: usize, outputs: usize) -> Self {
        let mut rng = rand::thread_rng();

        let input_layer: Vec<_> = (0..inputs)
            .map(|_| Rc::new(Neuron::new(vec![], &mut rng)))
            .collect();

        let hidden_layers: Vec<_> = (0..hidden)
            .map(|_| Rc::new(Neuron::new(input_layer.clone(), &mut rng)))
            .collect();

        let output_layer: Vec<_> = (0..outputs)
            .map(|_| Rc::new(Neuron::new(hidden_layers.clone(), &mut rng)))
            .collect();

        Self {
            input_layer,
            hidden_layers,
            output_layer,
        }
    }

    /// Runs the neural network based on the given input. **IMPORTANT: you must run [flush_state][NeuralNetwork::flush_state] if you wish to run this network multiple times.**
    /// Input length must be the same as the original one provided to the network.
    pub fn predict(&mut self, inputs: Vec<f32>) -> Vec<f32> {
        if inputs.len() != self.input_layer.len() {
            // TODO comptime input shape? possible with const generics.
            panic!("Invalid inputs length. Expected {}, found {}", self.input_layer.len(), inputs.len());
        }

        for (i, v) in inputs.into_iter().enumerate() {
            let n = Rc::get_mut(&mut self.input_layer[i]).unwrap();
            n.state.value = v;
            n.state.processed = true;
        }

        self.output_layer
            .iter_mut()
            .map(|n| Rc::get_mut(n).unwrap().process())
            .collect()
    }

    /// Flushes the neural network state after a call to [predict][NeuralNetwork::predict].
    pub fn flush_state(&mut self) {
        for n in self.input_layer.iter_mut() {
            Rc::get_mut(n).unwrap().flush_state();
        }

        for n in self.hidden_layers.iter_mut() {
            Rc::get_mut(n).unwrap().flush_state();
        }

        for n in self.output_layer.iter_mut() {
            Rc::get_mut(n).unwrap().flush_state();
        }
    }
}

impl From<&StatelessNeuralNetwork> for NeuralNetwork {
    fn from(value: &StatelessNeuralNetwork) -> Self {
        let input_layer: Vec<_> = value.input_layer
            .iter()
            .map(|n| Rc::new(Neuron {
                inputs: vec![],
                bias: n.bias,
                state: NeuronState::default(),
            }))
            .collect();

        let mut hidden_layers: Vec<Rc<Neuron>> = Vec::with_capacity(value.hidden_layers.len());

        for n in &value.hidden_layers {
            let inputs = n.inputs
                .iter()
                .map(|(n2, w)| match n2.location {
                    NeuronLocator::Input(i) => (input_layer[i].clone(), *w),
                    NeuronLocator::Hidden(i) => (hidden_layers[i].clone(), *w), // TODO fix index errors for neurons that are already loaded. maybe recursive algo?
                    NeuronLocator::Output(_) => panic!("Output layer should never be the input to a neuron"),
                })
                .collect();

            hidden_layers.push(Rc::new(Neuron {
                inputs,
                bias: n.bias,
                state: NeuronState {
                    value: n.bias,
                    ..Default::default()
                }
            }));
        }

        let mut output_layer = Vec::with_capacity(value.output_layer.len());

        for n in &value.output_layer {
            let inputs = n.inputs
                .iter()
                .map(|(n2, w)| match n2.location {
                    NeuronLocator::Input(i) => (input_layer[i].clone(), *w),
                    NeuronLocator::Hidden(i) => (hidden_layers[i].clone(), *w),
                    NeuronLocator::Output(_) => panic!("Output layer should never be the input to a neuron"),
                })
                .collect();

            output_layer.push(Rc::new(Neuron {
                inputs,
                bias: n.bias,
                state: NeuronState {
                    value: n.bias,
                    ..Default::default()
                }
            }));
        }

        Self {
            input_layer,
            hidden_layers,
            output_layer,
        }
    }
}

/// A neuron in the [NeuralNetwork] struct. Holds connections to previous layers and state.
#[derive(Clone, PartialEq)]
pub struct Neuron {
    inputs: Vec<(Rc<Neuron>, f32)>,
    bias: f32,

    /// The state of the neuron. Used in [NeuralNetwork::predict]
    pub state: NeuronState,
}

impl Neuron {
    /// Create a new neuron based on the preceding layer.
    pub fn new(inputs: Vec<Rc<Neuron>>, rng: &mut impl rand::Rng) -> Self {
        let inputs = inputs
            .into_iter()
            .map(|r| (r, rng.gen::<f32>()))
            .collect();

        let bias = rng.gen::<f32>();

        Self {
            inputs,
            state: NeuronState {
                value: bias,
                ..Default::default()
            },
            bias,
        }
    }

    /// Recursively solve the value of this neuron and its predecessors.
    pub fn process(&mut self) -> f32 {
        if self.state.processed {
            return self.state.value;
        }

        for (n, w) in self.inputs.iter_mut() {
            self.state.value += Rc::get_mut(n).unwrap().process() * *w;
        }

        self.state.processed = true;

        self.state.value
    }

    /// Flush the neuron's state. Called by [NeuralNetwork::flush_state]
    pub fn flush_state(&mut self) {
        self.state.value = self.bias;
        self.state.processed = false;
    }
}

/// The state of a neuron.
#[derive(Default, Clone, PartialEq)]
pub struct NeuronState {
    /// The neuron's value. This will likely be initialized by bias unless it is an input neuron.
    pub value: f32,

    /// Whether or not the neuron has been processed already. Used for caching in the recursive algo.
    pub processed: bool,
}

impl GenerateRandom for NetworkWideMutation {
    fn gen_random(rng: &mut impl Rng) -> Self {
        match rng.gen_range(0..3) {
            0 => Self::AddConnection,
            1 => Self::RemoveConnection,
            2 => Self::AddNeuron,
            _ => Self::RemoveNeuron,
        }
    }
}

#[cfg(test)]
mod tests {

}
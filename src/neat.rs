use crate::prelude::*;
use std::rc::Rc;

#[derive(Clone)]
pub struct NeuralNetwork {
    input_layer: Vec<Rc<Neuron>>,
    hidden_layers: Vec<Rc<Neuron>>,
    output_layer: Vec<Rc<Neuron>>,
}

impl NeuralNetwork {
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

    pub fn predict(&mut self, inputs: Vec<f32>) -> Vec<f32> {
        if inputs.len() != self.input_layer.len() {
            panic!("Invalid inputs length. Expected {}, found {}", self.input_layer.len(), inputs.len());
        }

        for (i, v) in inputs.into_iter().enumerate() {
            Rc::get_mut(&mut self.input_layer[i]).unwrap().state.value = v;
        }

        self.output_layer
            .iter_mut()
            .map(|n| Rc::get_mut(n).unwrap().process())
            .collect()
    }

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

    fn rand_neuron_mut(&mut self, rng: &mut impl rand::Rng) -> (&mut Rc<Neuron>, NeuronLocator) {
        if rng.gen::<f32>() <= 0.5 {
            let i = rng.gen_range(0..self.output_layer.len());
            return (&mut self.output_layer[i], NeuronLocator::OutputLayer(i));
        }

        let i = rng.gen_range(0..self.hidden_layers.len());
        (&mut self.hidden_layers[i], NeuronLocator::HiddenLayer(i))
    }


    fn rand_neuron(&self, rng: &mut impl rand::Rng) -> (&Rc<Neuron>, NeuronLocator) {
        if rng.gen::<f32>() <= 0.5 {
            let i = rng.gen_range(0..self.output_layer.len());
            return (&self.output_layer[i], NeuronLocator::OutputLayer(i));
        }

        let i = rng.gen_range(0..self.hidden_layers.len());
        (&self.hidden_layers[i], NeuronLocator::HiddenLayer(i))
    }

    fn is_connection_safe(&self, n1: &Rc<Neuron>, n2: &Rc<Neuron>) -> bool {
        // check if connection is safe (going n2 -> n1 if represented by forward propagation).
        for (n, _w) in &n2.inputs {
            if n == n1 || !self.is_connection_safe(n1, n) {
                return false; // if returned, instantly escape entire recursion.
            }
        }

        true // only returned once it reaches input layer (or some other neuron with no inputs)
    }
}

impl RandomlyMutable for NeuralNetwork {
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
                    NeuronLocator::InputLayer(i) => self.input_layer[i].clone(),
                    NeuronLocator::HiddenLayer(i) => self.hidden_layers[i].clone(),
                    NeuronLocator::OutputLayer(i) => self.output_layer[i].clone(),
                };

                let n1 = match loc1 {
                    NeuronLocator::InputLayer(i) => Rc::get_mut(&mut self.input_layer[i]).unwrap(),
                    NeuronLocator::HiddenLayer(i) => Rc::get_mut(&mut self.hidden_layers[i]).unwrap(),
                    NeuronLocator::OutputLayer(i) => Rc::get_mut(&mut self.output_layer[i]).unwrap(),
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
                let n3 = Rc::new(Neuron::new(vec![n2], rng));

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

impl DivisionReproduction for NeuralNetwork {
    fn spawn_child(&self, rng: &mut impl rand::Rng) -> Self {
        let mut child = self.clone();
        child.mutate(0.01, rng); // TODO customizable rate
        child
    }
}

#[derive(Clone, PartialEq)]
pub struct Neuron {
    inputs: Vec<(Rc<Neuron>, f32)>,
    bias: f32,
    state: NeuronState,
}

impl Neuron {
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

    pub fn flush_state(&mut self) {
        self.state.value = self.bias;
    }
}

#[derive(Default, Clone, PartialEq)]
pub struct NeuronState {
    pub value: f32,
    pub processed: bool,
}

pub enum NetworkWideMutation {
    AddConnection,
    RemoveConnection,
    AddNeuron,
    RemoveNeuron,
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

enum NeuronLocator {
    InputLayer(usize),
    HiddenLayer(usize),
    OutputLayer(usize),
}

#[cfg(test)]
mod tests {

}
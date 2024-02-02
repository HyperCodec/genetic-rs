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
}

impl RandomlyMutable for NeuralNetwork {
    fn mutate(&mut self, rate: f32, rng: &mut impl rand::Rng) {
        todo!();
    }
}

impl DivisionReproduction for NeuralNetwork {
    fn spawn_child(&self, rng: &mut impl rand::Rng) -> Self {
        let mut child = self.clone();
        child.mutate(0.01, rng);
        child
    }
}

#[derive(Clone)]
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

#[derive(Default, Clone)]
pub struct NeuronState {
    pub value: f32,
    pub processed: bool,
}

#[cfg(test)]
mod tests {

}
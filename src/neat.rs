use std::{cell::{Ref, RefCell, RefMut}, rc::Rc, sync::{Mutex, MutexGuard}};

use crate::prelude::*;

#[derive(Clone)]
pub struct StatelessNeuralNetwork {
    input_layer: Vec<StatelessNeuron>,
    hidden_layers: Vec<StatelessNeuron>,
    output_layer: Vec<StatelessNeuron>,
}

impl StatelessNeuralNetwork {
    pub fn new(inputs: usize, hidden: usize, outputs: usize) -> Self {
        let mut rng = rand::thread_rng(); // TODO maybe make a param?

        let input_layer: Vec<_> = (0..inputs)
            .map(|i| StatelessNeuron::new(vec![], NeuronPointer::Input(i), &mut rng))
            .collect();

        let hidden_layers: Vec<_> = (0..hidden)
            .map(|i| StatelessNeuron::new((0..inputs).map(|i| NeuronPointer::Input(i)), NeuronPointer::Hidden(i), &mut rng))
            .collect();

        let output_layer: Vec<_> = (0..outputs)
            .map(|i| StatelessNeuron::new((0..hidden).map(|i| NeuronPointer::Hidden(i)), NeuronPointer::Output(i), &mut rng))
            .collect();

        Self {
            input_layer,
            hidden_layers,
            output_layer,
        }
    }

    fn get_neuron(&self, ptr: NeuronPointer) -> &StatelessNeuron {
        match ptr {
            NeuronPointer::Input(i) => &self.input_layer[i],
            NeuronPointer::Hidden(i) => &self.hidden_layers[i],
            NeuronPointer::Output(i) => &self.output_layer[i],
        }
    }

    fn get_neuron_mut(&mut self, ptr: NeuronPointer) -> &mut StatelessNeuron {
        match ptr {
            NeuronPointer::Input(i) => &mut self.input_layer[i],
            NeuronPointer::Hidden(i) => &mut self.hidden_layers[i],
            NeuronPointer::Output(i) => &mut self.output_layer[i],
        }
    }

    fn rand_neuron_mut(&mut self, rng: &mut impl rand::Rng) -> (&mut StatelessNeuron, NeuronPointer) {
        if rng.gen::<f32>() <= 0.5 {
            let i = rng.gen_range(0..self.output_layer.len());
            return (&mut self.output_layer[i], NeuronPointer::Output(i));
        }

        let i = rng.gen_range(0..self.hidden_layers.len());
        (&mut self.hidden_layers[i], NeuronPointer::Hidden(i))
    }


    fn rand_neuron(&self, rng: &mut impl rand::Rng) -> (&StatelessNeuron, NeuronPointer) {
        if rng.gen::<f32>() <= 0.5 {
            let i = rng.gen_range(0..self.output_layer.len());
            return (&self.output_layer[i], NeuronPointer::Output(i));
        }

        let i = rng.gen_range(0..self.hidden_layers.len());
        (&self.hidden_layers[i], NeuronPointer::Hidden(i))
    }

    fn is_connection_safe(&self, p1: NeuronPointer, p2: NeuronPointer) -> bool {
        // check if connection is safe (going n2 -> n1 if represented by forward propagation).

        if let NeuronPointer::Output(_) = p2 {
            return false;
        }

        let n2 = self.get_neuron(p2);
        for (p, _w) in &n2.inputs {
            if *p == p1 || !self.is_connection_safe(p1, *p) {
                return false; // if returned, instantly escape entire recursion.
            }
        }

        true // only returned once it reaches input layer (or some other neuron with no inputs)
    }

    fn delete_neuron_raw(&mut self, ptr: NeuronPointer) -> Option<StatelessNeuron> {
        if let NeuronPointer::Hidden(i) = ptr {
            let out = self.hidden_layers.remove(i); // still errors if out of bounds

            self.shift_neuron_pointers_for_deletion(ptr);

            return Some(out);
        }

        None // invalid layer
    }

    fn shift_neuron_pointers_for_deletion(&mut self, removed_ptr: NeuronPointer) {
        if let NeuronPointer::Hidden(i) = removed_ptr {
            for n in &mut self.output_layer {
                if let NeuronPointer::Hidden(j) = &mut n.location {
                    if *j >= i {
                        *j -= 1;
                    }
                }

                for (ptr, _w) in &mut n.inputs {
                    if let NeuronPointer::Hidden(j) = ptr {
                        if *j < i {
                            continue;
                        }

                        *j -= 1;
                    }
                }
            }
        }
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
                while !self.is_connection_safe(loc1, loc2) {
                    (n1, loc1) = self.rand_neuron(rng);
                    (n2, loc2) = self.rand_neuron(rng);
                }

                let n1 = self.get_neuron_mut(loc1);

                n1.inputs.push((loc2, rng.gen::<f32>()));
            },
            NetworkWideMutation::RemoveConnection => {
                let n = self.rand_neuron_mut(rng).0;
                n.inputs.remove(rng.gen_range(0..n.inputs.len()));
            },
            NetworkWideMutation::AddNeuron => {
                // split preexisting connection to put new neuron in.
                let (pn, i, n2, w);
                
                {
                    let npn = self.rand_neuron_mut(rng);
                    let n = npn.0;
                    pn = npn.1;

                    i = rng.gen_range(0..n.inputs.len());
                    (n2, w) = n.inputs.remove(i);
                    
                }
                
                let n3 = StatelessNeuron::new(vec![n2], NeuronPointer::Input(i), rng);
                let loc = NeuronPointer::Hidden(self.hidden_layers.len());
                self.hidden_layers.push(n3);


                let n = self.get_neuron_mut(pn);

                n.inputs.push((loc, w));
            },
            NetworkWideMutation::RemoveNeuron => {
                let i = rng.gen_range(0..self.hidden_layers.len());
                let ptr = NeuronPointer::Hidden(i);
                self.delete_neuron_raw(ptr);
            },
        }

        // change weights
        for n in self.hidden_layers.iter_mut() {
            for (_n2, w) in n.inputs.iter_mut() {
                if rng.gen::<f32>() < rate {
                    *w += rng.gen::<f32>() * rate;
                }
            }
        }

        for n in self.output_layer.iter_mut() {
            for (_n2, w) in n.inputs.iter_mut() {
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
    inputs: Vec<(NeuronPointer, f32)>,
    bias: f32,
    location: NeuronPointer,
}

impl StatelessNeuron {
    pub fn new(inputs: impl IntoIterator<Item = NeuronPointer>, location: NeuronPointer, rng: &mut impl rand::Rng) -> Self {
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

#[derive(Clone, Copy, PartialEq)]
pub enum NeuronPointer {
    Input(usize),
    Hidden(usize),
    Output(usize),
}

impl NeuronPointer {
    pub fn is_input(&self) -> bool {
        match self {
            Self::Input(_) => true,
            _ => false,
        }
    }

    pub fn is_hidden(&self) -> bool {
        match self {
            Self::Hidden(_) => true,
            _ => false,
        }
    }

    pub fn is_output(&self) -> bool {
        match self {
            Self::Output(_) => true,
            _ => false,
        }
    }
}

/// A builtin struct that uses the NEAT (Neuro-Evolution Augmented Topology) algorithm.
/// TODO example
#[derive(Clone)]
pub struct NeuralNetwork {
    input_layer: Vec<Rc<Mutex<Neuron>>>,
    hidden_layers: Vec<Rc<Mutex<Neuron>>>,
    output_layer: Vec<Rc<Mutex<Neuron>>>,
}

impl NeuralNetwork {
    /// Creates a simple neural network with 1 hidden layer. This is so that it is still able to be functional, while also mutating without being restrained by any layer boundaries.
    pub fn new(inputs: usize, hidden: usize, outputs: usize) -> Self {
        let mut rng = rand::thread_rng();

        let input_layer: Vec<_> = (0..inputs)
            .map(|_| Rc::new(Mutex::new(Neuron::new(vec![], &mut rng))))
            .collect();

        let hidden_layers: Vec<_> = (0..hidden)
            .map(|_| Rc::new(Mutex::new(Neuron::new((0..inputs).map(|i| NeuronPointer::Input(i)), &mut rng))))
            .collect();

        let output_layer: Vec<_> = (0..outputs)
            .map(|_| Rc::new(Mutex::new(Neuron::new((0..hidden).map(|i| NeuronPointer::Hidden(i)), &mut rng))))
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
            let mut n = self.input_layer[i].try_lock().unwrap();
            n.state.value = v;
            n.state.processed = true;
        }

        let mut outputs = Vec::with_capacity(self.output_layer.len());
        for i in 0..self.output_layer.len() {
            let nrc = Rc::clone(&self.output_layer[i]);
            let mut n = nrc.try_lock().unwrap();
            let mut work = n.inputs.clone();

            while let Some((ptr, w)) = work.pop() {
                let n2rc = self.get_neuron(ptr);
                let n2 = n2rc.try_lock().unwrap(); // cause of hang

                if n2.state.processed {
                    n.state.value += n2.state.value * w;
                }
            }

            n.state.processed = true;

            outputs.push(n.state.value);
        }

        outputs
    }

    /// Flushes the neural network state after a call to [predict][NeuralNetwork::predict].
    pub fn flush_state(&mut self) {
        for n in &self.input_layer {
            n.lock().unwrap().flush_state();
        }

        for n in &self.hidden_layers {
            n.lock().unwrap().flush_state();
        }

        for n in &self.output_layer {
            n.lock().unwrap().flush_state();
        }
    }

    // TODO merge mut and normal version (also probably just return mutex instead of guard, safer that way)
    fn get_neuron(&self, ptr: NeuronPointer) -> Rc<Mutex<Neuron>> {
        match ptr {
            NeuronPointer::Input(i) => self.input_layer[i].clone(),
            NeuronPointer::Hidden(i) => self.hidden_layers[i].clone(),
            NeuronPointer::Output(i) => self.output_layer[i].clone(),
        }
    }
}

impl From<&StatelessNeuralNetwork> for NeuralNetwork {
    fn from(value: &StatelessNeuralNetwork) -> Self {
        let input_layer = value.input_layer
            .iter()
            .map(Neuron::from)
            .map(Mutex::new)
            .map(Rc::new)
            .collect();

        let hidden_layers = value.hidden_layers
            .iter()
            .map(Neuron::from)
            .map(Mutex::new)
            .map(Rc::new)
            .collect();

        let output_layer = value.output_layer
            .iter()
            .map(Neuron::from)
            .map(Mutex::new)
            .map(Rc::new)
            .collect();

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
    inputs: Vec<(NeuronPointer, f32)>,
    bias: f32,

    /// The state of the neuron. Used in [NeuralNetwork::predict]
    pub state: NeuronState,
}

impl Neuron {
    /// Create a new neuron based on the preceding layer.
    pub fn new(inputs: impl IntoIterator<Item = NeuronPointer>, rng: &mut impl rand::Rng) -> Self {
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

    /// Flush the neuron's state. Called by [NeuralNetwork::flush_state]
    pub fn flush_state(&mut self) {
        self.state.value = self.bias;
        self.state.processed = false;
    }
}

impl From<&StatelessNeuron> for Neuron {
    fn from(value: &StatelessNeuron) -> Self {
        Self {
            inputs: value.inputs.clone(),
            bias: value.bias,
            state: NeuronState {
                value: value.bias,
                ..Default::default()
            }
        }
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